import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.append(str(PROJECT_ROOT))


DEFAULT_SAM_WEIGHTS = PROJECT_ROOT / "weights" / "sam2.1_t.pt"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "weights" / "sam2.1_t_box.pte"


def _try_import_executorch_to_edge():
	"""ExecuTorch 버전 차이를 흡수해 to_edge 함수를 가져온다."""
	try:
		from executorch.exir import to_edge  # type: ignore

		return to_edge
	except Exception:
		try:
			from executorch import exir  # type: ignore

			return exir.to_edge
		except Exception as exc:
			raise ImportError(
				"Failed to import ExecuTorch export API. Install a compatible executorch package first."
			) from exc


def _default_output_path(weights_path: Path) -> Path:
	return weights_path.with_suffix(".pte")


def _load_sam_model(weights_path: Path):
	from ultralytics import SAM

	model = SAM(str(weights_path))
	if not hasattr(model, "model"):
		raise RuntimeError("Ultralytics SAM object does not expose the underlying torch model.")
	return model.model.eval()


def _get_image_size(model: nn.Module, fallback: int = 1024) -> int:
	image_size = getattr(model, "image_size", None)
	if isinstance(image_size, int) and image_size > 0:
		return int(image_size)
	return fallback


class ContiguousWrapper(nn.Module):
	"""Wrap a model to ensure all outputs are contiguous, avoiding dim_order_ops in ExecuTorch.

	This wrapper makes tensor outputs memory-contiguous and clears striding metadata,
	which reduces the likelihood of ExecuTorch generating custom dim_order operators
	that may not be available in mobile runtimes.
	"""
	def __init__(self, model: nn.Module):
		super().__init__()
		self.model = model

	def forward(self, *args, **kwargs):
		out = self.model(*args, **kwargs)

		def make_contiguous(x: Any) -> Any:
			if isinstance(x, torch.Tensor):
				# Ensure contiguous layout; this also clears dim_order metadata
				return x.contiguous().clone()
			return x

		if isinstance(out, torch.Tensor):
			return make_contiguous(out)
		elif isinstance(out, (list, tuple)):
			return type(out)(make_contiguous(x) for x in out)
		elif isinstance(out, dict):
			return {k: make_contiguous(v) for k, v in out.items()}
		else:
			return out


class SAM2BoxExportWrapper(nn.Module):
	"""Wrap SAM2.1 into a box-prompted tensor-only export graph.

	The exported contract is:
	- images: float32 tensor of shape (B, 3, H, W)
	- boxes: float32 tensor of shape (B, 4) in xyxy pixel coordinates
	- output: float32 tensor of shape (B, 1, H, W) with a binary mask
	"""

	def __init__(
		self,
		sam_model: nn.Module,
		image_size: int,
		mask_threshold: float = 0.5,
		multimask_output: bool = False,
	):
		super().__init__()
		self.sam_model = sam_model
		self.image_size = int(image_size)
		self.mask_threshold = float(mask_threshold)
		self.multimask_output = bool(multimask_output)

	def forward(self, images: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
		backbone = self.sam_model.forward_image(images)
		image_embeddings = backbone["vision_features"]
		high_res_features = backbone["backbone_fpn"][:2]

		sparse_embeddings, dense_embeddings = self.sam_model.sam_prompt_encoder(
			points=None,
			boxes=boxes,
			masks=None,
		)

		low_res_multimasks, ious, _, _ = self.sam_model.sam_mask_decoder(
			image_embeddings=image_embeddings,
			image_pe=self.sam_model.sam_prompt_encoder.get_dense_pe(),
			sparse_prompt_embeddings=sparse_embeddings,
			dense_prompt_embeddings=dense_embeddings,
			multimask_output=self.multimask_output,
			repeat_image=False,
			high_res_features=high_res_features,
		)

		if self.multimask_output:
			best_indices = torch.argmax(ious, dim=-1)
			batch_indices = torch.arange(low_res_multimasks.size(0), device=low_res_multimasks.device)
			selected_low_res = low_res_multimasks[batch_indices, best_indices].unsqueeze(1)
		else:
			selected_low_res = low_res_multimasks

		selected_high_res = F.interpolate(
			selected_low_res,
			size=(self.image_size, self.image_size),
			mode="bilinear",
			align_corners=False,
		)
		mask_probs = torch.sigmoid(selected_high_res)
		return (mask_probs >= self.mask_threshold).to(dtype=torch.float32)


def build_export_model(
	weights_path: Path,
	mask_threshold: float,
	multimask_output: bool,
) -> Tuple[nn.Module, int]:
	sam_model = _load_sam_model(weights_path)
	image_size = _get_image_size(sam_model)
	wrapper = SAM2BoxExportWrapper(
		sam_model=sam_model,
		image_size=image_size,
		mask_threshold=mask_threshold,
		multimask_output=multimask_output,
	)
	# Wrap with ContiguousWrapper to avoid dim_order_ops in ExecuTorch
	contiguous_wrapper = ContiguousWrapper(wrapper)
	return contiguous_wrapper, image_size


def export_sam_to_pte(
	weights_path: Path,
	output_path: Path,
	image_size: int,
	batch_size: int,
	strict_export: bool,
	mask_threshold: float,
	multimask_output: bool,
) -> None:
	model, inferred_image_size = build_export_model(
		weights_path=weights_path,
		mask_threshold=mask_threshold,
		multimask_output=multimask_output,
	)
	if image_size <= 0:
		image_size = inferred_image_size

	sample_images = torch.randn(batch_size, 3, image_size, image_size, dtype=torch.float32)
	sample_boxes = torch.tensor(
		[[0.0, 0.0, float(image_size - 1), float(image_size - 1)] for _ in range(batch_size)],
		dtype=torch.float32,
	)

	# Use non-strict mode by default to avoid excessive IR verification
	# that may reject valid inference patterns. strict_export flag overrides.
	effective_strict = strict_export if strict_export else False
	print(f"[samConverter] torch.export with strict={effective_strict}")

	exported_program = torch.export.export(
		model,
		(sample_images, sample_boxes),
		strict=effective_strict,
	)

	print(f"[samConverter] torch.export succeeded. Converting to ExecuTorch...")
	to_edge = _try_import_executorch_to_edge()
	edge_program = to_edge(exported_program)
	executorch_program = edge_program.to_executorch()

	program_buffer = getattr(executorch_program, "buffer", None)
	if program_buffer is None:
		raise RuntimeError("Failed to access ExecuTorch program buffer.")

	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("wb") as f:
		f.write(program_buffer)


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Convert sam2.1_t.pt to ExecuTorch .pte using box prompt export")
	parser.add_argument(
		"--sam_weights",
		type=str,
		default=str(DEFAULT_SAM_WEIGHTS),
		help="Path to sam2.1_t.pt weights",
	)
	parser.add_argument(
		"--output_path",
		type=str,
		default=None,
		help="Output .pte path (default: same name as weights with .pte)",
	)
	parser.add_argument(
		"--image_size",
		type=int,
		default=1024,
		help="Dummy image size used for export tracing",
	)
	parser.add_argument(
		"--batch_size",
		type=int,
		default=1,
		help="Dummy batch size used for export tracing",
	)
	parser.add_argument(
		"--mask_threshold",
		type=float,
		default=0.5,
		help="Threshold used to convert mask probabilities into a binary output",
	)
	parser.add_argument(
		"--multimask_output",
		action="store_true",
		help="Export the multimask branch and pick the best IoU mask at runtime",
	)
	parser.add_argument(
		"--strict_export",
		action="store_true",
		help="Use strict mode in torch.export.export",
	)
	return parser


def main() -> None:
	parser = build_arg_parser()
	args = parser.parse_args()

	weights_path = Path(args.sam_weights)
	if not weights_path.exists():
		raise FileNotFoundError(f"SAM weights not found: {weights_path}")

	output_path = Path(args.output_path) if args.output_path else _default_output_path(weights_path)

	export_sam_to_pte(
		weights_path=weights_path,
		output_path=output_path,
		image_size=args.image_size,
		batch_size=args.batch_size,
		strict_export=args.strict_export,
		mask_threshold=args.mask_threshold,
		multimask_output=args.multimask_output,
	)

	print(f"Converted SAM weights: {weights_path}")
	print(f"Saved ExecuTorch program: {output_path}")


if __name__ == "__main__":
	main()
