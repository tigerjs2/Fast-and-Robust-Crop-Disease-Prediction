import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.append(str(PROJECT_ROOT))

DEFAULT_SECOND_STAGE_WEIGHTS = PROJECT_ROOT / "weights" / "mob3_1" / "best_classifier.pth"
DEFAULT_SAM_WEIGHTS = PROJECT_ROOT / "weights" / "sam2.1_t.pt"
DEFAULT_TEXT_EMBEDDING_DIR = PROJECT_ROOT / "data" / "TextEmbeddings"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "mobile_assets"


def export_second_stage_checkpoint(checkpoint_path: Path, output_dir: Path) -> Tuple[Path, Path, Path]:
	from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
	from executorch.exir import to_edge_transform_and_lower

	from scripts.SecondStageTest import build_model_from_checkpoint, load_checkpoint

	checkpoint = load_checkpoint(checkpoint_path, device=torch.device("cpu"))
	class_names = checkpoint.get("class_names", [])
	if not class_names:
		raise KeyError("Checkpoint does not contain class_names")

	config = checkpoint.get("config", {})
	image_size = int(config.get("image_size", 224))
	embed_dim = int(config.get("embed_dim", 512))

	model = build_model_from_checkpoint(checkpoint, num_classes=len(class_names), device=torch.device("cpu"))
	model.eval()

	sample_inputs = (
		torch.randn(1, 3, image_size, image_size),
		torch.randn(1, 5, embed_dim),
	)

	et_program = to_edge_transform_and_lower(
		torch.export.export(model, sample_inputs),
		partitioner=[XnnpackPartitioner()],
	).to_executorch()

	output_dir.mkdir(parents=True, exist_ok=True)
	pte_path = output_dir / "second_stage.pte"
	with pte_path.open("wb") as f:
		f.write(et_program.buffer)

	class_names_path = output_dir / "class_names.json"
	class_names_path.write_text(json.dumps(class_names, indent=2), encoding="utf-8")

	config_path = output_dir / "second_stage_config.json"
	config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

	return pte_path, class_names_path, config_path


def export_text_embeddings(text_embedding_dir: Path, output_dir: Path) -> Tuple[Path, Path]:
	output_dir.mkdir(parents=True, exist_ok=True)
	embedding_dir = output_dir / "embeddings"
	embedding_dir.mkdir(parents=True, exist_ok=True)

	crop_names: List[str] = []
	shape_info: Dict[str, int] = {}

	for emb_path in sorted(text_embedding_dir.glob("*.pt")):
		crop = emb_path.stem.lower()
		emb = torch.load(str(emb_path), map_location="cpu")
		if not isinstance(emb, torch.Tensor):
			raise TypeError(f"Embedding file must contain a torch.Tensor: {emb_path}")
		if emb.dim() != 2:
			raise ValueError(f"Embedding tensor must be 2D (num_queries, dim): {emb_path}")

		emb_np = emb.detach().cpu().float().numpy()
		out_path = embedding_dir / f"{crop}.npy"
		np.save(out_path, emb_np)

		crop_names.append(crop)
		shape_info["num_queries"] = int(emb_np.shape[0])
		shape_info["embed_dim"] = int(emb_np.shape[1])

	if not crop_names:
		raise FileNotFoundError(f"No .pt embeddings found in {text_embedding_dir}")

	crops_path = output_dir / "crops.json"
	crops_path.write_text(json.dumps(sorted(crop_names), indent=2), encoding="utf-8")

	shape_path = output_dir / "embedding_shape.json"
	shape_path.write_text(json.dumps(shape_info, indent=2), encoding="utf-8")

	return crops_path, shape_path


def export_sam_weights(sam_weights: Path, output_dir: Path) -> Path:
	output_dir.mkdir(parents=True, exist_ok=True)
	out_path = output_dir / sam_weights.name
	shutil.copy2(sam_weights, out_path)
	note_path = output_dir / "sam_export_note.txt"
	note_path.write_text(
		"SAM2 weights copied only. Export to ExecuTorch is not implemented in this repo.\n"
		"You need a mobile-compatible SAM model (e.g., custom export or lighter model)\n"
		"that accepts image + bbox and returns a mask for on-device inference.\n",
		encoding="utf-8",
	)
	return out_path


def main() -> None:
	parser = argparse.ArgumentParser(description="Export mobile assets for ExecuTorch")
	parser.add_argument("--second_stage_ckpt", type=str, default=str(DEFAULT_SECOND_STAGE_WEIGHTS))
	parser.add_argument("--sam_weights", type=str, default=str(DEFAULT_SAM_WEIGHTS))
	parser.add_argument("--text_embedding_dir", type=str, default=str(DEFAULT_TEXT_EMBEDDING_DIR))
	parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
	parser.add_argument("--skip_sam", action="store_true", help="Skip SAM weight copy")
	args = parser.parse_args()

	output_dir = Path(args.output_dir)
	pte_path, class_names_path, config_path = export_second_stage_checkpoint(
		checkpoint_path=Path(args.second_stage_ckpt),
		output_dir=output_dir,
	)
	crops_path, shape_path = export_text_embeddings(
		text_embedding_dir=Path(args.text_embedding_dir),
		output_dir=output_dir,
	)

	print(f"Saved second-stage pte: {pte_path}")
	print(f"Saved class names: {class_names_path}")
	print(f"Saved config: {config_path}")
	print(f"Saved crops list: {crops_path}")
	print(f"Saved embedding shape: {shape_path}")

	if not args.skip_sam:
		sam_path = export_sam_weights(Path(args.sam_weights), output_dir=output_dir)
		print(f"Copied SAM weights: {sam_path}")


if __name__ == "__main__":
	main()
