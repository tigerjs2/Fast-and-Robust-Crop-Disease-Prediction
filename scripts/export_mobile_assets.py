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

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExportableCrossAttention(nn.Module):
    """
    nn.MultiheadAttention(batch_first=True) 와 수치적으로 동등하지만
    torch.export + XNNPACK 에 안전한 manual 구현.
    학습된 nn.MultiheadAttention 가중치를 그대로 로드 가능.
    """
    def __init__(self, embed_dim: int, num_heads: int, residual_scale: float = 1.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.residual_scale = residual_scale

        # nn.MultiheadAttention 의 in_proj_weight (3*E, E), in_proj_bias (3*E,) 와 호환
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, image_features: torch.Tensor, text_queries: torch.Tensor) -> torch.Tensor:
        # image_features: (B, N, E), text_queries: (B, M, E)
        B, N, E = image_features.shape
        M = text_queries.shape[1]
        H, D = self.num_heads, self.head_dim

        # Q from image, K/V from text  — nn.MultiheadAttention 와 동일한 분할
        Wq, Wk, Wv = self.in_proj_weight.chunk(3, dim=0)   # (E,E) x 3
        bq, bk, bv = self.in_proj_bias.chunk(3, dim=0)     # (E,)  x 3

        q = F.linear(image_features, Wq, bq)   # (B, N, E)
        k = F.linear(text_queries,  Wk, bk)    # (B, M, E)
        v = F.linear(text_queries,  Wv, bv)    # (B, M, E)

        # (B, H, N, D)
        q = q.reshape(B, N, H, D).transpose(1, 2)
        k = k.reshape(B, M, H, D).transpose(1, 2)
        v = v.reshape(B, M, H, D).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale   # (B, H, N, M)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)                                # (B, H, N, D)
        out = out.transpose(1, 2).reshape(B, N, E)                 # (B, N, E)
        out = self.out_proj(out)                                   # (B, N, E)

        return self.norm(image_features + self.residual_scale * out)


def convert_mha_to_exportable(model: nn.Module) -> nn.Module:
    """
    SecondStageClassifier 내부의 CrossAttentionModule 을
    ExportableCrossAttention 으로 in-place 교체하면서 가중치를 그대로 옮긴다.
    """
    ca = model.cross_attention            # 기존 CrossAttentionModule
    mha = ca.multihead_attn               # nn.MultiheadAttention
    embed_dim = mha.embed_dim
    num_heads = mha.num_heads

    new_ca = ExportableCrossAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        residual_scale=getattr(ca, "residual_scale", 1.0),
    )

    with torch.no_grad():
        # nn.MultiheadAttention의 가중치 명명을 그대로 매핑
        new_ca.in_proj_weight.copy_(mha.in_proj_weight)
        new_ca.in_proj_bias.copy_(mha.in_proj_bias)
        new_ca.out_proj.weight.copy_(mha.out_proj.weight)
        new_ca.out_proj.bias.copy_(mha.out_proj.bias)
        new_ca.norm.weight.copy_(ca.norm.weight)
        new_ca.norm.bias.copy_(ca.norm.bias)

    model.cross_attention = new_ca
    return model



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
	"""
	model = convert_mha_to_exportable(model)   # ← 추가

	# 그 다음 PyTorch ↔ Export 일치 확인 (강력 권장)
	with torch.no_grad():
		img = torch.randn(1, 3, image_size, image_size)
		txt = torch.randn(1, 5, embed_dim)
		ref_logits = model(img, txt)
	"""
	sample_inputs = (
		torch.randn(1, 3, image_size, image_size),
		torch.randn(1, 5, embed_dim),
	)
	#sample_inputs = (img, txt)
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
	# export_mobile_assets.py 의 export_second_stage_checkpoint 끝부분에 추가
	from executorch.runtime import Runtime

	# (1) PyTorch 원본 출력
	with torch.no_grad():
		pt_out = model(sample_inputs[0], sample_inputs[1])

	# (2) ExecuTorch 출력 (방금 export한 buffer 재로드)
	runtime = Runtime.get()
	program = runtime.load_program(et_program.buffer)
	method = program.load_method("forward")
	et_out = method.execute([sample_inputs[0], sample_inputs[1]])[0]

	diff = (pt_out - et_out).abs()
	print(f"max diff: {diff.max().item():.6e}")
	print(f"mean diff: {diff.mean().item():.6e}")
	print(f"argmax match: {(pt_out.argmax(-1) == et_out.argmax(-1)).all().item()}")
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
