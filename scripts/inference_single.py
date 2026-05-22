import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.append(str(PROJECT_ROOT))

from base_model.SAM import SAM2MaskGenerator
from scripts.SecondStageTest import build_model_from_checkpoint, load_checkpoint
from scripts.SecondStageTrain import build_image_transform, load_crop_text_embeddings


DEFAULT_SAM_WEIGHTS = PROJECT_ROOT / "weights" / "sam2.1_t.pt"
DEFAULT_SECOND_STAGE_WEIGHTS = PROJECT_ROOT / "weights" / "mob3_1" / "best_classifier.pth"
DEFAULT_TEXT_EMBEDDING_DIR = PROJECT_ROOT / "data" / "TextEmbeddings"


def _resolve_device(device: Optional[str]) -> torch.device:
	if device:
		return torch.device(device)
	return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _normalize_crop_name(crop_name: str) -> str:
	return crop_name.strip().lower()


def _build_text_queries(
	embedding_map: Dict[str, torch.Tensor],
	crop_name: str,
	device: torch.device,
) -> torch.Tensor:
	if crop_name not in embedding_map:
		available = ", ".join(sorted(embedding_map.keys()))
		raise KeyError(f"Text embedding not loaded for crop: {crop_name}. Available: {available}")
	return embedding_map[crop_name].unsqueeze(0).to(device)


def run_single_inference(
	image_path: str,
	crop_name: str,
	bbox_xyxy: Tuple[int, int, int, int],
	sam_weights: str = str(DEFAULT_SAM_WEIGHTS),
	second_stage_weights: str = str(DEFAULT_SECOND_STAGE_WEIGHTS),
	text_embedding_dir: str = str(DEFAULT_TEXT_EMBEDDING_DIR),
	device: Optional[str] = None,
) -> Dict[str, object]:
	device_t = _resolve_device(device)

	checkpoint = load_checkpoint(Path(second_stage_weights), device_t)
	class_names: Sequence[str] = checkpoint.get("class_names", [])
	if not class_names:
		raise KeyError("Checkpoint does not contain class_names")

	model = build_model_from_checkpoint(checkpoint, num_classes=len(class_names), device=device_t)
	model.eval()

	config = checkpoint.get("config", {})
	image_size = int(config.get("image_size", 224))
	transform = build_image_transform(image_size=image_size, apply_augmentation=False)

	crop_name = _normalize_crop_name(crop_name)
	embedding_map = load_crop_text_embeddings(
		text_embedding_dir=Path(text_embedding_dir),
		crop_names=[crop_name],
		device=device_t,
		expected_num_queries=5,
	)
	text_queries = _build_text_queries(embedding_map, crop_name, device_t)

	sam_generator = SAM2MaskGenerator(model_path=sam_weights, mode="box")
	masked_bgr, region_info, mask_bool = sam_generator.generate_with_region(
		image_path=image_path,
		bbox=bbox_xyxy,
	)

	masked_rgb = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2RGB)
	image_tensor = transform(masked_rgb).unsqueeze(0).to(device_t)

	with torch.no_grad():
		logits = model(image_tensor, text_queries)
		probs = torch.softmax(logits, dim=1)
		pred_idx = int(probs.argmax(dim=1).item())
		pred_conf = float(probs[0, pred_idx].item())

	return {
		"pred_index": pred_idx,
		"pred_class": class_names[pred_idx],
		"pred_confidence": pred_conf,
		"class_names": list(class_names),
		"probs": probs.squeeze(0).detach().cpu().numpy(),
		"masked_bgr": masked_bgr,
		"region_info": region_info,
		"mask_bool": mask_bool,
	}


def _parse_bbox_xyxy(values: Sequence[str]) -> Tuple[int, int, int, int]:
	if len(values) != 4:
		raise ValueError("bbox must contain 4 values: x1 y1 x2 y2")
	try:
		coords = [int(float(v)) for v in values]
	except ValueError as e:
		raise ValueError("bbox values must be numeric") from e
	return coords[0], coords[1], coords[2], coords[3]


def main() -> None:
	parser = argparse.ArgumentParser(description="Single-image inference with SAM box + second-stage classifier")
	parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
	parser.add_argument("--crop", type=str, required=True, help="Crop name (e.g., cucumber, tomato)")
	parser.add_argument(
		"--bbox",
		type=str,
		required=True,
		nargs=4,
		help="Bounding box in xyxy format: x1 y1 x2 y2",
	)
	parser.add_argument("--sam_weights", type=str, default=str(DEFAULT_SAM_WEIGHTS), help="SAM weights path")
	parser.add_argument(
		"--second_stage_weights",
		type=str,
		default=str(DEFAULT_SECOND_STAGE_WEIGHTS),
		help="Second-stage model checkpoint path",
	)
	parser.add_argument(
		"--text_embedding_dir",
		type=str,
		default=str(DEFAULT_TEXT_EMBEDDING_DIR),
		help="Directory with crop text embeddings",
	)
	parser.add_argument("--device", type=str, default=None, help="cuda or cpu")
	parser.add_argument(
		"--save_masked",
		type=str,
		default=None,
		help="Optional output path to save masked image",
	)
	args = parser.parse_args()

	bbox_xyxy = _parse_bbox_xyxy(args.bbox)
	result = run_single_inference(
		image_path=args.image_path,
		crop_name=args.crop,
		bbox_xyxy=bbox_xyxy,
		sam_weights=args.sam_weights,
		second_stage_weights=args.second_stage_weights,
		text_embedding_dir=args.text_embedding_dir,
		device=args.device,
	)

	print(f"Predicted class: {result['pred_class']}")
	print(f"Confidence: {result['pred_confidence']:.6f}")
	print(f"Region info: {result['region_info']}")

	if args.save_masked:
		ok = cv2.imwrite(args.save_masked, result["masked_bgr"])
		if not ok:
			raise RuntimeError(f"Failed to save masked image: {args.save_masked}")
		print(f"Saved masked image: {args.save_masked}")


if __name__ == "__main__":
	main()
