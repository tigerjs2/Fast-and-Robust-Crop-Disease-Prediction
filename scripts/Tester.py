import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import sys

import cv2
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.append(str(PROJECT_ROOT))

try:
	from executorch.runtime import Runtime
except Exception as exc:
	raise ImportError("ExecuTorch runtime is required to run Tester.py") from exc

from scripts.SecondStageTest import build_model_from_checkpoint, load_checkpoint

DEFAULT_TEST_DIR = PROJECT_ROOT / "data" / "AIHub_box" / "test"
DEFAULT_ASSETS_DIR = PROJECT_ROOT / "mobile_assets"
DEFAULT_SECOND_STAGE_PTE = DEFAULT_ASSETS_DIR / "second_stage.pte"
DEFAULT_SAM_ENCODER_PTE = DEFAULT_ASSETS_DIR / "sam2.1_t_encoder.pte"
DEFAULT_SAM_DECODER_PTE = DEFAULT_ASSETS_DIR / "sam2.1_t_box_decoder_512.pte"
DEFAULT_CLASS_NAMES = DEFAULT_ASSETS_DIR / "class_names.json"
DEFAULT_CROPS = DEFAULT_ASSETS_DIR / "crops.json"
DEFAULT_EMBEDDING_SHAPE = DEFAULT_ASSETS_DIR / "embedding_shape.json"
DEFAULT_EMBEDDING_DIR = DEFAULT_ASSETS_DIR / "embeddings"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

NORM_MEAN_RGB = np.array([0.485, 0.456, 0.406], dtype=np.float32)
NORM_STD_RGB = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SAM_MEAN_RGB = np.array([123.675, 116.28, 103.53], dtype=np.float32)
SAM_STD_RGB = np.array([58.395, 57.12, 57.375], dtype=np.float32)


class ExecuTorchModule:
	def __init__(self, pte_path: Path):
		if not pte_path.exists():
			raise FileNotFoundError(f"PTE not found: {pte_path}")
		runtime = Runtime.get()
		self.program = runtime.load_program(str(pte_path))
		self.method = self.program.load_method("forward")

	def forward(self, *inputs: torch.Tensor):
		return _run_method(self.method, inputs)


def _run_method(method, inputs):
	if hasattr(method, "__call__"):
		try:
			return method(*inputs)
		except TypeError:
			pass
	if hasattr(method, "execute"):
		return method.execute(list(inputs))
	if hasattr(method, "run"):
		return method.run(*inputs)
	raise RuntimeError("Unsupported ExecuTorch method interface")


def _normalize_outputs(outputs) -> List[torch.Tensor]:
	if isinstance(outputs, (list, tuple)):
		return list(outputs)
	return [outputs]


def _read_json_list(path: Path) -> List[str]:
	with path.open("r", encoding="utf-8") as f:
		data = json.load(f)
	if not isinstance(data, list):
		raise ValueError(f"Expected JSON array at: {path}")
	return [str(x) for x in data]


def _read_embedding_shape(path: Path) -> Tuple[int, int]:
	with path.open("r", encoding="utf-8") as f:
		obj = json.load(f)
	return int(obj["num_queries"]), int(obj["embed_dim"])


def _load_embeddings(crops_path: Path, embedding_dir: Path) -> Dict[str, np.ndarray]:
	crops = _read_json_list(crops_path)
	embeddings: Dict[str, np.ndarray] = {}
	for crop in crops:
		npy_path = embedding_dir / f"{crop}.npy"
		if not npy_path.exists():
			raise FileNotFoundError(f"Missing embedding: {npy_path}")
		arr = np.load(str(npy_path))
		if arr.dtype != np.float32:
			arr = arr.astype(np.float32)
		embeddings[crop.lower()] = arr
	return embeddings


def _discover_dataset(test_dir: Path, class_names: Sequence[str]) -> List[Tuple[Path, int, str]]:
	class_to_idx = {name: idx for idx, name in enumerate(class_names)}
	items: List[Tuple[Path, int, str]] = []
	for class_name in class_names:
		class_dir = test_dir / class_name
		if not class_dir.is_dir():
			continue
		for ext in IMAGE_EXTENSIONS:
			for image_path in class_dir.rglob(f"*{ext}"):
				items.append((image_path, class_to_idx[class_name], class_name))
	if not items:
		raise RuntimeError(f"No images found in: {test_dir}")
	return items


def _resize_and_normalize(img_rgb: np.ndarray, image_size: int) -> np.ndarray:
	resized = cv2.resize(img_rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
	arr = resized.astype(np.float32) / 255.0
	arr = (arr - NORM_MEAN_RGB) / NORM_STD_RGB
	return arr


def _to_chw_tensor(img_rgb: np.ndarray) -> torch.Tensor:
	chw = np.transpose(img_rgb, (2, 0, 1))
	return torch.from_numpy(chw).unsqueeze(0)


def _to_sam_tensor(img_rgb: np.ndarray, sam_input_size: int) -> torch.Tensor:
	resized = cv2.resize(img_rgb, (sam_input_size, sam_input_size), interpolation=cv2.INTER_AREA)
	arr = resized.astype(np.float32)
	arr = (arr - SAM_MEAN_RGB) / SAM_STD_RGB
	chw = np.transpose(arr, (2, 0, 1))
	return torch.from_numpy(chw).unsqueeze(0)


def _scale_bbox(
	bbox_xyxy: np.ndarray,
	src_w: int,
	src_h: int,
	dst_w: int,
	dst_h: int,
) -> np.ndarray:
	scale_x = float(dst_w) / max(src_w, 1)
	scale_y = float(dst_h) / max(src_h, 1)
	x1, y1, x2, y2 = bbox_xyxy.tolist()
	x1 = min(max(x1 * scale_x, 0.0), float(dst_w - 1))
	y1 = min(max(y1 * scale_y, 0.0), float(dst_h - 1))
	x2 = min(max(x2 * scale_x, 0.0), float(dst_w))
	y2 = min(max(y2 * scale_y, 0.0), float(dst_h))
	return np.array([x1, y1, x2, y2], dtype=np.float32)


def _to_xyxy_clipped(
	x: float,
	y: float,
	w: float,
	h: float,
	width: int,
	height: int,
) -> np.ndarray:
	x1 = int(round(x))
	y1 = int(round(y))
	x2 = int(round(x + w))
	y2 = int(round(y + h))

	x1 = max(0, min(width - 1, x1))
	y1 = max(0, min(height - 1, y1))
	x2 = max(0, min(width, x2))
	y2 = max(0, min(height, y2))
	return np.array([x1, y1, x2, y2], dtype=np.float32)


def _find_bbox_json(image_path: Path, test_dir: Path, bbox_dir: Path) -> Path:
	try:
		rel = image_path.relative_to(test_dir)
		return (bbox_dir / rel).with_suffix(".json")
	except ValueError:
		return image_path.with_suffix(".json")


def _build_masked_output_path(image_path: Path, test_dir: Path, output_dir: Path) -> Path:
	try:
		rel = image_path.relative_to(test_dir)
		return output_dir / rel
	except ValueError:
		return output_dir / image_path.name


def _draw_bbox_on_image(img_rgb: np.ndarray, bbox_xyxy: np.ndarray) -> np.ndarray:
	x1, y1, x2, y2 = bbox_xyxy.astype(int).tolist()
	output = img_rgb.copy()
	color = (0, 255, 0)
	thickness = max(2, int(round(min(output.shape[0], output.shape[1]) * 0.003)))
	cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
	return output


def _load_bbox_from_json(json_path: Path, img_rgb: np.ndarray, bbox_source: str) -> np.ndarray:
	if not json_path.exists():
		raise FileNotFoundError(f"Missing bbox annotation: {json_path}")
	with json_path.open("r", encoding="utf-8") as f:
		data = json.load(f)
	ann = data.get("annotations", {})
	parts = ann.get("part", []) if bbox_source == "part_union" else []
	if parts:
		xs = [float(p.get("x", 0.0)) for p in parts]
		ys = [float(p.get("y", 0.0)) for p in parts]
		ws = [float(p.get("w", 0.0)) for p in parts]
		hs = [float(p.get("h", 0.0)) for p in parts]
		x1 = min(xs)
		y1 = min(ys)
		x2 = max(x + w for x, w in zip(xs, ws))
		y2 = max(y + h for y, h in zip(ys, hs))
		height, width = img_rgb.shape[:2]
		return _to_xyxy_clipped(x1, y1, x2 - x1, y2 - y1, width=width, height=height)

	bbox_list = ann.get("bbox", [])
	if not bbox_list:
		raise RuntimeError(f"No bbox in annotation: {json_path}")
	bbox = bbox_list[0]
	x = float(bbox.get("x", 0.0))
	y = float(bbox.get("y", 0.0))
	w = float(bbox.get("w", 0.0))
	h = float(bbox.get("h", 0.0))
	height, width = img_rgb.shape[:2]
	return _to_xyxy_clipped(x, y, w, h, width=width, height=height)


def _format_bbox_debug(
	json_path: Path,
	img_rgb: np.ndarray,
	bbox_source: str,
	bbox_xyxy: np.ndarray,
) -> str:
	h, w = img_rgb.shape[:2]
	x1, y1, x2, y2 = bbox_xyxy.tolist()
	return (
		f"[bbox-debug] json={json_path} source={bbox_source} "
		f"img_w={w} img_h={h} xyxy=({x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f})"
	)


def _format_tensor_stats(name: str, tensor: torch.Tensor) -> str:
	stats = {
		"min": float(tensor.min().item()),
		"max": float(tensor.max().item()),
		"mean": float(tensor.mean().item()),
		"std": float(tensor.std().item()),
	}
	return (
		f"[stats] {name} min={stats['min']:.6f} max={stats['max']:.6f} "
		f"mean={stats['mean']:.6f} std={stats['std']:.6f}"
	)


def _select_sam_features(outputs: Sequence[torch.Tensor], sam_input_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	image_embed: Optional[torch.Tensor] = None
	feat_s0: Optional[torch.Tensor] = None
	feat_s1: Optional[torch.Tensor] = None
	expected_s0 = 256 if sam_input_size >= 1024 else 128
	expected_s1 = 128 if sam_input_size >= 1024 else 64

	for tensor in outputs:
		if not isinstance(tensor, torch.Tensor):
			tensor = torch.as_tensor(tensor)
		shape = list(tensor.shape)
		if len(shape) == 4:
			h = int(shape[2])
			w = int(shape[3])
			if h == expected_s0 and w == expected_s0:
				feat_s0 = tensor
				continue
			if h == expected_s1 and w == expected_s1:
				feat_s1 = tensor
				continue
		if image_embed is None:
			image_embed = tensor

	if image_embed is None:
		raise RuntimeError("SAM encoder output missing image embedding")
	if feat_s0 is None:
		raise RuntimeError(f"SAM encoder output missing {expected_s0}x{expected_s0} feature")
	if feat_s1 is None:
		raise RuntimeError(f"SAM encoder output missing {expected_s1}x{expected_s1} feature")
	return image_embed, feat_s0, feat_s1


def _apply_mask_to_image(img_rgb: np.ndarray, mask_tensor: torch.Tensor) -> np.ndarray:
	mask = mask_tensor
	while mask.dim() > 2:
		mask = mask[0]
	mask_np = mask.detach().cpu().numpy()
	mask_bin = mask_np > 0.0
	mask_u8 = (mask_bin.astype(np.uint8) * 255)

	h, w = img_rgb.shape[:2]
	mask_resized = cv2.resize(mask_u8, (w, h), interpolation=cv2.INTER_NEAREST)
	keep = mask_resized > 0
	bg = np.array([128, 128, 128], dtype=np.uint8)
	output = img_rgb.copy()
	output[~keep] = bg
	return output


def apply_sam_mask(
	img_rgb: np.ndarray,
	bbox_xyxy: np.ndarray,
	sam_encoder: ExecuTorchModule,
	sam_decoder: ExecuTorchModule,
	sam_input_size: int,
) -> np.ndarray:
	sam_tensor = _to_sam_tensor(img_rgb, sam_input_size)
	encoder_outputs = sam_encoder.forward(sam_tensor)
	encoder_tensors = _normalize_outputs(encoder_outputs)
	image_embed, feat_s0, feat_s1 = _select_sam_features(encoder_tensors, sam_input_size)
	scaled_bbox = _scale_bbox(bbox_xyxy, img_rgb.shape[1], img_rgb.shape[0], sam_input_size, sam_input_size)
	box_tensor = torch.from_numpy(scaled_bbox).view(1, 1, 4)
	decoder_outputs = sam_decoder.forward(image_embed, feat_s0, feat_s1, box_tensor)
	mask_tensor = _normalize_outputs(decoder_outputs)[0]
	return _apply_mask_to_image(img_rgb, mask_tensor)


def _softmax(scores: np.ndarray) -> np.ndarray:
	scores = scores.astype(np.float64)
	max_logit = np.max(scores) if scores.size else 0.0
	exp_values = np.exp(scores - max_logit)
	sum_exp = np.sum(exp_values)
	if sum_exp <= 1e-8:
		return np.zeros_like(scores, dtype=np.float32)
	return (exp_values / sum_exp).astype(np.float32)


def _compute_metrics(
	y_true: np.ndarray,
	y_pred: np.ndarray,
	class_names: Sequence[str],
) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
	labels = list(range(len(class_names)))
	cm = confusion_matrix(y_true, y_pred, labels=labels)
	precision, recall, f1, support = precision_recall_fscore_support(
		y_true,
		y_pred,
		labels=labels,
		zero_division=0,
	)

	per_class_accuracy = np.divide(
		np.diag(cm),
		cm.sum(axis=1),
		out=np.zeros(len(class_names), dtype=np.float64),
		where=cm.sum(axis=1) != 0,
	)

	rows: List[Dict[str, object]] = []
	for idx, name in enumerate(class_names):
		rows.append(
			{
				"class_name": name,
				"accuracy": float(per_class_accuracy[idx]),
				"precision": float(precision[idx]),
				"recall": float(recall[idx]),
				"f1_score": float(f1[idx]),
				"support": int(support[idx]),
			}
		)

	total_accuracy = float((y_true == y_pred).mean()) if len(y_true) else 0.0
	macro_avg = {
		"accuracy": float(np.nanmean(per_class_accuracy)) if len(per_class_accuracy) else 0.0,
		"precision": float(precision.mean()) if len(precision) else 0.0,
		"recall": float(recall.mean()) if len(recall) else 0.0,
		"f1_score": float(f1.mean()) if len(f1) else 0.0,
		"support": int(support.sum()),
	}

	metrics_summary = {
		"overall_accuracy": total_accuracy,
		"macro_accuracy": macro_avg["accuracy"],
		"macro_precision": macro_avg["precision"],
		"macro_recall": macro_avg["recall"],
		"macro_f1": macro_avg["f1_score"],
	}

	rows.append({"class_name": "macro avg", **macro_avg})
	return rows, metrics_summary


def _iter_images(samples: Sequence[Tuple[Path, int, str]], max_samples: Optional[int]) -> Iterable[Tuple[Path, int, str]]:
	count = 0
	for item in samples:
		yield item
		count += 1
		if max_samples is not None and max_samples > 0 and count >= max_samples:
			break


def _iter_batches(
	samples: Sequence[Tuple[Path, int, str]],
	batch_size: int,
	max_samples: Optional[int],
) -> Iterable[List[Tuple[Path, int, str]]]:
	limit = len(samples)
	if max_samples is not None and max_samples > 0:
		limit = min(limit, max_samples)
	for i in range(0, limit, max(1, batch_size)):
		yield list(samples[i : i + batch_size])


def main() -> None:
	parser = argparse.ArgumentParser(description="ExecuTorch test runner for AIHub test set")
	parser.add_argument("--test_dir", type=str, default=str(DEFAULT_TEST_DIR), help="Test data root")
	parser.add_argument("--assets_dir", type=str, default=str(DEFAULT_ASSETS_DIR), help="mobile_assets root")
	parser.add_argument("--second_stage_pte", type=str, default=str(DEFAULT_SECOND_STAGE_PTE), help="Second-stage PTE")
	parser.add_argument("--sam_encoder_pte", type=str, default=str(DEFAULT_SAM_ENCODER_PTE), help="SAM encoder PTE")
	parser.add_argument("--sam_decoder_pte", type=str, default=str(DEFAULT_SAM_DECODER_PTE), help="SAM decoder PTE")
	parser.add_argument(
		"--bbox_dir",
		type=str,
		default=None,
		help="Directory with bbox jsons mirroring test_dir structure",
	)
	parser.add_argument(
		"--save_masked_dir",
		type=str,
		default=None,
		help="Directory to save SAM-masked images (mirrors test_dir structure)",
	)
	parser.add_argument(
		"--save_bbox_dir",
		type=str,
		default=None,
		help="Directory to save images with GT bbox overlay (mirrors test_dir structure)",
	)
	parser.add_argument(
		"--bbox_source",
		type=str,
		default="bbox",
		choices=["bbox", "part_union"],
		help="Which annotation to use for SAM box: bbox or union of part",
	)
	parser.add_argument(
		"--debug_bbox",
		action="store_true",
		help="Print bbox debug info (json path, image size, xyxy)",
	)
	parser.add_argument("--no_sam", action="store_true", help="Disable SAM masking (default uses full-image box)")
	parser.add_argument("--image_size", type=int, default=224, help="Second-stage input size")
	parser.add_argument("--sam_input_size", type=int, default=512, help="SAM input size")
	parser.add_argument("--batch_size", type=int, default=32, help="Batch size for --no_sam")
	parser.add_argument("--device", type=str, default="cpu", help="Device for --no_sam (e.g., cpu, cuda)")
	parser.add_argument("--debug_preprocess", action="store_true", help="Print preprocess tensor stats")
	parser.add_argument("--debug_text", action="store_true", help="Print text embedding stats")
	parser.add_argument("--debug_compare_pth", action="store_true", help="Compare PTE logits with PTH")
	parser.add_argument("--pth_only", action="store_true", help="Run PTH model only")
	parser.add_argument(
		"--pth_path",
		type=str,
		default=str(PROJECT_ROOT / "weights" / "mob3_1" / "best_classifier.pth"),
		help="Path to original .pth checkpoint for comparison",
	)
	parser.add_argument("--debug_samples", type=int, default=3, help="Number of samples to debug")
	parser.add_argument("--max_samples", type=int, default=None, help="Limit samples for quick test")
	args = parser.parse_args()

	assets_dir = Path(args.assets_dir)
	class_names = _read_json_list(assets_dir / "class_names.json")
	print("Classifier class index mapping:")
	for idx, name in enumerate(class_names):
		print(f"  {idx}: {name}")
	crops_path = assets_dir / "crops.json"
	embedding_shape_path = assets_dir / "embedding_shape.json"
	embedding_dir = assets_dir / "embeddings"

	num_queries, embed_dim = _read_embedding_shape(embedding_shape_path)
	embedding_map = _load_embeddings(crops_path, embedding_dir)

	use_sam = not args.no_sam
	second_stage = None if args.pth_only else ExecuTorchModule(Path(args.second_stage_pte))
	sam_encoder = ExecuTorchModule(Path(args.sam_encoder_pte)) if use_sam else None
	sam_decoder = ExecuTorchModule(Path(args.sam_decoder_pte)) if use_sam else None
	bbox_dir = Path(args.bbox_dir) if args.bbox_dir else Path(args.test_dir)
	debug_seen = 0
	pth_model = None
	if args.debug_compare_pth or args.pth_only:
		ckpt = load_checkpoint(Path(args.pth_path), torch.device("cpu"))
		pth_model = build_model_from_checkpoint(ckpt, num_classes=len(class_names), device=torch.device("cpu"))
		pth_model.eval()

	samples = _discover_dataset(Path(args.test_dir), class_names)
	y_true: List[int] = []
	y_pred: List[int] = []
	infer_times_ms: List[float] = []
	save_masked_dir = Path(args.save_masked_dir) if args.save_masked_dir else None
	save_bbox_dir = Path(args.save_bbox_dir) if args.save_bbox_dir else None

	if use_sam:
		for image_path, label_idx, class_name in _iter_images(samples, args.max_samples):
			img_bgr = cv2.imread(str(image_path))
			if img_bgr is None:
				raise RuntimeError(f"Failed to read image: {image_path}")
			img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
			crop_name = class_name.split("_", 1)[0].strip().lower()
			if crop_name not in embedding_map:
				raise KeyError(f"Missing embedding for crop: {crop_name}")

			t_start = time.perf_counter()
			work_rgb = img_rgb
			if sam_encoder is not None and sam_decoder is not None:
				json_path = _find_bbox_json(image_path, Path(args.test_dir), bbox_dir)
				bbox_xyxy = _load_bbox_from_json(json_path, img_rgb, args.bbox_source)
				if args.debug_bbox:
					print(_format_bbox_debug(json_path, img_rgb, args.bbox_source, bbox_xyxy))
				if save_bbox_dir is not None:
					bbox_out_path = _build_masked_output_path(image_path, Path(args.test_dir), save_bbox_dir)
					bbox_out_path.parent.mkdir(parents=True, exist_ok=True)
					bbox_bgr = cv2.cvtColor(_draw_bbox_on_image(img_rgb, bbox_xyxy), cv2.COLOR_RGB2BGR)
					ok = cv2.imwrite(str(bbox_out_path), bbox_bgr)
					if not ok:
						raise RuntimeError(f"Failed to save bbox image: {bbox_out_path}")
				work_rgb = apply_sam_mask(work_rgb, bbox_xyxy, sam_encoder, sam_decoder, args.sam_input_size)
				if save_masked_dir is not None:
					out_path = _build_masked_output_path(image_path, Path(args.test_dir), save_masked_dir)
					out_path.parent.mkdir(parents=True, exist_ok=True)
					out_bgr = cv2.cvtColor(work_rgb, cv2.COLOR_RGB2BGR)
					ok = cv2.imwrite(str(out_path), out_bgr)
					if not ok:
						raise RuntimeError(f"Failed to save masked image: {out_path}")

			norm = _resize_and_normalize(work_rgb, args.image_size)
			image_tensor = _to_chw_tensor(norm)
			embedding = embedding_map[crop_name]
			if embedding.shape != (num_queries, embed_dim):
				embedding = embedding.reshape(num_queries, embed_dim)
			text_tensor = torch.from_numpy(embedding).view(1, num_queries, embed_dim)

			if debug_seen < args.debug_samples:
				if args.debug_preprocess:
					print(_format_tensor_stats("image_tensor", image_tensor))
				if args.debug_text:
					print(f"[text] crop={crop_name} shape={text_tensor.shape}")
					print(_format_tensor_stats("text_tensor", text_tensor))

			if args.pth_only:
				with torch.no_grad():
					pth_logits = pth_model(image_tensor, text_tensor).detach().cpu().numpy().reshape(-1)
				logits = pth_logits
			else:
				outputs = second_stage.forward(image_tensor, text_tensor)
				logits_tensor = _normalize_outputs(outputs)[0]
				logits = logits_tensor.detach().cpu().numpy().reshape(-1)
			probs = _softmax(logits)
			pred_idx = int(np.argmax(probs)) if probs.size else 0
			t_end = time.perf_counter()
			infer_times_ms.append((t_end - t_start) * 1000.0)

			if debug_seen < args.debug_samples and args.debug_compare_pth and pth_model is not None:
				with torch.no_grad():
					pth_logits = pth_model(image_tensor, text_tensor).detach().cpu().numpy().reshape(-1)
				pte_top = int(np.argmax(logits)) if logits.size else -1
				pth_top = int(np.argmax(pth_logits)) if pth_logits.size else -1
				diff = float(np.linalg.norm(logits - pth_logits)) if logits.size else 0.0
				print(
					f"[compare] pte_top={pte_top} pth_top={pth_top} logit_l2={diff:.6f}"
				)

			if debug_seen < args.debug_samples:
				debug_seen += 1

			y_true.append(label_idx)
			y_pred.append(pred_idx)
	else:
		device = torch.device(args.device)
		for batch in _iter_batches(samples, args.batch_size, args.max_samples):
			batch_images: List[np.ndarray] = []
			batch_labels: List[int] = []
			batch_texts: List[np.ndarray] = []

			for image_path, label_idx, class_name in batch:
				img_bgr = cv2.imread(str(image_path))
				if img_bgr is None:
					raise RuntimeError(f"Failed to read image: {image_path}")
				img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
				crop_name = class_name.split("_", 1)[0].strip().lower()
				if crop_name not in embedding_map:
					raise KeyError(f"Missing embedding for crop: {crop_name}")
				norm = _resize_and_normalize(img_rgb, args.image_size)
				chw = np.transpose(norm, (2, 0, 1))
				batch_images.append(chw)
				batch_labels.append(label_idx)
				embedding = embedding_map[crop_name]
				if embedding.shape != (num_queries, embed_dim):
					embedding = embedding.reshape(num_queries, embed_dim)
				batch_texts.append(embedding)

				if debug_seen < args.debug_samples:
					if args.debug_preprocess:
						print(_format_tensor_stats("image_tensor", torch.from_numpy(chw).unsqueeze(0)))
					if args.debug_text:
						text_tmp = torch.from_numpy(embedding).view(1, num_queries, embed_dim)
						print(f"[text] crop={crop_name} shape={text_tmp.shape}")
						print(_format_tensor_stats("text_tensor", text_tmp))

			image_batch = torch.from_numpy(np.stack(batch_images, axis=0)).to(device)
			text_batch = torch.from_numpy(np.stack(batch_texts, axis=0)).to(device)
			if device.type == "cuda":
				torch.cuda.synchronize()
			t_start = time.perf_counter()
			if args.pth_only:
				with torch.no_grad():
					pth_logits = pth_model(image_batch.cpu(), text_batch.cpu()).detach().cpu().numpy()
				logits = pth_logits
			else:
				outputs = second_stage.forward(image_batch, text_batch)
			if device.type == "cuda":
				torch.cuda.synchronize()
			t_end = time.perf_counter()
			if not args.pth_only:
				logits_tensor = _normalize_outputs(outputs)[0]
				logits = logits_tensor.detach().cpu().numpy()
			pred_indices = np.argmax(logits, axis=1) if logits.size else np.zeros(len(batch_labels), dtype=np.int64)

			per_image_ms = ((t_end - t_start) * 1000.0) / max(len(batch_labels), 1)
			infer_times_ms.extend([per_image_ms] * len(batch_labels))
			y_true.extend(batch_labels)
			y_pred.extend(pred_indices.tolist())

			if args.debug_compare_pth and pth_model is not None and debug_seen < args.debug_samples:
				for idx_in_batch in range(min(len(batch_labels), args.debug_samples - debug_seen)):
					img_t = image_batch[idx_in_batch : idx_in_batch + 1].cpu()
					text_t = text_batch[idx_in_batch : idx_in_batch + 1].cpu()
					with torch.no_grad():
						pth_logits = pth_model(img_t, text_t).detach().cpu().numpy().reshape(-1)
					pte_logits = logits[idx_in_batch]
					pte_top = int(np.argmax(pte_logits)) if pte_logits.size else -1
					pth_top = int(np.argmax(pth_logits)) if pth_logits.size else -1
					diff = float(np.linalg.norm(pte_logits - pth_logits)) if pte_logits.size else 0.0
					print(
						f"[compare] pte_top={pte_top} pth_top={pth_top} logit_l2={diff:.6f}"
					)
					debug_seen += 1

	y_true_np = np.asarray(y_true, dtype=np.int64)
	y_pred_np = np.asarray(y_pred, dtype=np.int64)
	rows, summary = _compute_metrics(y_true_np, y_pred_np, class_names)
	mean_ms = float(np.mean(infer_times_ms)) if infer_times_ms else 0.0

	print("=" * 80)
	print(f"Test dir: {args.test_dir}")
	print(f"Second-stage PTE: {args.second_stage_pte}")
	print(f"Use SAM: {use_sam}")
	if use_sam:
		print(f"SAM encoder PTE: {args.sam_encoder_pte}")
		print(f"SAM decoder PTE: {args.sam_decoder_pte}")
		print(f"BBox dir: {bbox_dir}")
	print(f"Samples: {len(y_true_np)}")
	print(f"Average inference time (ms): {mean_ms:.3f}")
	print("=" * 80)
	print(f"Overall Accuracy: {summary['overall_accuracy']:.6f}")
	print(f"Macro Accuracy: {summary['macro_accuracy']:.6f}")
	print(f"Macro Precision: {summary['macro_precision']:.6f}")
	print(f"Macro Recall: {summary['macro_recall']:.6f}")
	print(f"Macro F1: {summary['macro_f1']:.6f}")
	print("=" * 80)
	print("Per-class metrics:")
	for row in rows:
		if row["class_name"] == "macro avg":
			continue
		print(
			f"{row['class_name']}: acc={row['accuracy']:.6f}, "
			f"precision={row['precision']:.6f}, recall={row['recall']:.6f}, "
			f"f1={row['f1_score']:.6f}, support={row['support']}"
		)
	print("=" * 80)
	macro_row = rows[-1]
	print(
		f"macro avg: acc={macro_row['accuracy']:.6f}, precision={macro_row['precision']:.6f}, "
		f"recall={macro_row['recall']:.6f}, f1={macro_row['f1_score']:.6f}, support={macro_row['support']}"
	)


if __name__ == "__main__":
	main()
