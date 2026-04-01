import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.append(str(PROJECT_ROOT))

from base_model.SAM import SAM2MaskGenerator
from dataset.datasets import load_aihubbbox_label


def find_label_json_for_image(image_path: Path) -> Optional[Path]:
	"""Find sibling JSON label by matching image stem in the same folder."""
	label_path = image_path.with_suffix(".json")
	if label_path.exists():
		return label_path
	return None


def load_bbox_xyxy_from_label(image_path: Path) -> Optional[Tuple[int, int, int, int]]:
	"""Load bbox(x1,y1,x2,y2) from sibling AIHub JSON label."""
	json_path = find_label_json_for_image(image_path)
	if json_path is None:
		return None

	parsed = load_aihubbbox_label(json_path)
	if parsed is None:
		return None
	return tuple(parsed["bbox_xyxy"])


def iter_images(root: Path):
	"""Yield image paths recursively under root."""
	for p in root.rglob("*"):
		if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
			yield p


def _class_from_rel_path(rel_path: Path) -> str:
	"""Use the immediate parent folder as class name; root-level files become 'root'."""
	if rel_path.parent == Path("."):
		return "root"
	return rel_path.parent.name


def write_issue_report(
	output_dir: Path,
	skipped_items: List[Tuple[str, str, str]],
	failed_items: List[Tuple[str, str, str]],
) -> Path:
	"""Write skipped/failed item report to output directory as txt."""
	report_path = output_dir / "sam_inference_issues.txt"
	lines: List[str] = []
	lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
	lines.append("")
	lines.append(f"Skipped: {len(skipped_items)}")
	lines.append(f"Failed: {len(failed_items)}")
	lines.append("")

	lines.append("[SKIPPED]")
	if skipped_items:
		lines.append("class\tfile\treason")
		for cls, name, reason in skipped_items:
			lines.append(f"{cls}\t{name}\t{reason}")
	else:
		lines.append("(none)")
	lines.append("")

	lines.append("[FAILED]")
	if failed_items:
		lines.append("class\tfile\treason")
		for cls, name, reason in failed_items:
			lines.append(f"{cls}\t{name}\t{reason}")
	else:
		lines.append("(none)")

	report_path.write_text("\n".join(lines), encoding="utf-8")
	return report_path


def encode_mask_rle(mask_bool: np.ndarray) -> str:
	"""Run-length encode a binary mask in row-major order.

	Format:
		<start_value>;<count1,count2,...>
	Example:
		0;10,3,5 means 0 repeated 10, then 1 repeated 3, then 0 repeated 5...
	"""
	flat = mask_bool.astype(np.uint8).reshape(-1)
	if flat.size == 0:
		return "0;"

	start_value = int(flat[0])
	runs: List[int] = []
	current = start_value
	count = 1

	for v in flat[1:]:
		iv = int(v)
		if iv == current:
			count += 1
		else:
			runs.append(count)
			current = iv
			count = 1
	runs.append(count)

	return f"{start_value};" + ",".join(str(c) for c in runs)


def write_region_info(region_txt_path: Path, class_name: str, file_name: str, region_info: dict, mask_bool: np.ndarray) -> bool:
	"""Write selected mask bbox + exact mask RLE using the same image stem."""
	h, w = mask_bool.shape[:2]
	rle = encode_mask_rle(mask_bool)
	lines = [
		f"class: {class_name}",
		f"file: {file_name}",
		f"mask_height: {h}",
		f"mask_width: {w}",
		f"x1: {int(region_info['x1'])}",
		f"y1: {int(region_info['y1'])}",
		f"x2: {int(region_info['x2'])}",
		f"y2: {int(region_info['y2'])}",
		f"area_pixels: {int(region_info['area_pixels'])}",
		f"area_ratio: {float(region_info['area_ratio']):.8f}",
		f"rle: {rle}",
	]
	try:
		region_txt_path.parent.mkdir(parents=True, exist_ok=True)
		region_txt_path.write_text("\n".join(lines), encoding="utf-8")
		return True
	except Exception:
		return False


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Run SAM masking for all images in a target directory and save with the same subfolder structure.")
	parser.add_argument("--target-dir", type=str, required=True, help="Input root directory containing images")
	parser.add_argument("--output-dir", type=str, required=True, help="Output root directory to save masked images")
	parser.add_argument(
		"--mode",
		type=str,
		required=True,
		choices=["center_point", "box", "autoselector"],
		help="SAM masking mode",
	)
	parser.add_argument("--sam-weights", type=str, default="weights/sam2.1_t.pt", help="Path to SAM model weights")
	parser.add_argument(
		"--selector-checkpoint",
		type=str,
		default="weights/samselector/best_checkpoint.pth",
		help="Selector checkpoint path used when mode=autoselector",
	)
	parser.add_argument("--selector-device", type=str, default=None, help="cpu or cuda for selector (autoselector mode)")
	args = parser.parse_args()

	target_dir = Path(args.target_dir)
	output_dir = Path(args.output_dir)

	if not target_dir.exists() or not target_dir.is_dir():
		raise FileNotFoundError(f"Invalid target directory: {target_dir}")

	output_dir.mkdir(parents=True, exist_ok=True)

	generator = SAM2MaskGenerator(
		model_path=args.sam_weights,
		mode=args.mode,
		selector_checkpoint_path=args.selector_checkpoint,
		selector_device=args.selector_device,
	)

	total = 0
	saved = 0
	saved_region = 0
	skipped = 0
	failed = 0
	skipped_items: List[Tuple[str, str, str]] = []
	failed_items: List[Tuple[str, str, str]] = []

	for image_path in iter_images(target_dir):
		total += 1
		rel_path = image_path.relative_to(target_dir)
		out_path = output_dir / rel_path
		region_path = output_dir / rel_path.with_suffix(".txt")
		out_path.parent.mkdir(parents=True, exist_ok=True)

		bbox = None
		class_name = _class_from_rel_path(rel_path)
		file_name = image_path.name
		if args.mode in {"box", "autoselector"}:
			bbox = load_bbox_xyxy_from_label(image_path)
			if bbox is None:
				skipped += 1
				skipped_items.append((class_name, file_name, "no_valid_bbox_label"))
				print(f"[SKIP] No valid bbox label: {image_path}")
				continue

		try:
			masked, region_info, selected_mask = generator.generate_with_region(str(image_path), bbox=bbox)
			ok = cv2.imwrite(str(out_path), masked)
			if not ok:
				failed += 1
				failed_items.append((class_name, file_name, "save_failed"))
				print(f"[FAIL] Could not save image: {out_path}")
				continue

			ok_region = write_region_info(
				region_path,
				class_name=class_name,
				file_name=file_name,
				region_info=region_info,
				mask_bool=selected_mask,
			)
			if not ok_region:
				failed += 1
				failed_items.append((class_name, file_name, "save_region_failed"))
				print(f"[FAIL] Could not save region info: {region_path}")
				continue

			saved += 1
			saved_region += 1
			print(f"[SAVE] {out_path}")
			print(f"[REGION] {region_path}")
		except Exception as e:
			failed += 1
			failed_items.append((class_name, file_name, f"exception:{type(e).__name__}"))
			print(f"[ERROR] {image_path} -> {e}")

	report_path = write_issue_report(output_dir, skipped_items=skipped_items, failed_items=failed_items)

	print("\n=== SAM Inference Summary ===")
	print(f"Mode: {args.mode}")
	print(f"Target dir: {target_dir}")
	print(f"Output dir: {output_dir}")
	print(f"Total images: {total}")
	print(f"Saved: {saved}")
	print(f"Saved regions: {saved_region}")
	print(f"Skipped: {skipped}")
	print(f"Failed: {failed}")
	print(f"Issue report: {report_path}")


if __name__ == "__main__":
	main()

