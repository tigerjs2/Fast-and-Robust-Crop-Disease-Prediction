import argparse
import csv
import os
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.append(str(PROJECT_ROOT))

from base_model.ImageExtractor import CrossAttentionModule, FeatureExtractor
from dataset.datasets import build_classification_dataloader, classification_collate_fn
from scripts.SecondStageTrain import (
	SecondStageClassifier,
	build_batch_text_queries,
	build_image_transform,
	load_crop_text_embeddings,
	parse_gt_class_name,
	set_seed,
)


DEFAULT_WEIGHT_PATH = PROJECT_ROOT / "weights" / "second_stage" / "best_classifier.pth"
DEFAULT_TEST_DIR = PROJECT_ROOT / "data" / "AIHub" / "test"
DEFAULT_TEXT_EMBEDDING_DIR = PROJECT_ROOT / "data" / "TextEmbeddings"
DEFAULT_MISMATCH_SEED = 314159


def seed_worker(worker_id: int) -> None:
	worker_seed = torch.initial_seed() % (2 ** 32)
	np.random.seed(worker_seed)
	random.seed(worker_seed)


def get_output_paths(weight_path: Path) -> Dict[str, Path]:
	output_dir = weight_path.parent
	prefix = weight_path.stem
	return {
		"report_txt": output_dir / f"{prefix}_test_report.txt",
		"metrics_csv": output_dir / f"{prefix}_metrics.csv",
		"confusion_svg": output_dir / f"{prefix}_confusion_matrix.svg",
		"mismatch_csv": output_dir / f"{prefix}_mismatch_report.csv",
		"mismatch_summary_txt": output_dir / f"{prefix}_mismatch_summary.txt",
	}


def load_checkpoint(weight_path: Path, device: torch.device) -> Dict:
	if not weight_path.exists():
		raise FileNotFoundError(f"Checkpoint not found: {weight_path}")
	return torch.load(str(weight_path), map_location=device)


def build_model_from_checkpoint(checkpoint: Dict, num_classes: int, device: torch.device) -> SecondStageClassifier:
	config = checkpoint.get("config", {})
	model = SecondStageClassifier(
		num_classes=num_classes,
		backbone_name=config.get("backbone_name", "mobilenet_v3_large"),
		backbone_weights=config.get("backbone_weights", "DEFAULT"),
		embed_dim=config.get("embed_dim", 512),
		num_heads=config.get("num_heads", 8),
		attn_dropout=config.get("attn_dropout", 0.0),
		residual_scale=config.get("residual_scale", 1.0),
		classifier_dropout=config.get("classifier_dropout", 0.1),
	).to(device)
	model.load_state_dict(checkpoint["model_state_dict"], strict=True)
	model.eval()
	return model


def build_test_loader(
	test_dir: str,
	class_names: Sequence[str],
	image_size: int,
	batch_size: int,
	num_workers: int,
	seed: int,
) -> DataLoader:
	transform = build_image_transform(image_size=image_size, apply_augmentation=False)
	return build_classification_dataloader(
		root_dir=test_dir,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		transform=transform,
		class_names=class_names,
		use_default_classes=False,
		return_path=True,
		seed=seed,
	)


def evaluate_matching_texts(
	model: SecondStageClassifier,
	loader: DataLoader,
	embedding_map: Dict[str, torch.Tensor],
	device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], List[str]]:
	y_true: List[int] = []
	y_pred: List[int] = []
	true_class_names: List[str] = []
	image_paths: List[str] = []
	gt_crops: List[str] = []

	with torch.no_grad():
		pbar = tqdm(loader, desc="Test Eval", dynamic_ncols=True)
		for batch in pbar:
			images = batch["images"].to(device, non_blocking=True)
			labels = batch["labels"].to(device, non_blocking=True)
			batch_class_names = batch["class_names"]
			matched_text_queries, crops, _ = build_batch_text_queries(
				batch_class_names=batch_class_names,
				embedding_map=embedding_map,
				device=device,
			)

			logits = model(images, matched_text_queries)
			preds = logits.argmax(dim=1)

			y_true.extend(labels.detach().cpu().tolist())
			y_pred.extend(preds.detach().cpu().tolist())
			true_class_names.extend(batch_class_names)
			gt_crops.extend(crops)
			if "image_paths" in batch:
				image_paths.extend(batch["image_paths"])
			else:
				image_paths.extend([""] * len(batch_class_names))

	return (
		np.asarray(y_true, dtype=np.int64),
		np.asarray(y_pred, dtype=np.int64),
		true_class_names,
		image_paths,
		gt_crops,
	)


def compute_metrics_table(
	y_true: np.ndarray,
	y_pred: np.ndarray,
	class_names: Sequence[str],
) -> Tuple[List[Dict[str, object]], Dict[str, float], np.ndarray]:
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
	for idx, class_name in enumerate(class_names):
		rows.append(
			{
				"class_name": class_name,
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
	weighted_avg = {
		"accuracy": float(np.average(per_class_accuracy, weights=support)) if support.sum() else 0.0,
		"precision": float(np.average(precision, weights=support)) if support.sum() else 0.0,
		"recall": float(np.average(recall, weights=support)) if support.sum() else 0.0,
		"f1_score": float(np.average(f1, weights=support)) if support.sum() else 0.0,
		"support": int(support.sum()),
	}

	metrics_summary = {
		"overall_accuracy": total_accuracy,
		"macro_accuracy": macro_avg["accuracy"],
		"macro_precision": macro_avg["precision"],
		"macro_recall": macro_avg["recall"],
		"macro_f1": macro_avg["f1_score"],
		"weighted_accuracy": weighted_avg["accuracy"],
		"weighted_precision": weighted_avg["precision"],
		"weighted_recall": weighted_avg["recall"],
		"weighted_f1": weighted_avg["f1_score"],
	}

	rows.append({"class_name": "macro avg", **macro_avg})
	rows.append({"class_name": "weighted avg", **weighted_avg})
	return rows, metrics_summary, cm


def save_metrics_csv(rows: List[Dict[str, object]], csv_path: Path) -> None:
	csv_path.parent.mkdir(parents=True, exist_ok=True)
	fieldnames = ["class_name", "accuracy", "precision", "recall", "f1_score", "support"]
	with csv_path.open("w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		for row in rows:
			writer.writerow(row)


def save_metrics_txt(
	rows: List[Dict[str, object]],
	metrics_summary: Dict[str, float],
	output_path: Path,
	weight_path: Path,
	test_dir: str,
) -> None:
	lines: List[str] = []
	lines.append("=" * 80)
	lines.append(f"Evaluation Time: {datetime.now().isoformat(timespec='seconds')}")
	lines.append(f"Weight Path: {weight_path}")
	lines.append(f"Test Dir: {test_dir}")
	lines.append("=" * 80)
	lines.append(f"Overall Accuracy: {metrics_summary['overall_accuracy']:.6f}")
	lines.append(f"Macro Accuracy: {metrics_summary['macro_accuracy']:.6f}")
	lines.append(f"Macro Precision: {metrics_summary['macro_precision']:.6f}")
	lines.append(f"Macro Recall: {metrics_summary['macro_recall']:.6f}")
	lines.append(f"Macro F1: {metrics_summary['macro_f1']:.6f}")
	lines.append(f"Weighted Accuracy: {metrics_summary['weighted_accuracy']:.6f}")
	lines.append(f"Weighted Precision: {metrics_summary['weighted_precision']:.6f}")
	lines.append(f"Weighted Recall: {metrics_summary['weighted_recall']:.6f}")
	lines.append(f"Weighted F1: {metrics_summary['weighted_f1']:.6f}")
	lines.append("=" * 80)
	lines.append("Per-class metrics:")
	for row in rows:
		if row["class_name"] in {"macro avg", "weighted avg"}:
			continue
		lines.append(
			f"{row['class_name']}: acc={row['accuracy']:.6f}, precision={row['precision']:.6f}, "
			f"recall={row['recall']:.6f}, f1={row['f1_score']:.6f}, support={row['support']}"
		)
	lines.append("=" * 80)
	lines.append("Macro avg:")
	macro = next(row for row in rows if row["class_name"] == "macro avg")
	lines.append(
		f"acc={macro['accuracy']:.6f}, precision={macro['precision']:.6f}, recall={macro['recall']:.6f}, "
		f"f1={macro['f1_score']:.6f}, support={macro['support']}"
	)
	lines.append("Weighted avg:")
	weighted = next(row for row in rows if row["class_name"] == "weighted avg")
	lines.append(
		f"acc={weighted['accuracy']:.6f}, precision={weighted['precision']:.6f}, recall={weighted['recall']:.6f}, "
		f"f1={weighted['f1_score']:.6f}, support={weighted['support']}"
	)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", encoding="utf-8") as f:
		f.write("\n".join(lines))


def save_confusion_matrix_svg(cm: np.ndarray, class_names: Sequence[str], svg_path: Path) -> None:
	svg_path.parent.mkdir(parents=True, exist_ok=True)
	fig_size = max(8.0, len(class_names) * 0.6)
	fig, ax = plt.subplots(figsize=(fig_size, fig_size))
	im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
	ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
	tick_positions = np.arange(len(class_names))
	ax.set_xticks(tick_positions)
	ax.set_yticks(tick_positions)
	ax.set_xticklabels(class_names)
	ax.set_yticklabels(class_names)
	ax.set_title("Confusion Matrix")
	ax.set_ylabel("True label")
	ax.set_xlabel("Predicted label")
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

	thresh = cm.max() / 2.0 if cm.size else 0.0
	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			ax.text(
				j,
				i,
				format(cm[i, j], "d"),
				ha="center",
				va="center",
				color="white" if cm[i, j] > thresh else "black",
				fontsize=8,
			)

	fig.tight_layout()
	fig.savefig(str(svg_path), format="svg")
	plt.close(fig)


def build_mismatch_selection(
	dataset,
	class_names: Sequence[str],
	mismatch_ratio: float,
	mismatch_seed: int,
) -> List[int]:
	rng = random.Random(mismatch_seed)
	indices_by_class: Dict[int, List[int]] = defaultdict(list)
	for index, (_, label) in enumerate(dataset.samples):
		indices_by_class[int(label)].append(index)

	selected_indices: List[int] = []
	for class_index in range(len(class_names)):
		class_indices = indices_by_class.get(class_index, [])
		if not class_indices:
			continue
		select_count = max(1, int(round(len(class_indices) * mismatch_ratio)))
		select_count = min(select_count, len(class_indices))
		selected_indices.extend(rng.sample(class_indices, select_count))

	return selected_indices


def run_mismatch_analysis(
	model: SecondStageClassifier,
	dataset,
	selected_indices: List[int],
	class_names: Sequence[str],
	embedding_map: Dict[str, torch.Tensor],
	device: torch.device,
	batch_size: int,
	num_workers: int,
	seed: int,
	mismatch_seed: int,
) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
	if not selected_indices:
		return [], {"overall_success_rate": 0.0}

	subset = Subset(dataset, selected_indices)
	loader = DataLoader(
		subset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		collate_fn=classification_collate_fn,
		pin_memory=torch.cuda.is_available(),
		worker_init_fn=seed_worker,
		generator=torch.Generator().manual_seed(seed),
	)

	all_crops = [parse_gt_class_name(class_name)[0] for class_name in class_names]
	rng = random.Random(mismatch_seed)
	records: List[Dict[str, object]] = []
	total_success = 0
	total_count = 0
	per_class_counts: Dict[str, int] = defaultdict(int)
	per_class_success: Dict[str, int] = defaultdict(int)

	with torch.no_grad():
		pbar = tqdm(loader, desc="Mismatch Eval", dynamic_ncols=True)
		for batch in pbar:
			images = batch["images"].to(device, non_blocking=True)
			batch_class_names = batch["class_names"]
			batch_paths = batch.get("image_paths", [""] * len(batch_class_names))

			mismatch_texts: List[torch.Tensor] = []
			mismatch_crop_names: List[str] = []
			for class_name in batch_class_names:
				gt_crop, _ = parse_gt_class_name(class_name)
				candidates = [crop for crop in all_crops if crop != gt_crop]
				if not candidates:
					raise RuntimeError(f"No mismatch candidates available for crop: {gt_crop}")
				picked_crop = rng.choice(candidates)
				mismatch_crop_names.append(picked_crop)
				mismatch_texts.append(embedding_map[picked_crop])

			mismatch_text_queries = torch.stack(mismatch_texts, dim=0).to(device)
			mismatch_logits = model(images, mismatch_text_queries)
			mismatch_probs = F.softmax(mismatch_logits, dim=1)
			max_probs, pred_indices = mismatch_probs.max(dim=1)
			pred_class_names = [class_names[idx] for idx in pred_indices.detach().cpu().tolist()]

			for i, class_name in enumerate(batch_class_names):
				confidence = float(max_probs[i].item())
				pred_crop = parse_gt_class_name(pred_class_names[i])[0]
				# Success if: confidence < 0.65 OR predicted crop differs from mismatched crop
				success = (confidence < 0.65) or (pred_crop != mismatch_crop_names[i])
				total_count += 1
				total_success += int(success)
				per_class_counts[class_name] += 1
				per_class_success[class_name] += int(success)
				records.append(
					{
						"image_path": batch_paths[i],
						"gt_class": class_name,
						"gt_crop": parse_gt_class_name(class_name)[0],
						"mismatch_text_crop": mismatch_crop_names[i],
						"pred_class": pred_class_names[i],
						"max_softmax": confidence,
						"result": "success" if success else "fail",
					}
				)

	overall_success_rate = float(total_success / total_count) if total_count else 0.0
	per_class_summary: Dict[str, float] = {"overall_success_rate": overall_success_rate}
	for class_name in class_names:
		count = per_class_counts.get(class_name, 0)
		success = per_class_success.get(class_name, 0)
		per_class_summary[f"{class_name}_success_rate"] = float(success / count) if count else 0.0
		per_class_summary[f"{class_name}_count"] = float(count)

	return records, per_class_summary


def save_mismatch_csv(records: List[Dict[str, object]], output_path: Path) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fieldnames = ["image_path", "gt_class", "gt_crop", "mismatch_text_crop", "pred_class", "max_softmax", "result"]
	with output_path.open("w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		for row in records:
			writer.writerow(row)


def save_mismatch_summary(summary: Dict[str, float], class_names: Sequence[str], output_path: Path) -> None:
	lines = ["=" * 80, "Mismatch Summary", "=" * 80]
	lines.append(f"Overall success rate: {summary.get('overall_success_rate', 0.0):.6f}")
	for class_name in class_names:
		success_key = f"{class_name}_success_rate"
		count_key = f"{class_name}_count"
		lines.append(
			f"{class_name}: success_rate={summary.get(success_key, 0.0):.6f}, count={int(summary.get(count_key, 0.0))}"
		)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", encoding="utf-8") as f:
		f.write("\n".join(lines))


def main() -> None:
	parser = argparse.ArgumentParser(description="Second-stage model evaluation script")
	parser.add_argument("--weight_path", type=str, default=str(DEFAULT_WEIGHT_PATH), help="Trained checkpoint path")
	parser.add_argument("--test_dir", type=str, default=str(DEFAULT_TEST_DIR), help="Test root folder")
	parser.add_argument(
		"--text_embedding_dir",
		type=str,
		default=str(DEFAULT_TEXT_EMBEDDING_DIR),
		help="Directory containing crop text embedding .pt files",
	)
	parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
	parser.add_argument("--num_workers", type=int, default=4, help="DataLoader num_workers")
	parser.add_argument("--image_size", type=int, default=224, help="Input image size")
	parser.add_argument("--device", type=str, default=None, help="cuda or cpu. default is auto")
	parser.add_argument("--seed", type=int, default=42, help="Random seed")
	parser.add_argument(
		"--mismatch_seed",
		type=int,
		default=DEFAULT_MISMATCH_SEED,
		help="Fixed seed used only for mismatch pair generation",
	)
	parser.add_argument("--mismatch_ratio", type=float, default=0.1, help="Per-class mismatch sampling ratio")
	args = parser.parse_args()

	set_seed(args.seed)
	device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
	weight_path = Path(args.weight_path)
	output_paths = get_output_paths(weight_path)

	checkpoint = load_checkpoint(weight_path, device)
	class_names = checkpoint.get("class_names")
	if not class_names:
		raise KeyError("Checkpoint does not contain class_names")

	model = build_model_from_checkpoint(checkpoint, num_classes=len(class_names), device=device)
	test_loader = build_test_loader(
		test_dir=args.test_dir,
		class_names=class_names,
		image_size=args.image_size,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
		seed=args.seed,
	)
	if list(test_loader.dataset.class_names) != list(class_names):
		raise ValueError(
			"Test folder class order does not match checkpoint class_names. "
			f"checkpoint={list(class_names)}, test={list(test_loader.dataset.class_names)}"
		)

	all_crops = [parse_gt_class_name(class_name)[0] for class_name in class_names]
	embedding_map = load_crop_text_embeddings(
		text_embedding_dir=Path(args.text_embedding_dir),
		crop_names=all_crops,
		device=device,
		expected_num_queries=5,
	)

	y_true, y_pred, true_class_names, image_paths, gt_crops = evaluate_matching_texts(
		model=model,
		loader=test_loader,
		embedding_map=embedding_map,
		device=device,
	)

	rows, metrics_summary, cm = compute_metrics_table(y_true, y_pred, class_names)
	save_metrics_csv(rows, output_paths["metrics_csv"])
	save_metrics_txt(rows, metrics_summary, output_paths["report_txt"], weight_path, args.test_dir)
	save_confusion_matrix_svg(cm, class_names, output_paths["confusion_svg"])

	selected_indices = build_mismatch_selection(
		dataset=test_loader.dataset,
		class_names=class_names,
		mismatch_ratio=args.mismatch_ratio,
		mismatch_seed=args.mismatch_seed,
	)
	mismatch_records, mismatch_summary = run_mismatch_analysis(
		model=model,
		dataset=test_loader.dataset,
		selected_indices=selected_indices,
		class_names=class_names,
		embedding_map=embedding_map,
		device=device,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
		seed=args.seed,
		mismatch_seed=args.mismatch_seed,
	)
	save_mismatch_csv(mismatch_records, output_paths["mismatch_csv"])
	save_mismatch_summary(mismatch_summary, class_names, output_paths["mismatch_summary_txt"])

	print(f"Saved report: {output_paths['report_txt']}")
	print(f"Saved metrics csv: {output_paths['metrics_csv']}")
	print(f"Saved confusion matrix svg: {output_paths['confusion_svg']}")
	print(f"Saved mismatch csv: {output_paths['mismatch_csv']}")
	print(f"Saved mismatch summary: {output_paths['mismatch_summary_txt']}")
	print(f"Overall accuracy: {metrics_summary['overall_accuracy']:.6f}")
	print(f"Macro F1: {metrics_summary['macro_f1']:.6f}")
	print(f"Weighted F1: {metrics_summary['weighted_f1']:.6f}")
	print(f"Mismatch overall success rate: {mismatch_summary.get('overall_success_rate', 0.0):.6f}")


if __name__ == "__main__":
	main()
