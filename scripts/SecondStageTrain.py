import argparse
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.append(str(PROJECT_ROOT))

from base_model.ImageExtractor import CrossAttentionModule, FeatureExtractor
from dataset.datasets import build_classification_dataloader


DEFAULT_TRAIN_DIR = PROJECT_ROOT / "data" / "AIHub" / "train"
DEFAULT_VAL_DIR = PROJECT_ROOT / "data" / "AIHub" / "val"
DEFAULT_TEXT_EMBEDDING_DIR = PROJECT_ROOT / "data" / "TextEmbeddings"
DEFAULT_SAVE_PATH = PROJECT_ROOT / "weights" / "second_stage" / "best_classifier.pth"


def derive_last_save_path(best_save_path: Path) -> Path:
	"""best 경로와 같은 폴더에 last 체크포인트 파일 경로를 만든다."""
	return best_save_path.with_name(f"last_{best_save_path.name}")


def derive_log_path(best_save_path: Path) -> Path:
	"""체크포인트 경로와 같은 폴더에 학습 로그 파일 경로를 만든다."""
	return best_save_path.with_name("train_log.txt")


def append_log(log_path: Path, message: str) -> None:
	"""학습 로그를 텍스트 파일에 누적 저장한다."""
	log_path.parent.mkdir(parents=True, exist_ok=True)
	with log_path.open("a", encoding="utf-8") as f:
		f.write(message + "\n")


def set_seed(seed: int) -> None:
	os.environ["PYTHONHASHSEED"] = str(seed)
	os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	try:
		torch.use_deterministic_algorithms(True, warn_only=True)
	except Exception:
		# 구버전 호환용 fallback
		pass


def _apply_random_linear_augmentation(
	img_rgb: np.ndarray,
	hflip_prob: float,
	vflip_prob: float,
	rotate_deg: float,
	translate_ratio: float,
	scale_min: float,
	scale_max: float,
	shear_deg: float,
) -> np.ndarray:
	"""객체 형태를 크게 훼손하지 않는 범위에서 선형 기반 랜덤 증강을 적용한다."""
	out = img_rgb.copy()
	h, w = out.shape[:2]

	if random.random() < hflip_prob:
		out = cv2.flip(out, 1)
	if random.random() < vflip_prob:
		out = cv2.flip(out, 0)

	angle = random.uniform(-rotate_deg, rotate_deg)
	tx = random.uniform(-translate_ratio, translate_ratio) * w
	ty = random.uniform(-translate_ratio, translate_ratio) * h
	scale = random.uniform(scale_min, scale_max)
	shear = np.deg2rad(random.uniform(-shear_deg, shear_deg))

	# 회전/스케일/이동 + x축 쉬어를 하나의 affine matrix로 결합
	center = (w / 2.0, h / 2.0)
	rot = cv2.getRotationMatrix2D(center, angle, scale)  # 2x3
	rot3 = np.vstack([rot, np.array([0.0, 0.0, 1.0], dtype=np.float32)]).astype(np.float32)
	shear_m = np.array(
		[
			[1.0, np.tan(shear), 0.0],
			[0.0, 1.0, 0.0],
			[0.0, 0.0, 1.0],
		],
		dtype=np.float32,
	)
	affine = shear_m @ rot3
	affine[0, 2] += tx
	affine[1, 2] += ty

	out = cv2.warpAffine(
		out,
		affine[:2, :],
		(w, h),
		flags=cv2.INTER_LINEAR,
		borderMode=cv2.BORDER_REFLECT_101,
	)
	return out


class ImageTransform:
	def __init__(
		self,
		image_size: int,
		apply_augmentation: bool = False,
		hflip_prob: float = 0.5,
		vflip_prob: float = 0.1,
		rotate_deg: float = 10.0,
		translate_ratio: float = 0.05,
		scale_min: float = 0.95,
		scale_max: float = 1.05,
		shear_deg: float = 5.0,
	):
		self.image_size = image_size
		self.apply_augmentation = apply_augmentation
		self.hflip_prob = hflip_prob
		self.vflip_prob = vflip_prob
		self.rotate_deg = rotate_deg
		self.translate_ratio = translate_ratio
		self.scale_min = scale_min
		self.scale_max = scale_max
		self.shear_deg = shear_deg
		self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
		self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

	def __call__(self, img_rgb: np.ndarray) -> torch.Tensor:
		work = img_rgb
		if self.apply_augmentation:
			work = _apply_random_linear_augmentation(
				img_rgb=work,
				hflip_prob=self.hflip_prob,
				vflip_prob=self.vflip_prob,
				rotate_deg=self.rotate_deg,
				translate_ratio=self.translate_ratio,
				scale_min=self.scale_min,
				scale_max=self.scale_max,
				shear_deg=self.shear_deg,
			)

		resized = cv2.resize(work, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
		tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
		tensor = (tensor - self.mean) / self.std
		return tensor


def build_image_transform(
	image_size: int,
	apply_augmentation: bool = False,
	hflip_prob: float = 0.5,
	vflip_prob: float = 0.1,
	rotate_deg: float = 10.0,
	translate_ratio: float = 0.05,
	scale_min: float = 0.95,
	scale_max: float = 1.05,
	shear_deg: float = 5.0,
):
	return ImageTransform(
		image_size=image_size,
		apply_augmentation=apply_augmentation,
		hflip_prob=hflip_prob,
		vflip_prob=vflip_prob,
		rotate_deg=rotate_deg,
		translate_ratio=translate_ratio,
		scale_min=scale_min,
		scale_max=scale_max,
		shear_deg=shear_deg,
	)


def parse_class_names(raw: Optional[str]) -> Optional[List[str]]:
	if raw is None:
		return None
	parsed = [x.strip() for x in raw.split(",") if x.strip()]
	return parsed if parsed else None


def parse_gt_class_name(class_name: str) -> Tuple[str, bool]:
	"""
	클래스 이름을 {crop}_{condition} 형태로 해석한다.
	예: cucumber_healthy -> (cucumber, True), tomato_graymold -> (tomato, False)
	"""
	name = class_name.strip().lower()
	if "_" not in name:
		raise ValueError(f"Invalid class name format: {class_name}")
	crop = name.split("_", 1)[0]
	is_healthy = name.endswith("_healthy")
	return crop, is_healthy


def resolve_crop_embedding_path(text_embedding_dir: Path, crop_name: str) -> Path:
	"""작물명에 해당하는 임베딩 파일 경로를 찾는다."""
	alias_map = {
		"paprica": "paprika",
	}

	candidates = [crop_name]
	if crop_name in alias_map:
		candidates.append(alias_map[crop_name])

	for name in candidates:
		p = text_embedding_dir / f"{name}.pt"
		if p.exists():
			return p

	raise FileNotFoundError(
		f"No text embedding .pt found for crop '{crop_name}' in {text_embedding_dir}"
	)


def load_crop_text_embeddings(
	text_embedding_dir: Path,
	crop_names: Sequence[str],
	device: torch.device,
	expected_num_queries: int = 5,
) -> Dict[str, torch.Tensor]:
	"""
	작물별 CLIP 쿼리 임베딩을 로드한다.
	각 파일은 (5, D) 텐서여야 하며, 0번은 "a {vegetable} leaf" 벡터로 사용한다.
	"""
	embedding_map: Dict[str, torch.Tensor] = {}

	for crop in sorted(set(crop_names)):
		emb_path = resolve_crop_embedding_path(text_embedding_dir, crop)
		emb = torch.load(str(emb_path), map_location="cpu")

		if not isinstance(emb, torch.Tensor):
			raise TypeError(f"Embedding file must contain a torch.Tensor: {emb_path}")
		if emb.dim() != 2:
			raise ValueError(f"Embedding tensor must be 2D (num_queries, dim): {emb_path}")
		if emb.size(0) != expected_num_queries:
			raise ValueError(
				f"Expected {expected_num_queries} text queries, got {emb.size(0)} in {emb_path}"
			)

		embedding_map[crop] = emb.float().to(device)

	return embedding_map


class SecondStageClassifier(nn.Module):
	"""
	이미지 feature 추출 -> 텍스트 query 기반 cross-attention -> 평균 pooling -> 선형 분류.
	"""

	def __init__(
		self,
		num_classes: int,
		backbone_name: str = "mobilenet_v3_large",
		backbone_weights: str = "DEFAULT",
		embed_dim: int = 512,
		num_heads: int = 8,
		attn_dropout: float = 0.0,
		residual_scale: float = 1.0,
		classifier_dropout: float = 0.1,
	):
		super().__init__()
		self.feature_extractor = FeatureExtractor(
			backbone_name=backbone_name,
			weights=backbone_weights,
			embed_dim=embed_dim,
		)
		self.cross_attention = CrossAttentionModule(
			embed_dim=embed_dim,
			num_heads=num_heads,
			attn_dropout=attn_dropout,
			residual_scale=residual_scale,
		)
		self.classifier = nn.Sequential(
			nn.Dropout(classifier_dropout),
			nn.Linear(embed_dim, num_classes),
		)

	def forward(self, images: torch.Tensor, text_queries: torch.Tensor, return_features: bool = False):
		image_features = self.feature_extractor(images)
		enhanced_features = self.cross_attention(image_features, text_queries)
		pooled = enhanced_features.mean(dim=1)
		logits = self.classifier(pooled)
		if return_features:
			return logits, pooled
		return logits


def build_dataloaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader, Sequence[str]]:
	train_transform = build_image_transform(
		image_size=args.image_size,
		apply_augmentation=True,
		hflip_prob=args.aug_hflip_prob,
		vflip_prob=args.aug_vflip_prob,
		rotate_deg=args.aug_rotate_deg,
		translate_ratio=args.aug_translate_ratio,
		scale_min=args.aug_scale_min,
		scale_max=args.aug_scale_max,
		shear_deg=args.aug_shear_deg,
	)
	val_transform = build_image_transform(
		image_size=args.image_size,
		apply_augmentation=False,
	)
	class_names = parse_class_names(args.class_names)

	train_loader = build_classification_dataloader(
		root_dir=args.train_dir,
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=args.num_workers,
		transform=train_transform,
		class_names=class_names,
		use_default_classes=not args.no_default_classes,
		return_path=False,
		max_samples=args.max_train_samples,
		seed=args.seed,
	)

	val_loader = build_classification_dataloader(
		root_dir=args.val_dir,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.num_workers,
		transform=val_transform,
		class_names=getattr(train_loader.dataset, "class_names", class_names),
		use_default_classes=False,
		return_path=False,
		max_samples=args.max_val_samples,
		seed=args.seed,
	)

	return train_loader, val_loader, train_loader.dataset.class_names


def build_batch_text_queries(
	batch_class_names: Sequence[str],
	embedding_map: Dict[str, torch.Tensor],
	device: torch.device,
) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
	text_queries: List[torch.Tensor] = []
	crops: List[str] = []
	healthy_flags: List[bool] = []

	for class_name in batch_class_names:
		crop, is_healthy = parse_gt_class_name(class_name)
		if crop not in embedding_map:
			raise KeyError(f"Text embedding not loaded for crop: {crop}")
		text_queries.append(embedding_map[crop])
		crops.append(crop)
		healthy_flags.append(is_healthy)

	batch_text_queries = torch.stack(text_queries, dim=0).to(device)
	healthy_mask = torch.tensor(healthy_flags, dtype=torch.bool, device=device)
	return batch_text_queries, crops, healthy_mask


def sample_mismatch_queries(
	batch_crops: Sequence[str],
	embedding_map: Dict[str, torch.Tensor],
	mismatch_ratio: float,
	device: torch.device,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
	"""배치 내 소수 샘플에 대해 작물이 다른 텍스트 쿼리를 랜덤 매칭한다."""
	all_crops = list(embedding_map.keys())
	if mismatch_ratio <= 0.0 or len(all_crops) < 2:
		return None, None

	queries: List[torch.Tensor] = []
	mismatch_mask: List[bool] = []

	for crop in batch_crops:
		use_mismatch = random.random() < mismatch_ratio
		if use_mismatch:
			candidates = [c for c in all_crops if c != crop]
			if candidates:
				picked = random.choice(candidates)
				queries.append(embedding_map[picked])
				mismatch_mask.append(True)
				continue

		queries.append(embedding_map[crop])
		mismatch_mask.append(False)

	if not any(mismatch_mask):
		return None, None

	return (
		torch.stack(queries, dim=0).to(device),
		torch.tensor(mismatch_mask, dtype=torch.bool, device=device),
	)


def compute_alignment_losses(
	image_features: torch.Tensor,
	matched_text_queries: torch.Tensor,
	healthy_mask: torch.Tensor,
	far_margin: float,
	mismatch_text_queries: Optional[torch.Tensor] = None,
	mismatch_mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
	"""요청한 규칙에 따른 이미지-텍스트 정렬 손실을 계산한다."""
	img = F.normalize(image_features, dim=-1)
	text = F.normalize(matched_text_queries, dim=-1)
	sims = (text * img.unsqueeze(1)).sum(dim=-1)  # (B, 5)

	loss_q0_close = (1.0 - sims[:, 0]).mean()

	rest = sims[:, 1:]
	disease_mask = ~healthy_mask

	if healthy_mask.any():
		loss_rest_healthy_far = F.relu(rest[healthy_mask] - far_margin).mean()
	else:
		loss_rest_healthy_far = torch.zeros((), device=image_features.device)

	if disease_mask.any():
		loss_rest_disease_close = (1.0 - rest[disease_mask]).mean()
	else:
		loss_rest_disease_close = torch.zeros((), device=image_features.device)

	if mismatch_text_queries is not None and mismatch_mask is not None and mismatch_mask.any():
		mtext = F.normalize(mismatch_text_queries, dim=-1)
		m_sims = (mtext[mismatch_mask] * img[mismatch_mask].unsqueeze(1)).sum(dim=-1)
		loss_mismatch_far = F.relu(m_sims - far_margin).mean()
	else:
		loss_mismatch_far = torch.zeros((), device=image_features.device)

	return {
		"q0_close": loss_q0_close,
		"rest_healthy_far": loss_rest_healthy_far,
		"rest_disease_close": loss_rest_disease_close,
		"mismatch_far": loss_mismatch_far,
	}


def run_one_epoch(
	model: nn.Module,
	loader: DataLoader,
	criterion: nn.Module,
	device: torch.device,
	embedding_map: Dict[str, torch.Tensor],
	far_margin: float,
	mismatch_ratio: float,
	loss_weight_cls: float,
	loss_weight_q0: float,
	loss_weight_rest: float,
	loss_weight_mismatch: float,
	loss_weight_mismatch_conf: float,
	epoch: int,
	total_epochs: int,
	phase: str,
	optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, float]:
	training = optimizer is not None
	model.train(training)

	total_loss = 0.0
	total_cls_loss = 0.0
	total_q0_loss = 0.0
	total_rest_loss = 0.0
	total_mismatch_loss = 0.0
	total_mismatch_conf_loss = 0.0
	correct = 0
	total = 0

	pbar = tqdm(
		loader,
		desc=f"{phase} {epoch:03d}/{total_epochs:03d}",
		leave=False,
		dynamic_ncols=True,
	)

	for batch in pbar:
		images = batch["images"].to(device, non_blocking=True)
		labels = batch["labels"].to(device, non_blocking=True)
		matched_text_queries, crops, healthy_mask = build_batch_text_queries(
			batch_class_names=batch["class_names"],
			embedding_map=embedding_map,
			device=device,
		)

		mismatch_text_queries = None
		mismatch_mask = None
		if training and mismatch_ratio > 0.0:
			mismatch_text_queries, mismatch_mask = sample_mismatch_queries(
				batch_crops=crops,
				embedding_map=embedding_map,
				mismatch_ratio=mismatch_ratio,
				device=device,
			)

		if training:
			optimizer.zero_grad(set_to_none=True)

		logits, image_features = model(images, matched_text_queries, return_features=True)
		loss_cls = criterion(logits, labels)

		align_losses = compute_alignment_losses(
			image_features=image_features,
			matched_text_queries=matched_text_queries,
			healthy_mask=healthy_mask,
			far_margin=far_margin,
			mismatch_text_queries=mismatch_text_queries,
			mismatch_mask=mismatch_mask,
		)

		if mismatch_text_queries is not None and mismatch_mask is not None and mismatch_mask.any():
			# mismatch confidence는 mismatch로 치환된 샘플 subset에만 계산해서
			# 불필요한 전체 배치 2회 forward를 피한다.
			mismatch_images = images[mismatch_mask]
			mismatch_text_subset = mismatch_text_queries[mismatch_mask]
			mismatch_logits = model(mismatch_images, mismatch_text_subset)
			mismatch_probs = torch.softmax(mismatch_logits, dim=1)
			loss_mismatch_conf = mismatch_probs.max(dim=1).values.mean()
		else:
			loss_mismatch_conf = torch.zeros((), device=device)

		loss_rest = align_losses["rest_healthy_far"] + align_losses["rest_disease_close"]
		loss = (
			loss_weight_cls * loss_cls
			+ loss_weight_q0 * align_losses["q0_close"]
			+ loss_weight_rest * loss_rest
			+ loss_weight_mismatch * align_losses["mismatch_far"]
			+ loss_weight_mismatch_conf * loss_mismatch_conf
		)

		if training:
			loss.backward()
			optimizer.step()

		total_loss += loss.item() * images.size(0)
		total_cls_loss += loss_cls.item() * images.size(0)
		total_q0_loss += align_losses["q0_close"].item() * images.size(0)
		total_rest_loss += loss_rest.item() * images.size(0)
		total_mismatch_loss += align_losses["mismatch_far"].item() * images.size(0)
		total_mismatch_conf_loss += loss_mismatch_conf.item() * images.size(0)
		preds = logits.argmax(dim=1)
		correct += (preds == labels).sum().item()
		total += labels.size(0)

		pbar.set_postfix(
			loss=f"{loss.item():.4f}",
			cls=f"{loss_cls.item():.4f}",
			q0=f"{align_losses['q0_close'].item():.4f}",
			rest=f"{loss_rest.item():.4f}",
			mis=f"{align_losses['mismatch_far'].item():.4f}",
			misconf=f"{loss_mismatch_conf.item():.4f}",
		)

	pbar.close()

	avg_loss = total_loss / max(total, 1)
	acc = correct / max(total, 1)
	return {
		"loss": avg_loss,
		"cls_loss": total_cls_loss / max(total, 1),
		"q0_loss": total_q0_loss / max(total, 1),
		"rest_loss": total_rest_loss / max(total, 1),
		"mismatch_loss": total_mismatch_loss / max(total, 1),
		"mismatch_conf_loss": total_mismatch_conf_loss / max(total, 1),
		"acc": acc,
	}


def save_checkpoint(
	save_path: Path,
	model: nn.Module,
	optimizer: torch.optim.Optimizer,
	epoch: int,
	class_names: Sequence[str],
	args: argparse.Namespace,
	val_acc: float,
	val_loss: float,
	best_val_acc: float,
	best_val_loss: float,
	is_best: bool,
) -> None:
	save_path.parent.mkdir(parents=True, exist_ok=True)
	torch.save(
		{
			"epoch": epoch,
			"model_state_dict": model.state_dict(),
			"optimizer_state_dict": optimizer.state_dict(),
			"class_names": list(class_names),
			"val_acc": val_acc,
			"val_loss": val_loss,
			"best_val_acc": best_val_acc,
			"best_val_loss": best_val_loss,
			"is_best": is_best,
			"config": vars(args),
		},
		str(save_path),
	)


def main(args: argparse.Namespace) -> None:
	set_seed(args.seed)

	device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
	train_loader, val_loader, class_names = build_dataloaders(args)
	all_crops = [parse_gt_class_name(c)[0] for c in class_names]
	embedding_map = load_crop_text_embeddings(
		text_embedding_dir=Path(args.text_embedding_dir),
		crop_names=all_crops,
		device=device,
		expected_num_queries=5,
	)

	any_crop = next(iter(embedding_map.keys()))
	text_embed_dim = embedding_map[any_crop].size(1)
	if text_embed_dim != args.embed_dim:
		raise ValueError(
			f"embed_dim ({args.embed_dim}) must match text embedding dim ({text_embed_dim})"
		)

	model = SecondStageClassifier(
		num_classes=len(class_names),
		backbone_name=args.backbone_name,
		backbone_weights=args.backbone_weights,
		embed_dim=args.embed_dim,
		num_heads=args.num_heads,
		attn_dropout=args.attn_dropout,
		residual_scale=args.residual_scale,
		classifier_dropout=args.classifier_dropout,
	).to(device)

	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.AdamW(
		model.parameters(),
		lr=args.learning_rate,
		weight_decay=args.weight_decay,
	)

	best_val_acc = -1.0
	best_val_loss = float("inf")
	best_save_path = Path(args.save_path)
	last_save_path = derive_last_save_path(best_save_path)
	log_path = derive_log_path(best_save_path)

	append_log(log_path, "=" * 80)
	append_log(log_path, f"Start Time: {datetime.now().isoformat(timespec='seconds')}")
	append_log(log_path, f"Best Checkpoint Path: {best_save_path}")
	append_log(log_path, f"Last Checkpoint Path: {last_save_path}")
	append_log(log_path, f"Seed: {args.seed}")
	append_log(log_path, f"Config: {vars(args)}")
	append_log(log_path, "=" * 80)
	for epoch in range(1, args.epochs + 1):
		train_metrics = run_one_epoch(
			model=model,
			loader=train_loader,
			criterion=criterion,
			device=device,
			embedding_map=embedding_map,
			far_margin=args.far_margin,
			mismatch_ratio=args.mismatch_ratio,
			loss_weight_cls=args.loss_weight_cls,
			loss_weight_q0=args.loss_weight_q0,
			loss_weight_rest=args.loss_weight_rest,
			loss_weight_mismatch=args.loss_weight_mismatch,
			loss_weight_mismatch_conf=args.loss_weight_mismatch_conf,
			epoch=epoch,
			total_epochs=args.epochs,
			phase="Train",
			optimizer=optimizer,
		)
		val_metrics = run_one_epoch(
			model=model,
			loader=val_loader,
			criterion=criterion,
			device=device,
			embedding_map=embedding_map,
			far_margin=args.far_margin,
			mismatch_ratio=0.0,
			loss_weight_cls=args.loss_weight_cls,
			loss_weight_q0=args.loss_weight_q0,
			loss_weight_rest=args.loss_weight_rest,
			loss_weight_mismatch=args.loss_weight_mismatch,
			loss_weight_mismatch_conf=args.loss_weight_mismatch_conf,
			epoch=epoch,
			total_epochs=args.epochs,
			phase="Val",
			optimizer=None,
		)

		epoch_message = (
			f"[Epoch {epoch:03d}/{args.epochs:03d}] "
			f"train_loss={train_metrics['loss']:.4f} "
			f"train_cls={train_metrics['cls_loss']:.4f} "
			f"train_q0={train_metrics['q0_loss']:.4f} "
			f"train_rest={train_metrics['rest_loss']:.4f} "
			f"train_mis={train_metrics['mismatch_loss']:.4f} "
			f"train_misconf={train_metrics['mismatch_conf_loss']:.4f} "
			f"train_acc={train_metrics['acc']:.4f} "
			f"val_loss={val_metrics['loss']:.4f} "
			f"val_cls={val_metrics['cls_loss']:.4f} "
			f"val_q0={val_metrics['q0_loss']:.4f} "
			f"val_rest={val_metrics['rest_loss']:.4f} "
			f"val_misconf={val_metrics['mismatch_conf_loss']:.4f} "
			f"val_acc={val_metrics['acc']:.4f}"
		)
		print(epoch_message)
		append_log(log_path, epoch_message)

		save_checkpoint(
			save_path=last_save_path,
			model=model,
			optimizer=optimizer,
			epoch=epoch,
			class_names=class_names,
			args=args,
			val_acc=val_metrics["acc"],
			val_loss=val_metrics["loss"],
			best_val_acc=best_val_acc,
			best_val_loss=best_val_loss,
			is_best=False,
		)
		last_message = f"Updated last checkpoint: {last_save_path}"
		print(last_message)
		append_log(log_path, last_message)

		is_better_acc = val_metrics["acc"] > best_val_acc
		is_tie_acc_better_loss = (
			abs(val_metrics["acc"] - best_val_acc) <= 1e-12 and val_metrics["loss"] < best_val_loss
		)
		if is_better_acc or is_tie_acc_better_loss:
			best_val_acc = val_metrics["acc"]
			best_val_loss = val_metrics["loss"]
			save_checkpoint(
				save_path=best_save_path,
				model=model,
				optimizer=optimizer,
				epoch=epoch,
				class_names=class_names,
				args=args,
				val_acc=val_metrics["acc"],
				val_loss=val_metrics["loss"],
				best_val_acc=best_val_acc,
				best_val_loss=best_val_loss,
				is_best=True,
			)
			best_message = f"Saved best checkpoint to: {best_save_path}"
			print(best_message)
			append_log(log_path, best_message)

	print("Training finished.")
	print(f"Best validation accuracy: {best_val_acc:.4f}")
	print(f"Best validation loss (tie-break): {best_val_loss:.4f}")
	print(f"Num classes: {len(class_names)}")
	print("Class order:", ", ".join(class_names))
	append_log(log_path, "Training finished.")
	append_log(log_path, f"Best validation accuracy: {best_val_acc:.4f}")
	append_log(log_path, f"Best validation loss (tie-break): {best_val_loss:.4f}")
	append_log(log_path, f"Num classes: {len(class_names)}")
	append_log(log_path, "Class order: " + ", ".join(class_names))
	append_log(log_path, f"End Time: {datetime.now().isoformat(timespec='seconds')}")


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Second-stage classifier training script")

	parser.add_argument("--train_dir", type=str, default=str(DEFAULT_TRAIN_DIR), help="Train root folder")
	parser.add_argument("--val_dir", type=str, default=str(DEFAULT_VAL_DIR), help="Validation root folder")
	parser.add_argument(
		"--text_embedding_dir",
		type=str,
		default=str(DEFAULT_TEXT_EMBEDDING_DIR),
		help="Directory containing crop text embedding .pt files",
	)
	parser.add_argument("--save_path", type=str, default=str(DEFAULT_SAVE_PATH), help="Best model checkpoint path")

	parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
	parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
	parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
	parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
	parser.add_argument("--num_workers", type=int, default=4, help="DataLoader num_workers")

	parser.add_argument("--image_size", type=int, default=224, help="Input image size")
	parser.add_argument("--aug_hflip_prob", type=float, default=0.5, help="Train-only horizontal flip probability")
	parser.add_argument("--aug_vflip_prob", type=float, default=0.1, help="Train-only vertical flip probability")
	parser.add_argument("--aug_rotate_deg", type=float, default=10.0, help="Train-only max rotation degree")
	parser.add_argument("--aug_translate_ratio", type=float, default=0.05, help="Train-only max translation ratio")
	parser.add_argument("--aug_scale_min", type=float, default=0.95, help="Train-only min scaling factor")
	parser.add_argument("--aug_scale_max", type=float, default=1.05, help="Train-only max scaling factor")
	parser.add_argument("--aug_shear_deg", type=float, default=5.0, help="Train-only max shear degree")
	parser.add_argument("--backbone_name", type=str, default="mobilenet_v3_large", help="Feature extractor backbone")
	parser.add_argument("--backbone_weights", type=str, default="DEFAULT", help="Torchvision backbone weights")
	parser.add_argument("--embed_dim", type=int, default=512, help="Embedding dimension")
	parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
	parser.add_argument("--attn_dropout", type=float, default=0.0, help="Attention dropout")
	parser.add_argument("--residual_scale", type=float, default=1.0, help="Residual scale for attention output")
	parser.add_argument("--classifier_dropout", type=float, default=0.1, help="Dropout before linear classifier")
	parser.add_argument("--far_margin", type=float, default=0.2, help="Cosine similarity upper bound for far condition")
	parser.add_argument("--mismatch_ratio", type=float, default=0.1, help="Ratio of crop-mismatched pairs in train")
	parser.add_argument("--loss_weight_cls", type=float, default=1.0, help="Weight for classification loss")
	parser.add_argument("--loss_weight_q0", type=float, default=0.5, help="Weight for query-0 close loss")
	parser.add_argument("--loss_weight_rest", type=float, default=0.5, help="Weight for query-1~4 conditional loss")
	parser.add_argument("--loss_weight_mismatch", type=float, default=0.5, help="Weight for mismatch far loss")
	parser.add_argument(
		"--loss_weight_mismatch_conf",
		type=float,
		default=0.2,
		help="Weight for mismatch argmax-confidence suppression loss",
	)

	parser.add_argument(
		"--class_names",
		type=str,
		default=None,
		help="Comma-separated class names. If omitted, default AIHub class list is used.",
	)
	parser.add_argument(
		"--no_default_classes",
		action="store_true",
		help="Do not use built-in default class list when class_names is omitted.",
	)

	parser.add_argument("--max_train_samples", type=int, default=None, help="Limit train samples for quick debug")
	parser.add_argument("--max_val_samples", type=int, default=None, help="Limit val samples for quick debug")
	parser.add_argument("--device", type=str, default=None, help="cuda or cpu. default is auto")
	parser.add_argument("--seed", type=int, default=42, help="Random seed")
	return parser


if __name__ == "__main__":
	parser = build_arg_parser()
	args = parser.parse_args()
	main(args)
