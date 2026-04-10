import json
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


IMAGE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _seed_worker(worker_id: int) -> None:
	"""DataLoader worker마다 동일 규칙으로 난수 시드를 고정한다."""
	worker_seed = torch.initial_seed() % (2 ** 32)
	np.random.seed(worker_seed)
	random.seed(worker_seed)


def _to_xyxy_clipped(
	x: float,
	y: float,
	w: float,
	h: float,
	width: int,
	height: int,
) -> Tuple[int, int, int, int]:
	"""xywh 박스를 이미지 범위에 맞춰 clip한 xyxy 정수 좌표로 변환한다."""
	x1 = int(round(x))
	y1 = int(round(y))
	x2 = int(round(x + w))
	y2 = int(round(y + h))

	x1 = max(0, min(width - 1, x1))
	y1 = max(0, min(height - 1, y1))
	x2 = max(0, min(width, x2))
	y2 = max(0, min(height, y2))
	return x1, y1, x2, y2


def _resolve_image_path(json_path: Path, image_name: Optional[str]) -> Optional[Path]:
	"""JSON과 같은 폴더에서 이미지 파일 경로를 찾는다."""
	if image_name:
		p = json_path.with_name(image_name)
		if p.exists():
			return p

	stem = json_path.stem
	for ext in IMAGE_EXTENSIONS:
		p = json_path.with_name(stem + ext)
		if p.exists():
			return p
	return None


def load_aihubbbox_label(json_path: Path) -> Optional[Dict[str, Any]]:
	"""
	AIHub JSON 1개를 읽어 학습에 필요한 메타데이터 딕셔너리로 변환한다.

	필수 라벨은 annotations.bbox[0]의 x, y, w, h를 사용한다.
	"""
	try:
		with json_path.open("r", encoding="utf-8") as f:
			data = json.load(f)
	except Exception:
		return None

	desc = data.get("description", {})
	ann = data.get("annotations", {})
	bbox_list = ann.get("bbox", [])
	if not bbox_list:
		return None

	bbox = bbox_list[0]
	x = float(bbox.get("x", 0.0))
	y = float(bbox.get("y", 0.0))
	w = float(bbox.get("w", 0.0))
	h = float(bbox.get("h", 0.0))

	width = int(desc.get("width", 0))
	height = int(desc.get("height", 0))
	image_name = desc.get("image")
	image_path = _resolve_image_path(json_path, image_name)
	if image_path is None:
		return None

	if width <= 0 or height <= 0:
		img = cv2.imread(str(image_path))
		if img is None:
			return None
		height, width = img.shape[:2]

	x1, y1, x2, y2 = _to_xyxy_clipped(x, y, w, h, width=width, height=height)
	if x2 <= x1 or y2 <= y1:
		return None

	return {
		"image_path": image_path,
		"json_path": json_path,
		"image_name": image_name if image_name else image_path.name,
		"width": width,
		"height": height,
		"bbox_xywh": (x, y, w, h),
		"bbox_xyxy": (x1, y1, x2, y2),
		"crop": ann.get("crop"),
		"disease": ann.get("disease"),
		"area": ann.get("area"),
		"risk": ann.get("risk"),
		"grow": ann.get("grow"),
	}


class AIHubBBoxDataset(Dataset):
	"""
	AIHub bbox 데이터셋을 읽는 PyTorch Dataset.

	Dataset for AIHub data where each image has a side-by-side JSON label file.

	Expected JSON schema:
		annotations.bbox[0] with fields x, y, w, h
	"""

	def __init__(
		self,
		root_dir: str,
		transform: Optional[Callable[[np.ndarray], Any]] = None,
		return_path: bool = True,
		max_samples: Optional[int] = None,
	):
		"""루트 폴더를 재귀 탐색해 유효한 JSON-이미지 쌍 목록을 만든다."""
		self.root_dir = Path(root_dir)
		self.transform = transform
		self.return_path = return_path
		self.samples: List[Dict[str, Any]] = []

		if not self.root_dir.exists():
			raise FileNotFoundError(f"Dataset root does not exist: {self.root_dir}")

		json_files = sorted(self.root_dir.rglob("*.json"))
		for jp in json_files:
			item = load_aihubbbox_label(jp)
			if item is None:
				continue
			self.samples.append(item)
			if max_samples is not None and max_samples > 0 and len(self.samples) >= max_samples:
				break

		if not self.samples:
			raise RuntimeError(f"No valid AIHub samples found in: {self.root_dir}")

	def __len__(self) -> int:
		"""데이터셋 샘플 개수를 반환한다."""
		return len(self.samples)

	def __getitem__(self, idx: int) -> Dict[str, Any]:
		"""index에 해당하는 이미지와 bbox target 딕셔너리를 반환한다."""
		sample = self.samples[idx]
		image_path: Path = sample["image_path"]
		img = cv2.imread(str(image_path))
		if img is None:
			raise RuntimeError(f"Failed to read image: {image_path}")
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		image: Any = img
		if self.transform is not None:
			image = self.transform(img)

		x1, y1, x2, y2 = sample["bbox_xyxy"]
		target: Dict[str, Any] = {
			"bbox_xywh": torch.tensor(sample["bbox_xywh"], dtype=torch.float32),
			"bbox_xyxy": torch.tensor([x1, y1, x2, y2], dtype=torch.float32),
			"boxes": torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32),
			"labels": torch.tensor([1], dtype=torch.int64),
			"image_size": torch.tensor([sample["height"], sample["width"]], dtype=torch.int64),
			"crop": sample["crop"],
			"disease": sample["disease"],
			"area": sample["area"],
			"risk": sample["risk"],
			"grow": sample["grow"],
		}

		out: Dict[str, Any] = {
			"image": image,
			"target": target,
		}
		if self.return_path:
			out["image_path"] = str(sample["image_path"])
			out["json_path"] = str(sample["json_path"])
		return out


def aihub_collate_fn(batch: Sequence[Dict[str, Any]]) -> Dict[str, List[Any]]:
	"""가변 크기 이미지를 위해 리스트 기반으로 배치를 묶는 collate 함수."""
	images = [b["image"] for b in batch]
	targets = [b["target"] for b in batch]
	out: Dict[str, List[Any]] = {
		"images": images,
		"targets": targets,
	}

	if batch and "image_path" in batch[0]:
		out["image_paths"] = [b["image_path"] for b in batch]
	if batch and "json_path" in batch[0]:
		out["json_paths"] = [b["json_path"] for b in batch]
	return out


def build_aihubbbox_dataloader(
	root_dir: str,
	batch_size: int = 8,
	shuffle: bool = True,
	num_workers: int = 0,
	transform: Optional[Callable[[np.ndarray], Any]] = None,
	return_path: bool = True,
	max_samples: Optional[int] = None,
) -> DataLoader:
	"""AIHubBBoxDataset을 생성하고 표준 설정의 DataLoader를 반환한다."""
	dataset = AIHubBBoxDataset(
		root_dir=root_dir,
		transform=transform,
		return_path=return_path,
		max_samples=max_samples,
	)
	return DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=num_workers,
		collate_fn=aihub_collate_fn,
		pin_memory=torch.cuda.is_available(),
	)


DEFAULT_AIHUB_DISEASE_CLASSES: Tuple[str, ...] = (
	"cucumber_downy",
	"cucumber_healthy",
	"cucumber_powdery",
	"grape_downy",
	"grape_healthy",
	"paprica_healthy",
	"paprica_powdery",
	"pepper_healthy",
	"pepper_powdery",
	"strawberry_healthy",
	"strawberry_powdery",
	"tomato_graymold",
	"tomato_healthy",
	"tomato_powdery",
)


def get_default_aihud_disease_classes() -> List[str]:
	"""현재 프로젝트 기본 분류 클래스 목록을 반환한다."""
	return list(DEFAULT_AIHUB_DISEASE_CLASSES)


class FolderClassificationDataset(Dataset):
	"""클래스별 하위 폴더 구조(root/class_name/*.jpg)를 읽는 분류용 Dataset."""

	def __init__(
		self,
		root_dir: str,
		class_names: Optional[Sequence[str]] = None,
		transform: Optional[Callable[[np.ndarray], Any]] = None,
		return_path: bool = False,
		max_samples: Optional[int] = None,
	):
		self.root_dir = Path(root_dir)
		self.transform = transform
		self.return_path = return_path

		if not self.root_dir.exists():
			raise FileNotFoundError(f"Dataset root does not exist: {self.root_dir}")

		discovered_classes = sorted([p.name for p in self.root_dir.iterdir() if p.is_dir()])
		if class_names is None:
			self.class_names = discovered_classes
		else:
			# 요청한 순서를 우선 유지하고, 실제 폴더가 있는 클래스만 사용한다.
			requested = list(class_names)
			self.class_names = [c for c in requested if (self.root_dir / c).is_dir()]

			# 클래스 확장에 대비해 요청 목록에 없는 신규 폴더는 뒤에 자동 추가한다.
			for c in discovered_classes:
				if c not in self.class_names:
					self.class_names.append(c)

		if not self.class_names:
			raise RuntimeError(f"No class folders found in: {self.root_dir}")

		self.class_to_idx: Dict[str, int] = {c: i for i, c in enumerate(self.class_names)}
		self.samples: List[Tuple[Path, int]] = []

		for class_name in self.class_names:
			class_dir = self.root_dir / class_name
			if not class_dir.exists() or not class_dir.is_dir():
				continue
			for ext in IMAGE_EXTENSIONS:
				for image_path in class_dir.rglob(f"*{ext}"):
					self.samples.append((image_path, self.class_to_idx[class_name]))
					if max_samples is not None and max_samples > 0 and len(self.samples) >= max_samples:
						break
				if max_samples is not None and max_samples > 0 and len(self.samples) >= max_samples:
					break
			if max_samples is not None and max_samples > 0 and len(self.samples) >= max_samples:
				break

		if not self.samples:
			raise RuntimeError(f"No image files found in: {self.root_dir}")

	def __len__(self) -> int:
		return len(self.samples)

	def __getitem__(self, idx: int) -> Dict[str, Any]:
		image_path, label = self.samples[idx]
		img = cv2.imread(str(image_path))
		if img is None:
			raise RuntimeError(f"Failed to read image: {image_path}")
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		image: Any = img
		if self.transform is not None:
			image = self.transform(img)
		else:
			image = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

		out: Dict[str, Any] = {
			"image": image,
			"label": torch.tensor(label, dtype=torch.long),
			"class_name": self.class_names[label],
		}
		if self.return_path:
			out["image_path"] = str(image_path)
		return out


def classification_collate_fn(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
	"""고정 크기 이미지 텐서를 stack하는 분류용 collate 함수."""
	images = torch.stack([b["image"] for b in batch], dim=0)
	labels = torch.stack([b["label"] for b in batch], dim=0)
	class_names = [b["class_name"] for b in batch]

	out: Dict[str, Any] = {
		"images": images,
		"labels": labels,
		"class_names": class_names,
	}
	if batch and "image_path" in batch[0]:
		out["image_paths"] = [b["image_path"] for b in batch]
	return out


def build_classification_dataloader(
	root_dir: str,
	batch_size: int = 16,
	shuffle: bool = True,
	num_workers: int = 0,
	transform: Optional[Callable[[np.ndarray], Any]] = None,
	class_names: Optional[Sequence[str]] = None,
	use_default_classes: bool = True,
	return_path: bool = False,
	max_samples: Optional[int] = None,
	seed: Optional[int] = 42,
) -> DataLoader:
	"""폴더 구조 기반 이미지 분류 DataLoader를 생성한다."""
	selected_class_names: Optional[Sequence[str]] = class_names
	if selected_class_names is None and use_default_classes:
		selected_class_names = DEFAULT_AIHUB_DISEASE_CLASSES

	dataset = FolderClassificationDataset(
		root_dir=root_dir,
		class_names=selected_class_names,
		transform=transform,
		return_path=return_path,
		max_samples=max_samples,
	)

	generator: Optional[torch.Generator] = None
	worker_init_fn: Optional[Callable[[int], None]] = None
	if seed is not None:
		generator = torch.Generator()
		generator.manual_seed(seed)
		worker_init_fn = _seed_worker

	return DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=num_workers,
		collate_fn=classification_collate_fn,
		pin_memory=torch.cuda.is_available(),
		worker_init_fn=worker_init_fn,
		generator=generator,
	)

