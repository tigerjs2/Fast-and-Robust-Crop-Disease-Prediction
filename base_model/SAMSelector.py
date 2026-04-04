"""
SAM Mask Selector - CNN 기반 마스크 품질 평가 모델

마스킹된 이미지(배경 회색, 객체만 컬러)를 입력으로 받아
"적절한 마스크인지" 확률값을 출력하는 경량 CNN.

MobileNetV3보다 훨씬 작고 빠르면서도 마스크 품질을 정확하게 평가.
"""

import argparse
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from dataset.datasets import AIHubBBoxDataset


DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "weights" / "samselector"
DEFAULT_TRAINSET_CACHE_DIR = PROJECT_ROOT / "data" / "selector_trainset_cnn"

# 입력 이미지 크기 (작을수록 빠르고 가벼움)
INPUT_SIZE = 64
GRAY_BG_VALUE = 128  # 배경 회색 값


class TinyMaskScorer(nn.Module):
    """
    마스킹된 이미지를 입력으로 받아 품질 점수를 예측하는 경량 CNN.
    
    Architecture (~30K params):
        Conv3x3(3→16) → BN → ReLU → MaxPool
        Conv3x3(16→32) → BN → ReLU → MaxPool
        Conv3x3(32→64) → BN → ReLU → MaxPool
        GlobalAvgPool → FC(64→64) → ReLU → Dropout → FC(64→1)
    
    Input: (B, 3, 64, 64) - 마스킹된 이미지 (배경 회색)
    Output: (B, 1) - 품질 점수 logit
    """
    
    def __init__(self, dropout: float = 0.3):
        super().__init__()
        
        # Feature extractor
        self.features = nn.Sequential(
            # Block 1: 64 → 32
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 2: 32 → 16
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 3: 16 → 8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 4: 8 → 4
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # Global Average Pooling
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 64, 64) → (B, 1) logit"""
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def count_parameters(self) -> int:
        """학습 가능한 파라미터 수 반환"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MaskedImageDataset(Dataset):
    """마스킹된 이미지, target, bbox 메트릭을 담는 Dataset."""
    
    def __init__(self, images: np.ndarray, targets: np.ndarray, bbox_metrics: Optional[np.ndarray] = None):
        """
        Args:
            images: (N, H, W, 3) uint8 배열 (BGR)
            targets: (N,) float32 배열
            bbox_metrics: (N, 4) float32 배열 [inside_ratio, gt_coverage, bbox_iou, size_ratio]
        """
        self.images = images
        self.targets = torch.from_numpy(targets.astype(np.float32)).unsqueeze(1)
        if bbox_metrics is not None:
            self.bbox_metrics = torch.from_numpy(bbox_metrics.astype(np.float32))
        else:
            # 메트릭이 없으면 더미 값 (호환성)
            self.bbox_metrics = torch.zeros(len(images), 4, dtype=torch.float32)
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img = self.images[idx]
        # BGR → RGB, HWC → CHW, normalize to [0, 1]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img, self.targets[idx], self.bbox_metrics[idx]


class SAMMaskCandidateGenerator:
    """Ultralytics SAM으로 이미지별 후보 마스크들을 생성하는 래퍼."""

    def __init__(self, model_path: str):
        """SAM 가중치를 로드해 추론 객체를 초기화한다."""
        from ultralytics import SAM

        self.model = SAM(model_path)

    @torch.no_grad()
    def generate_masks(self, image_path: str) -> List[np.ndarray]:
        """입력 이미지에서 후보 마스크들을 bool 배열(H, W) 리스트로 반환한다."""
        image = cv2.imread(image_path)
        if image is None:
            return []

        height, width = image.shape[:2]
        results = self.model(image_path, verbose=False)
        if not results or results[0].masks is None or results[0].masks.data is None:
            return []

        masks = results[0].masks.data.detach().cpu().numpy()
        out: List[np.ndarray] = []
        for m in masks:
            resized = cv2.resize(m.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
            out.append(resized > 0)
        return out


def create_masked_image(
    image: np.ndarray,
    mask: np.ndarray,
    output_size: int = INPUT_SIZE,
    bg_value: int = GRAY_BG_VALUE,
) -> np.ndarray:
    """
    마스크를 적용해 배경을 회색으로 만든 이미지 생성.
    
    Args:
        image: (H, W, 3) BGR 이미지
        mask: (H, W) bool 마스크
        output_size: 출력 이미지 크기
        bg_value: 배경 회색 값
    
    Returns:
        (output_size, output_size, 3) uint8 BGR 이미지
    """
    h, w = image.shape[:2]
    
    # 배경을 회색으로
    result = np.full_like(image, bg_value, dtype=np.uint8)
    result[mask] = image[mask]
    
    # 마스크 영역의 bounding box로 crop (여백 포함)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        # 빈 마스크면 회색 이미지 반환
        return cv2.resize(result, (output_size, output_size))
    
    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()) + 1, int(ys.max()) + 1
    
    # 약간의 여백 추가 (10%)
    pad_x = max(1, int((x2 - x1) * 0.1))
    pad_y = max(1, int((y2 - y1) * 0.1))
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    
    # Crop and resize
    cropped = result[y1:y2, x1:x2]
    resized = cv2.resize(cropped, (output_size, output_size), interpolation=cv2.INTER_AREA)
    
    return resized


def compute_bbox_metrics(
    mask: np.ndarray,
    gt_bbox: Tuple[int, int, int, int],
) -> Tuple[float, float, float, float]:
    """
    마스크와 GT bbox 간의 메트릭 계산.
    
    Returns:
        inside_ratio: mask 픽셀 중 bbox 안에 있는 비율 (containment)
        gt_coverage: bbox를 mask가 채우는 비율 (coverage)
        bbox_iou: mask bbox와 GT bbox의 IoU
        size_ratio: mask 면적 / GT bbox 면적
    """
    gx1, gy1, gx2, gy2 = gt_bbox
    gt_area = float(max(0, gx2 - gx1) * max(0, gy2 - gy1))
    mask_area = float(mask.sum())
    
    if mask_area == 0 or gt_area == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    mask_in_gt = float(mask[gy1:gy2, gx1:gx2].sum())
    
    inside_ratio = mask_in_gt / mask_area
    gt_coverage = mask_in_gt / gt_area
    size_ratio = mask_area / gt_area
    
    # bbox IoU 계산
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return inside_ratio, gt_coverage, 0.0, size_ratio
    
    mx1, my1 = int(xs.min()), int(ys.min())
    mx2, my2 = int(xs.max()) + 1, int(ys.max()) + 1
    
    # intersection
    ix1, iy1 = max(mx1, gx1), max(my1, gy1)
    ix2, iy2 = min(mx2, gx2), min(my2, gy2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = float(iw * ih)
    
    # union
    mask_bbox_area = float((mx2 - mx1) * (my2 - my1))
    union = mask_bbox_area + gt_area - inter
    bbox_iou = inter / max(union, 1.0)
    
    return inside_ratio, gt_coverage, bbox_iou, size_ratio


def compute_mask_quality_target(
    mask: np.ndarray,
    gt_bbox: Tuple[int, int, int, int],
) -> float:
    """
    마스크와 GT bbox를 비교해 품질 점수(target)를 계산.
    
    Target = inside_ratio * gt_coverage * size_penalty
    - inside_ratio: mask가 GT bbox 안에 있는 비율
    - gt_coverage: GT bbox를 mask가 채우는 비율
    - size_penalty: 너무 작은 mask에 패널티
    """
    gx1, gy1, gx2, gy2 = gt_bbox
    gt_area = float(max(0, gx2 - gx1) * max(0, gy2 - gy1))
    mask_area = float(mask.sum())
    
    if mask_area == 0 or gt_area == 0:
        return 0.0
    
    mask_in_gt = float(mask[gy1:gy2, gx1:gx2].sum())
    
    inside_ratio = mask_in_gt / mask_area
    gt_coverage = mask_in_gt / gt_area
    size_ratio = mask_area / gt_area
    size_penalty = min(1.0, size_ratio / 0.3)
    
    target = inside_ratio * gt_coverage * size_penalty
    return float(np.clip(target, 0.0, 1.0))


def build_training_data(
    annotation_root: str,
    sam_model_path: str,
    max_images: Optional[int] = None,
    max_masks_per_image: int = 10,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    학습용 마스킹된 이미지, target, bbox 메트릭을 생성.
    
    Returns:
        images: (N, INPUT_SIZE, INPUT_SIZE, 3) uint8 배열
        targets: (N,) float32 배열
        bbox_metrics: (N, 4) float32 배열 [inside_ratio, gt_coverage, bbox_iou, size_ratio]
    """
    random.seed(seed)
    np.random.seed(seed)

    dataset = AIHubBBoxDataset(
        root_dir=annotation_root,
        transform=None,
        return_path=True,
        max_samples=max_images,
    )
    samples = list(dataset.samples)
    random.shuffle(samples)

    generator = SAMMaskCandidateGenerator(sam_model_path)
    images_list: List[np.ndarray] = []
    targets_list: List[float] = []
    metrics_list: List[Tuple[float, float, float, float]] = []

    for sample in tqdm(samples, desc="Build CNN trainset"):
        image_path = str(sample["image_path"])
        gt_bbox = tuple(sample["bbox_xyxy"])
        
        # 원본 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            continue
        
        # SAM으로 후보 마스크 생성
        masks = generator.generate_masks(image_path)
        if not masks:
            continue

        # 마스크 수 제한
        if len(masks) > max_masks_per_image:
            idx = np.random.choice(len(masks), size=max_masks_per_image, replace=False)
            masks = [masks[i] for i in idx]

        for mask in masks:
            if mask.sum() == 0:
                continue
            
            # 마스킹된 이미지 생성
            masked_img = create_masked_image(image, mask, INPUT_SIZE, GRAY_BG_VALUE)
            
            # 품질 target 계산
            target = compute_mask_quality_target(mask, gt_bbox)
            
            # bbox 메트릭 계산 (loss에서 사용)
            metrics = compute_bbox_metrics(mask, gt_bbox)
            
            images_list.append(masked_img)
            targets_list.append(target)
            metrics_list.append(metrics)

    if not images_list:
        raise RuntimeError("No training samples were built.")

    images = np.stack(images_list, axis=0)
    targets = np.array(targets_list, dtype=np.float32)
    bbox_metrics = np.array(metrics_list, dtype=np.float32)
    
    return images, targets, bbox_metrics


def save_trainset_cache(images: np.ndarray, targets: np.ndarray, bbox_metrics: np.ndarray, cache_dir: str) -> None:
    """학습셋을 디스크에 저장."""
    path = Path(cache_dir)
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "images.npy", images)
    np.save(path / "targets.npy", targets)
    np.save(path / "bbox_metrics.npy", bbox_metrics)
    print(f"[Cache] Saved {len(images)} samples to {path}")


def load_trainset_cache(cache_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """학습셋을 디스크에서 로드."""
    path = Path(cache_dir)
    images = np.load(path / "images.npy")
    targets = np.load(path / "targets.npy")
    bbox_metrics = np.load(path / "bbox_metrics.npy")
    print(f"[Cache] Loaded {len(images)} samples from {path}")
    return images, targets, bbox_metrics


def get_or_build_training_data(
    annotation_root: str,
    sam_model_path: str,
    cache_dir: str,
    max_images: Optional[int] = None,
    max_masks_per_image: int = 10,
    seed: int = 42,
    rebuild_cache: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """캐시가 있으면 로드, 없으면 생성."""
    cache_path = Path(cache_dir)
    images_file = cache_path / "images.npy"
    targets_file = cache_path / "targets.npy"
    metrics_file = cache_path / "bbox_metrics.npy"
    
    if images_file.exists() and targets_file.exists() and metrics_file.exists() and not rebuild_cache:
        return load_trainset_cache(cache_dir)
    
    print(f"[Cache] Building new trainset...")
    images, targets, bbox_metrics = build_training_data(
        annotation_root=annotation_root,
        sam_model_path=sam_model_path,
        max_images=max_images,
        max_masks_per_image=max_masks_per_image,
        seed=seed,
    )
    save_trainset_cache(images, targets, bbox_metrics, cache_dir)
    return images, targets, bbox_metrics


def train_mask_selector(
    images: np.ndarray,
    targets: np.ndarray,
    bbox_metrics: Optional[np.ndarray] = None,
    dropout: float = 0.3,
    batch_size: int = 64,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    val_ratio: float = 0.2,
    seed: int = 42,
    device: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
) -> Tuple[TinyMaskScorer, Dict[str, Any]]:
    """
    CNN 모델 학습.
    
    bbox_metrics가 제공되면 GT bbox 정보를 활용한 BBoxAwareLoss를 사용.
    bbox_metrics 열 순서: [inside_ratio, gt_coverage, bbox_iou, size_ratio]
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    n = len(images)
    idx = np.arange(n)
    np.random.shuffle(idx)
    images = images[idx]
    targets = targets[idx]
    if bbox_metrics is not None:
        bbox_metrics = bbox_metrics[idx]

    val_size = max(1, int(n * val_ratio))
    train_size = n - val_size

    x_train, y_train = images[:train_size], targets[:train_size]
    x_val, y_val = images[train_size:], targets[train_size:]
    
    if bbox_metrics is not None:
        m_train, m_val = bbox_metrics[:train_size], bbox_metrics[train_size:]
    else:
        m_train, m_val = None, None

    train_ds = MaskedImageDataset(x_train, y_train, m_train)
    val_ds = MaskedImageDataset(x_val, y_val, m_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    run_device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    
    model = TinyMaskScorer(dropout=dropout).to(run_device)
    print(f"[Model] TinyMaskScorer - {model.count_parameters():,} parameters")

    # BBox-Aware Loss: GT bbox 정보를 직접 활용
    def bbox_aware_loss(
        logits: torch.Tensor, 
        targets: torch.Tensor, 
        metrics: torch.Tensor
    ) -> torch.Tensor:
        """
        GT Bounding Box 정보를 직접 활용하는 loss.
        
        Loss 구성:
        1. Quality BCE: 예측 점수와 품질 target의 BCE
        2. Containment Loss: mask가 bbox 안에 있도록 (밖으로 나가면 패널티)
        3. Coverage Loss: 예측이 coverage와 일치하도록 (bbox 채움 정도 예측)
        4. IoU Alignment: 예측이 bbox_iou와 상관관계를 갖도록
        
        metrics 열: [inside_ratio, gt_coverage, bbox_iou, size_ratio]
        """
        pred = torch.sigmoid(logits)
        
        inside_ratio = metrics[:, 0:1]   # mask 중 bbox 안에 있는 비율
        gt_coverage = metrics[:, 1:2]    # bbox를 mask가 채우는 비율
        bbox_iou = metrics[:, 2:3]       # mask bbox와 GT bbox의 IoU
        
        # 1. 기본 BCE Loss (품질 점수 예측)
        bce = F.binary_cross_entropy(pred, targets, reduction='none')
        
        # 2. Containment Penalty: bbox 밖으로 삐져나가는 mask가 높은 점수를 받으면 패널티
        # inside_ratio가 낮을수록 (밖에 많이 있을수록) 높은 pred에 패널티
        outside_ratio = 1.0 - inside_ratio
        containment_penalty = outside_ratio * pred * 2.0  # 밖에 있는 비율 * 예측 점수
        
        # 3. Coverage Alignment: 예측이 gt_coverage와 일치하도록
        # box 모드는 bbox를 잘 채우는 mask를 생성하므로, 그런 mask에 높은 점수를 주도록
        coverage_loss = F.mse_loss(pred, gt_coverage, reduction='none')
        
        # 4. IoU Consistency: bbox_iou가 높은 mask는 높은 점수를 받도록
        # 예측과 IoU 간의 일관성 유도
        iou_loss = F.mse_loss(pred, bbox_iou, reduction='none')
        
        # 가중 합산 (coverage와 iou가 핵심이므로 가중치를 높게)
        alpha = 0.5   # containment penalty weight
        beta = 1.0    # coverage alignment weight  
        gamma = 0.5   # iou consistency weight
        
        total_loss = bce + alpha * containment_penalty + beta * coverage_loss + gamma * iou_loss
        
        return total_loss.mean()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_state = None
    best_val_loss = float("inf")
    history = []

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for xb, yb, mb in train_loader:
            xb = xb.to(run_device)
            yb = yb.to(run_device)
            mb = mb.to(run_device)
            
            logits = model(xb)
            loss = bbox_aware_loss(logits, yb, mb)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * xb.shape[0]
        
        train_loss /= len(train_ds)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_coverage_corr = 0.0
        with torch.no_grad():
            all_preds = []
            all_coverages = []
            for xb, yb, mb in val_loader:
                xb = xb.to(run_device)
                yb = yb.to(run_device)
                mb = mb.to(run_device)
                
                logits = model(xb)
                loss = bbox_aware_loss(logits, yb, mb)
                pred = torch.sigmoid(logits)
                
                val_loss += loss.item() * xb.shape[0]
                val_mae += torch.abs(pred - yb).sum().item()
                
                all_preds.append(pred.cpu())
                all_coverages.append(mb[:, 1:2].cpu())  # gt_coverage
            
            # Coverage correlation (예측이 coverage와 얼마나 상관있는지)
            all_preds = torch.cat(all_preds, dim=0).numpy().flatten()
            all_coverages = torch.cat(all_coverages, dim=0).numpy().flatten()
            if len(all_preds) > 1 and np.std(all_preds) > 1e-6 and np.std(all_coverages) > 1e-6:
                val_coverage_corr = float(np.corrcoef(all_preds, all_coverages)[0, 1])
            else:
                val_coverage_corr = 0.0
        
        val_loss /= len(val_ds)
        val_mae /= len(val_ds)
        
        scheduler.step()
        
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mae": val_mae,
            "val_coverage_corr": val_coverage_corr,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # Save checkpoints
        if checkpoint_dir:
            ckpt_dir = Path(checkpoint_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            
            metadata = {
                "dropout": dropout,
                "input_size": INPUT_SIZE,
                "best_val_loss": best_val_loss,
                "history": history,
            }
            
            save_checkpoint(model, metadata, ckpt_dir / "last_checkpoint.pth")
            if abs(val_loss - best_val_loss) < 1e-12:
                save_checkpoint(model, metadata, ckpt_dir / "best_checkpoint.pth")

        print(f"[Epoch {epoch:03d}] train={train_loss:.4f} val={val_loss:.4f} mae={val_mae:.4f} cov_corr={val_coverage_corr:.4f}")

    if best_state:
        model.load_state_dict(best_state)

    metadata = {
        "dropout": dropout,
        "input_size": INPUT_SIZE,
        "best_val_loss": best_val_loss,
        "history": history,
    }
    return model, metadata


def save_checkpoint(model: TinyMaskScorer, metadata: Dict[str, Any], save_path: Path) -> None:
    """체크포인트 저장."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        **metadata,
    }, save_path)


class AutoMaskSelector:
    """저장된 체크포인트를 로드해 후보 마스크 중 최적을 고르는 추론 클래스."""

    def __init__(self, checkpoint_path: str, device: Optional[str] = None):
        run_device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        ckpt = torch.load(checkpoint_path, map_location=run_device, weights_only=False)
        
        dropout = ckpt.get("dropout", 0.3)
        self.input_size = ckpt.get("input_size", INPUT_SIZE)
        
        self.model = TinyMaskScorer(dropout=dropout).to(run_device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        
        self.device = run_device

    @torch.no_grad()
    def select_best_mask(
        self,
        masks: List[np.ndarray],
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> Tuple[int, np.ndarray]:
        """
        후보 마스크들을 점수화해 최고 점수 인덱스와 전체 점수를 반환.
        
        Args:
            masks: List of (H, W) bool masks
            image: (H, W, 3) BGR 원본 이미지
            bbox: (x1, y1, x2, y2) GT bounding box
        """
        if not masks:
            raise ValueError("masks is empty.")
        
        # 각 마스크에 대해 마스킹된 이미지 생성
        masked_images = []
        for mask in masks:
            masked_img = create_masked_image(image, mask, self.input_size, GRAY_BG_VALUE)
            # BGR → RGB, HWC → CHW, normalize
            img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            masked_images.append(img)
        
        # Batch inference
        batch = torch.stack(masked_images, dim=0).to(self.device)
        logits = self.model(batch)
        scores = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        
        best_idx = int(np.argmax(scores))
        return best_idx, scores


@torch.no_grad()
def select_best_mask_from_ultralytics_result(
    selector: AutoMaskSelector,
    result: Any,
    image: np.ndarray,
    bbox_xyxy: Tuple[int, int, int, int],
) -> Tuple[int, np.ndarray]:
    """Ultralytics SAM 결과에서 최적 마스크 선택."""
    if result is None or result.masks is None or result.masks.data is None:
        raise ValueError("Invalid SAM result: masks are missing.")

    height, width = result.orig_shape
    masks_data = result.masks.data.detach().cpu().numpy()
    masks = [
        cv2.resize(m.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
        for m in masks_data
    ]
    return selector.select_best_mask(masks=masks, image=image, bbox=bbox_xyxy)


def move_result_mask_to_front(result: Any, best_idx: int) -> None:
    """선택된 마스크를 0번 위치로 이동."""
    if best_idx == 0:
        return
    if result is None or result.masks is None or result.masks.data is None:
        return

    data = result.masks.data
    if best_idx < 0 or best_idx >= data.shape[0]:
        return

    order = [best_idx] + [i for i in range(data.shape[0]) if i != best_idx]
    order_tensor = torch.tensor(order, device=data.device)
    result.masks.data = data.index_select(0, order_tensor)


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Train CNN-based SAM mask selector.")
    parser.add_argument("--anno-root", type=str, required=True, help="Annotation root folder")
    parser.add_argument("--sam-weights", type=str, default="weights/sam2.1_t.pt")
    parser.add_argument("--checkpoint-dir", type=str, default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_TRAINSET_CACHE_DIR))
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--max-masks-per-image", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    images, targets, bbox_metrics = get_or_build_training_data(
        annotation_root=args.anno_root,
        sam_model_path=args.sam_weights,
        cache_dir=args.cache_dir,
        max_images=args.max_images,
        max_masks_per_image=args.max_masks_per_image,
        seed=args.seed,
        rebuild_cache=args.rebuild_cache,
    )

    print(f"[Data] {len(images)} samples, target range: [{targets.min():.3f}, {targets.max():.3f}]")
    print(f"[Data] BBox metrics: inside_ratio={bbox_metrics[:, 0].mean():.3f}, "
          f"gt_coverage={bbox_metrics[:, 1].mean():.3f}, "
          f"bbox_iou={bbox_metrics[:, 2].mean():.3f}")

    model, metadata = train_mask_selector(
        images=images,
        targets=targets,
        bbox_metrics=bbox_metrics,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
    )

    print(f"\nTraining complete!")
    print(f"  Best val loss: {metadata['best_val_loss']:.4f}")
    print(f"  Checkpoint dir: {args.checkpoint_dir}")


if __name__ == "__main__":
    main()