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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from dataset.datasets import AIHubBBoxDataset


DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "weights" / "samselector"


class MaskScoringMLP(nn.Module):
    """
    마스크 품질 feature(7차원)를 입력으로 받아 점수(logit)를 예측하는 MLP.

    A lightweight MLP that maps a hand-crafted feature vector
    to a scalar suitability score (logit → sigmoid → [0, 1]).

    Architecture:
        Linear → BN → ReLU → Dropout → Linear → BN → ReLU → Dropout → Linear(1)

    The small size is intentional: the feature vector is only 7-D,
    so a deeper/wider net would overfit.
    """

    def __init__(
        self,
        feat_dim: int = 7,
        hidden_dims: Sequence[int] = (64, 32),
        dropout: float = 0.2,
    ):
        super().__init__()
        layers = []
        in_dim = feat_dim
        for h in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, h),
                    nn.BatchNorm1d(h),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, feat_dim) → (B, 1) logit"""
        return self.net(x)


class FeatureDataset(Dataset):
    """학습용 feature/target 배열을 PyTorch Dataset 형태로 감싸는 클래스."""

    def __init__(self, features: np.ndarray, targets: np.ndarray):
        """numpy 배열을 float32 tensor로 변환해 저장한다."""
        self.features = torch.from_numpy(features.astype(np.float32))
        self.targets = torch.from_numpy(targets.astype(np.float32)).unsqueeze(1)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]


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


def _bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """두 축정렬 박스(x1,y1,x2,y2)의 IoU를 계산한다."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = float(iw * ih)

    a_area = float(max(0, ax2 - ax1) * max(0, ay2 - ay1))
    b_area = float(max(0, bx2 - bx1) * max(0, by2 - by1))
    union = a_area + b_area - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """이진 마스크의 외접 bbox(x1,y1,x2,y2)를 계산한다."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    return x1, y1, x2, y2


def extract_mask_features(mask: np.ndarray, image_shape: Tuple[int, int], gt_bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """
    후보 마스크 1개에서 7차원 수치 feature를 추출한다.

    구성: 면적 비율, 박스 면적 비율, 채움 비율, bbox IoU,
    GT 영역 커버리지, 중심점 포함 여부, 중심 근접도*형상 compactness.
    """
    height, width = image_shape
    total_pixels = float(height * width)

    area = float(mask.sum())
    area_ratio = area / max(total_pixels, 1.0)

    mb = _mask_bbox(mask)
    if mb is None:
        return np.zeros(7, dtype=np.float32)

    mx1, my1, mx2, my2 = mb
    box_area = float(max(0, mx2 - mx1) * max(0, my2 - my1))
    box_area_ratio = box_area / max(total_pixels, 1.0)
    fill_ratio = area / max(box_area, 1.0)

    bbox_iou = _bbox_iou(mb, gt_bbox)

    gx1, gy1, gx2, gy2 = gt_bbox
    gt_area = float(max(0, gx2 - gx1) * max(0, gy2 - gy1))
    mask_in_gt = float(mask[gy1:gy2, gx1:gx2].sum()) if gt_area > 0 else 0.0
    gt_coverage = mask_in_gt / max(gt_area, 1.0)

    gcx = int((gx1 + gx2) / 2)
    gcy = int((gy1 + gy2) / 2)
    gcx = max(0, min(width - 1, gcx))
    gcy = max(0, min(height - 1, gcy))
    center_hit = 1.0 if mask[gcy, gcx] else 0.0

    ys, xs = np.where(mask)
    cx = float(xs.mean())
    cy = float(ys.mean())
    diag = math.sqrt(float(width * width + height * height))
    centroid_dist = math.sqrt((cx - gcx) ** 2 + (cy - gcy) ** 2) / max(diag, 1.0)
    centroid_closeness = 1.0 - min(1.0, centroid_dist)

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = float(sum(cv2.arcLength(c, True) for c in contours))
    compactness = (4.0 * math.pi * area) / max(perimeter * perimeter, 1.0)

    return np.array(
        [
            area_ratio,
            box_area_ratio,
            fill_ratio,
            bbox_iou,
            gt_coverage,
            center_hit,
            centroid_closeness * compactness,
        ],
        dtype=np.float32,
    )


def build_training_data(
    annotation_root: str,
    sam_model_path: str,
    max_images: Optional[int] = None,
    max_masks_per_image: Optional[int] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    AIHubBBoxDataset을 이용해 (features, targets) 학습 샘플을 구축한다.

    target은 GT bbox 내부 커버리지(0~1) soft label을 사용한다.
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
    features: List[np.ndarray] = []
    targets: List[float] = []

    for sample in tqdm(samples, desc="Build selector trainset"):
        image_path = sample["image_path"]
        gt_bbox = tuple(sample["bbox_xyxy"])
        height = int(sample["height"])
        width = int(sample["width"])

        masks = generator.generate_masks(str(image_path))
        if not masks:
            continue

        if max_masks_per_image is not None and max_masks_per_image > 0 and len(masks) > max_masks_per_image:
            idx = np.random.choice(len(masks), size=max_masks_per_image, replace=False)
            masks = [masks[i] for i in idx]

        for m in masks:
            feat = extract_mask_features(m, (height, width), gt_bbox)

            gx1, gy1, gx2, gy2 = gt_bbox
            gt_area = float(max(0, gx2 - gx1) * max(0, gy2 - gy1))
            mask_in_gt = float(m[gy1:gy2, gx1:gx2].sum()) if gt_area > 0 else 0.0
            gt_coverage = mask_in_gt / max(gt_area, 1.0)
            target = float(np.clip(gt_coverage, 0.0, 1.0))

            features.append(feat)
            targets.append(target)

    if not features:
        raise RuntimeError("No training samples were built. Check annotation path and SAM output.")

    return np.vstack(features).astype(np.float32), np.array(targets, dtype=np.float32)


def train_mask_selector(
    features: np.ndarray,
    targets: np.ndarray,
    hidden_dims: Sequence[int] = (64, 32),
    dropout: float = 0.2,
    batch_size: int = 256,
    epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    val_ratio: float = 0.2,
    seed: int = 42,
    device: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
) -> Tuple[MaskScoringMLP, Dict[str, Any]]:
    """
    마스크 스코어링 MLP를 학습하고, 정규화 통계/학습 이력을 metadata로 반환한다.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    n = features.shape[0]
    idx = np.arange(n)
    np.random.shuffle(idx)
    features = features[idx]
    targets = targets[idx]

    val_size = max(1, int(n * val_ratio))
    train_size = max(1, n - val_size)
    if train_size + val_size > n:
        val_size = n - train_size

    x_train = features[:train_size]
    y_train = targets[:train_size]
    x_val = features[train_size : train_size + val_size]
    y_val = targets[train_size : train_size + val_size]

    feat_mean = x_train.mean(axis=0, keepdims=True)
    feat_std = x_train.std(axis=0, keepdims=True) + 1e-6

    x_train = (x_train - feat_mean) / feat_std
    x_val = (x_val - feat_mean) / feat_std

    train_ds = FeatureDataset(x_train, y_train)
    val_ds = FeatureDataset(x_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    run_device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

    model = MaskScoringMLP(feat_dim=features.shape[1], hidden_dims=hidden_dims, dropout=dropout).to(run_device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_val_loss = float("inf")
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(run_device)
            yb = yb.to(run_device)

            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.shape[0]

        train_loss /= max(len(train_ds), 1)

        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(run_device)
                yb = yb.to(run_device)

                logits = model(xb)
                loss = criterion(logits, yb)
                pred = torch.sigmoid(logits)

                val_loss += loss.item() * xb.shape[0]
                val_mae += torch.abs(pred - yb).sum().item()

        val_loss /= max(len(val_ds), 1)
        val_mae /= max(len(val_ds), 1)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_mae": val_mae})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        epoch_metadata: Dict[str, Any] = {
            "feat_mean": feat_mean.squeeze(0).astype(np.float32),
            "feat_std": feat_std.squeeze(0).astype(np.float32),
            "hidden_dims": list(hidden_dims),
            "dropout": float(dropout),
            "best_val_loss": float(best_val_loss),
            "history": history,
        }

        if checkpoint_dir is not None:
            ckpt_dir = Path(checkpoint_dir)
            save_epoch_checkpoint(
                model=model,
                metadata=epoch_metadata,
                save_path=ckpt_dir / "last_checkpoint.pth",
                epoch=epoch,
                optimizer=optimizer,
            )
            if abs(val_loss - best_val_loss) < 1e-12:
                save_epoch_checkpoint(
                    model=model,
                    metadata=epoch_metadata,
                    save_path=ckpt_dir / "best_checkpoint.pth",
                    epoch=epoch,
                    optimizer=optimizer,
                )

        print(
            f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_mae={val_mae:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    metadata: Dict[str, Any] = {
        "feat_mean": feat_mean.squeeze(0).astype(np.float32),
        "feat_std": feat_std.squeeze(0).astype(np.float32),
        "hidden_dims": list(hidden_dims),
        "dropout": float(dropout),
        "best_val_loss": float(best_val_loss),
        "history": history,
    }
    return model, metadata


def save_selector_checkpoint(model: MaskScoringMLP, metadata: Dict[str, Any], save_path: str) -> None:
    """학습된 모델 가중치와 정규화 통계를 체크포인트 파일로 저장한다."""
    out = {
        "model_state_dict": model.state_dict(),
        "feat_mean": metadata["feat_mean"],
        "feat_std": metadata["feat_std"],
        "hidden_dims": metadata["hidden_dims"],
        "dropout": metadata["dropout"],
        "best_val_loss": metadata["best_val_loss"],
        "history": metadata["history"],
    }
    save_file = Path(save_path)
    save_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, save_file)


def save_epoch_checkpoint(
    model: MaskScoringMLP,
    metadata: Dict[str, Any],
    save_path: Path,
    epoch: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> None:
    """에포크 단위 체크포인트를 저장한다(last/best 공용)."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Any] = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "feat_mean": metadata["feat_mean"],
        "feat_std": metadata["feat_std"],
        "hidden_dims": metadata["hidden_dims"],
        "dropout": metadata["dropout"],
        "best_val_loss": metadata["best_val_loss"],
        "history": metadata["history"],
    }
    if optimizer is not None:
        out["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(out, save_path)


class AutoMaskSelector:
    """저장된 체크포인트를 로드해 후보 마스크 중 최적 마스크를 고르는 추론 클래스."""

    def __init__(self, checkpoint_path: str, device: Optional[str] = None):
        """모델과 feature 정규화 통계(mean/std)를 복원한다."""
        run_device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(checkpoint_path, map_location=run_device)

        hidden_dims = ckpt.get("hidden_dims", [64, 32])
        dropout = ckpt.get("dropout", 0.2)
        feat_mean = np.asarray(ckpt["feat_mean"], dtype=np.float32)
        feat_std = np.asarray(ckpt["feat_std"], dtype=np.float32)

        self.model = MaskScoringMLP(feat_dim=int(feat_mean.shape[0]), hidden_dims=hidden_dims, dropout=dropout).to(run_device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.feat_mean = feat_mean
        self.feat_std = feat_std
        self.device = run_device

    @torch.no_grad()
    def select_best_mask(
        self,
        masks: List[np.ndarray],
        image_shape: Tuple[int, int],
        bbox: Tuple[int, int, int, int],
    ) -> Tuple[int, np.ndarray]:
        """후보 마스크들을 점수화해 최고 점수 인덱스와 전체 점수를 반환한다."""
        if not masks:
            raise ValueError("masks is empty.")

        feats = np.vstack([extract_mask_features(m, image_shape, bbox) for m in masks]).astype(np.float32)
        feats = (feats - self.feat_mean[None, :]) / (self.feat_std[None, :] + 1e-6)

        xb = torch.from_numpy(feats).to(self.device)
        logits = self.model(xb)
        scores = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        best_idx = int(np.argmax(scores))
        return best_idx, scores


@torch.no_grad()
def select_best_mask_from_ultralytics_result(
    selector: AutoMaskSelector,
    result: Any,
    bbox_xyxy: Tuple[int, int, int, int],
) -> Tuple[int, np.ndarray]:
    """Ultralytics SAM 단일 result 객체에서 최적 마스크 인덱스를 선택한다."""
    if result is None or result.masks is None or result.masks.data is None:
        raise ValueError("Invalid SAM result: masks are missing.")

    height, width = result.orig_shape
    masks_data = result.masks.data.detach().cpu().numpy()
    masks = [
        cv2.resize(m.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
        for m in masks_data
    ]
    return selector.select_best_mask(masks=masks, image_shape=(height, width), bbox=bbox_xyxy)


def move_result_mask_to_front(result: Any, best_idx: int) -> None:
    """선택된 마스크를 0번 위치로 옮겨 후속 코드와 호환되게 재정렬한다."""
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
    """CLI 인자를 받아 데이터 구축, 학습, 체크포인트 저장까지 실행한다."""
    parser = argparse.ArgumentParser(description="Train and save SAM auto-selector MLP.")
    parser.add_argument("--anno-root", type=str, required=True, help="Root folder containing annotation json files.")
    parser.add_argument("--sam-weights", type=str, default="weights/sam2.1_t.pt", help="Ultralytics SAM weight path.")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(DEFAULT_CHECKPOINT_DIR),
        help="Directory to save last_checkpoint.pth and best_checkpoint.pth",
    )
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap on number of images for quick experiments.")
    parser.add_argument("--max-masks-per-image", type=int, default=10, help="Optional cap on masks sampled per image.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[64, 32])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="cpu or cuda")
    args = parser.parse_args()

    feats, targets = build_training_data(
        annotation_root=args.anno_root,
        sam_model_path=args.sam_weights,
        max_images=args.max_images,
        max_masks_per_image=args.max_masks_per_image,
        seed=args.seed,
    )

    model, metadata = train_mask_selector(
        features=feats,
        targets=targets,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
    )

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # 학습 완료 시점의 모델도 last_checkpoint.pth로 한 번 더 보장 저장
    save_epoch_checkpoint(
        model=model,
        metadata=metadata,
        save_path=ckpt_dir / "last_checkpoint.pth",
        epoch=args.epochs,
        optimizer=None,
    )

    print(f"Last checkpoint: {ckpt_dir / 'last_checkpoint.pth'}")
    print(f"Best checkpoint: {ckpt_dir / 'best_checkpoint.pth'}")


if __name__ == "__main__":
    main()