"""
SAM Masking Quality Evaluator

inference 디렉토리의 txt 파일(mask RLE + bbox)과 원본 이미지 디렉토리의 GT bbox를
비교하여 마스킹 품질을 평가합니다.

평가 지표:
1. Inside Ratio: mask 픽셀 중 GT bbox 안에 있는 비율 (containment)
2. GT Coverage: GT bbox 중 mask가 채우는 비율 (coverage)
3. BBox IoU: mask bbox와 GT bbox의 IoU
4. Size Ratio: mask 면적 / GT bbox 면적 (적절한 크기인지)
5. Success Rate: 성공적으로 마스킹된 비율 (실패 반영)
6. Quality Score: 종합 품질 점수 (inside * coverage * size_penalty)
"""

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from dataset.datasets import load_aihubbbox_label


@dataclass
class MaskMetrics:
    """단일 마스크에 대한 품질 지표"""
    inside_ratio: float = 0.0      # mask가 GT bbox 안에 있는 비율
    gt_coverage: float = 0.0       # GT bbox를 mask가 채우는 비율
    bbox_iou: float = 0.0          # mask bbox와 GT bbox의 IoU
    size_ratio: float = 0.0        # mask 면적 / GT bbox 면적
    quality_score: float = 0.0     # 종합 품질 점수
    is_valid: bool = True          # 유효한 마스크인지


@dataclass
class ClassMetrics:
    """클래스별 집계 지표"""
    class_name: str
    total_count: int = 0
    success_count: int = 0
    fail_count: int = 0
    
    inside_ratio_sum: float = 0.0
    gt_coverage_sum: float = 0.0
    bbox_iou_sum: float = 0.0
    size_ratio_sum: float = 0.0
    quality_score_sum: float = 0.0
    
    @property
    def success_rate(self) -> float:
        return self.success_count / max(self.total_count, 1)
    
    @property
    def avg_inside_ratio(self) -> float:
        return self.inside_ratio_sum / max(self.success_count, 1)
    
    @property
    def avg_gt_coverage(self) -> float:
        return self.gt_coverage_sum / max(self.success_count, 1)
    
    @property
    def avg_bbox_iou(self) -> float:
        return self.bbox_iou_sum / max(self.success_count, 1)
    
    @property
    def avg_size_ratio(self) -> float:
        return self.size_ratio_sum / max(self.success_count, 1)
    
    @property
    def avg_quality_score(self) -> float:
        return self.quality_score_sum / max(self.success_count, 1)


@dataclass
class SplitMetrics:
    """train/val/test split별 집계"""
    split_name: str
    class_metrics: Dict[str, ClassMetrics] = field(default_factory=dict)
    
    def get_or_create_class(self, class_name: str) -> ClassMetrics:
        if class_name not in self.class_metrics:
            self.class_metrics[class_name] = ClassMetrics(class_name=class_name)
        return self.class_metrics[class_name]
    
    @property
    def total_count(self) -> int:
        return sum(c.total_count for c in self.class_metrics.values())
    
    @property
    def success_count(self) -> int:
        return sum(c.success_count for c in self.class_metrics.values())
    
    @property
    def success_rate(self) -> float:
        return self.success_count / max(self.total_count, 1)
    
    @property
    def avg_inside_ratio(self) -> float:
        total = sum(c.inside_ratio_sum for c in self.class_metrics.values())
        count = sum(c.success_count for c in self.class_metrics.values())
        return total / max(count, 1)
    
    @property
    def avg_gt_coverage(self) -> float:
        total = sum(c.gt_coverage_sum for c in self.class_metrics.values())
        count = sum(c.success_count for c in self.class_metrics.values())
        return total / max(count, 1)
    
    @property
    def avg_bbox_iou(self) -> float:
        total = sum(c.bbox_iou_sum for c in self.class_metrics.values())
        count = sum(c.success_count for c in self.class_metrics.values())
        return total / max(count, 1)
    
    @property
    def avg_size_ratio(self) -> float:
        total = sum(c.size_ratio_sum for c in self.class_metrics.values())
        count = sum(c.success_count for c in self.class_metrics.values())
        return total / max(count, 1)
    
    @property
    def avg_quality_score(self) -> float:
        total = sum(c.quality_score_sum for c in self.class_metrics.values())
        count = sum(c.success_count for c in self.class_metrics.values())
        return total / max(count, 1)


def decode_mask_rle(rle_str: str, height: int, width: int) -> np.ndarray:
    """RLE 문자열을 binary mask로 디코딩"""
    if not rle_str or rle_str == "0;":
        return np.zeros((height, width), dtype=bool)
    
    parts = rle_str.split(";")
    if len(parts) != 2:
        return np.zeros((height, width), dtype=bool)
    
    start_value = int(parts[0])
    if not parts[1]:
        return np.zeros((height, width), dtype=bool)
    
    runs = [int(x) for x in parts[1].split(",")]
    
    flat = []
    current = start_value
    for count in runs:
        flat.extend([current] * count)
        current = 1 - current
    
    flat = np.array(flat, dtype=np.uint8)
    expected_size = height * width
    
    if flat.size < expected_size:
        flat = np.pad(flat, (0, expected_size - flat.size), constant_values=0)
    elif flat.size > expected_size:
        flat = flat[:expected_size]
    
    return flat.reshape((height, width)).astype(bool)


def parse_region_txt(txt_path: Path) -> Optional[Dict]:
    """inference에서 생성된 txt 파일 파싱"""
    if not txt_path.exists():
        return None
    
    try:
        content = txt_path.read_text(encoding="utf-8")
    except Exception:
        return None
    
    result = {}
    for line in content.strip().split("\n"):
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        
        if key in ("mask_height", "mask_width", "x1", "y1", "x2", "y2", "area_pixels"):
            result[key] = int(value)
        elif key == "area_ratio":
            result[key] = float(value)
        elif key == "rle":
            result[key] = value
        else:
            result[key] = value
    
    return result if result else None


def compute_bbox_iou(
    box_a: Tuple[int, int, int, int],
    box_b: Tuple[int, int, int, int],
) -> float:
    """두 bbox (x1,y1,x2,y2)의 IoU 계산"""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    
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
    
    if union <= 0:
        return 0.0
    return inter / union


def compute_mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """binary mask의 외접 bbox (x1,y1,x2,y2) 계산"""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def evaluate_single_mask(
    mask: np.ndarray,
    gt_bbox: Tuple[int, int, int, int],
) -> MaskMetrics:
    """단일 마스크와 GT bbox를 비교해 품질 지표 계산"""
    gx1, gy1, gx2, gy2 = gt_bbox
    gt_area = float(max(0, gx2 - gx1) * max(0, gy2 - gy1))
    mask_area = float(mask.sum())
    
    if mask_area == 0 or gt_area == 0:
        return MaskMetrics(is_valid=False)
    
    # mask bbox
    mask_bbox = compute_mask_bbox(mask)
    if mask_bbox is None:
        return MaskMetrics(is_valid=False)
    
    # mask가 GT bbox 안에 있는 픽셀 수
    mask_in_gt = float(mask[gy1:gy2, gx1:gx2].sum())
    
    # 1. Inside Ratio: containment
    inside_ratio = mask_in_gt / mask_area
    
    # 2. GT Coverage
    gt_coverage = mask_in_gt / gt_area
    
    # 3. BBox IoU
    bbox_iou = compute_bbox_iou(mask_bbox, gt_bbox)
    
    # 4. Size Ratio
    size_ratio = mask_area / gt_area
    
    # 5. Quality Score: inside * coverage * size_penalty
    size_penalty = min(1.0, size_ratio / 0.3)  # size_ratio < 0.3이면 패널티
    quality_score = inside_ratio * gt_coverage * size_penalty
    
    return MaskMetrics(
        inside_ratio=inside_ratio,
        gt_coverage=gt_coverage,
        bbox_iou=bbox_iou,
        size_ratio=size_ratio,
        quality_score=quality_score,
        is_valid=True,
    )


def find_gt_label_for_inference(
    inference_txt_path: Path,
    inference_root: Path,
    original_root: Path,
) -> Optional[Path]:
    """inference txt 파일에 대응하는 원본 GT JSON 경로 찾기"""
    rel_path = inference_txt_path.relative_to(inference_root)
    # txt -> json
    json_rel = rel_path.with_suffix(".json")
    gt_json_path = original_root / json_rel
    
    if gt_json_path.exists():
        return gt_json_path
    return None


def iter_inference_txt_files(inference_root: Path):
    """inference 디렉토리의 모든 txt 파일 순회"""
    for p in inference_root.rglob("*.txt"):
        # issue report 파일 제외
        if p.name == "sam_inference_issues.txt":
            continue
        yield p


def get_split_and_class(rel_path: Path) -> Tuple[str, str]:
    """상대 경로에서 split(train/val/test)과 class 이름 추출
    
    예: train/cucumber_downy/xxx.txt -> ("train", "cucumber_downy")
    """
    parts = rel_path.parts
    if len(parts) >= 2:
        return parts[0], parts[1]
    elif len(parts) == 1:
        return "root", parts[0]
    return "root", "unknown"


def iter_original_json_files(original_root: Path):
    """original 디렉토리의 모든 JSON 라벨 파일 순회"""
    for p in original_root.rglob("*.json"):
        yield p


def evaluate_directory(
    inference_root: Path,
    original_root: Path,
) -> Dict[str, SplitMetrics]:
    """전체 디렉토리 평가 - original 디렉토리 기준으로 total 카운트"""
    splits: Dict[str, SplitMetrics] = {}
    
    # original 디렉토리의 모든 JSON 라벨을 기준으로 평가
    for json_path in iter_original_json_files(original_root):
        rel_path = json_path.relative_to(original_root)
        split_name, class_name = get_split_and_class(rel_path)
        
        # split 초기화
        if split_name not in splits:
            splits[split_name] = SplitMetrics(split_name=split_name)
        
        split_metrics = splits[split_name]
        class_metrics = split_metrics.get_or_create_class(class_name)
        class_metrics.total_count += 1
        
        # GT bbox 로드
        gt_data = load_aihubbbox_label(json_path)
        if gt_data is None:
            class_metrics.fail_count += 1
            continue
        
        gt_bbox = tuple(gt_data["bbox_xyxy"])
        
        # 대응하는 inference txt 파일 찾기
        txt_rel_path = rel_path.with_suffix(".txt")
        txt_path = inference_root / txt_rel_path
        
        if not txt_path.exists():
            class_metrics.fail_count += 1
            continue
        
        # inference txt 파싱
        region_data = parse_region_txt(txt_path)
        if region_data is None or "rle" not in region_data:
            class_metrics.fail_count += 1
            continue
        
        # mask 디코딩
        height = region_data.get("mask_height", 0)
        width = region_data.get("mask_width", 0)
        if height == 0 or width == 0:
            class_metrics.fail_count += 1
            continue
        
        mask = decode_mask_rle(region_data["rle"], height, width)
        
        # 품질 평가
        metrics = evaluate_single_mask(mask, gt_bbox)
        if not metrics.is_valid:
            class_metrics.fail_count += 1
            continue
        
        # 성공 집계
        class_metrics.success_count += 1
        class_metrics.inside_ratio_sum += metrics.inside_ratio
        class_metrics.gt_coverage_sum += metrics.gt_coverage
        class_metrics.bbox_iou_sum += metrics.bbox_iou
        class_metrics.size_ratio_sum += metrics.size_ratio
        class_metrics.quality_score_sum += metrics.quality_score
    
    return splits


def print_report(splits: Dict[str, SplitMetrics], output_path: Optional[Path] = None) -> None:
    """평가 결과 출력 및 파일 저장"""
    lines = []
    
    lines.append("=" * 100)
    lines.append("SAM Masking Quality Evaluation Report")
    lines.append("=" * 100)
    lines.append("")
    lines.append("Metrics Description:")
    lines.append("  - Inside Ratio: Portion of mask pixels inside GT bbox (containment, higher=better)")
    lines.append("  - GT Coverage: Portion of GT bbox covered by mask (coverage, higher=better)")
    lines.append("  - BBox IoU: IoU between mask bbox and GT bbox (higher=better)")
    lines.append("  - Size Ratio: mask_area / gt_bbox_area (closer to 1.0 = better)")
    lines.append("  - Quality Score: inside * coverage * size_penalty (comprehensive score)")
    lines.append("  - Success Rate: Successfully segmented ratio")
    lines.append("")
    
    for split_name in sorted(splits.keys()):
        split = splits[split_name]
        
        lines.append("=" * 100)
        lines.append(f"[{split_name.upper()}] Summary")
        lines.append("=" * 100)
        lines.append(f"  Total: {split.total_count}, Success: {split.success_count}, "
                    f"Success Rate: {split.success_rate:.2%}")
        lines.append(f"  Avg Inside Ratio: {split.avg_inside_ratio:.4f}")
        lines.append(f"  Avg GT Coverage:  {split.avg_gt_coverage:.4f}")
        lines.append(f"  Avg BBox IoU:     {split.avg_bbox_iou:.4f}")
        lines.append(f"  Avg Size Ratio:   {split.avg_size_ratio:.4f}")
        lines.append(f"  Avg Quality Score: {split.avg_quality_score:.4f}")
        lines.append("")
        
        # 클래스별 상세
        lines.append(f"[{split_name.upper()}] Per-Class Details")
        lines.append("-" * 100)
        header = f"{'Class':<30} {'Total':>6} {'Succ':>6} {'Rate':>8} " \
                 f"{'Inside':>8} {'Cover':>8} {'IoU':>8} {'Size':>8} {'Quality':>8}"
        lines.append(header)
        lines.append("-" * 100)
        
        for class_name in sorted(split.class_metrics.keys()):
            cm = split.class_metrics[class_name]
            row = f"{class_name:<30} {cm.total_count:>6} {cm.success_count:>6} " \
                  f"{cm.success_rate:>7.2%} " \
                  f"{cm.avg_inside_ratio:>8.4f} {cm.avg_gt_coverage:>8.4f} " \
                  f"{cm.avg_bbox_iou:>8.4f} {cm.avg_size_ratio:>8.4f} " \
                  f"{cm.avg_quality_score:>8.4f}"
            lines.append(row)
        
        lines.append("")
    
    # 전체 요약
    lines.append("=" * 100)
    lines.append("[OVERALL] All Splits Combined")
    lines.append("=" * 100)
    
    total_count = sum(s.total_count for s in splits.values())
    success_count = sum(s.success_count for s in splits.values())
    inside_sum = sum(sum(c.inside_ratio_sum for c in s.class_metrics.values()) for s in splits.values())
    coverage_sum = sum(sum(c.gt_coverage_sum for c in s.class_metrics.values()) for s in splits.values())
    iou_sum = sum(sum(c.bbox_iou_sum for c in s.class_metrics.values()) for s in splits.values())
    size_sum = sum(sum(c.size_ratio_sum for c in s.class_metrics.values()) for s in splits.values())
    quality_sum = sum(sum(c.quality_score_sum for c in s.class_metrics.values()) for s in splits.values())
    
    lines.append(f"  Total: {total_count}, Success: {success_count}, "
                f"Success Rate: {success_count/max(total_count,1):.2%}")
    lines.append(f"  Avg Inside Ratio: {inside_sum/max(success_count,1):.4f}")
    lines.append(f"  Avg GT Coverage:  {coverage_sum/max(success_count,1):.4f}")
    lines.append(f"  Avg BBox IoU:     {iou_sum/max(success_count,1):.4f}")
    lines.append(f"  Avg Size Ratio:   {size_sum/max(success_count,1):.4f}")
    lines.append(f"  Avg Quality Score: {quality_sum/max(success_count,1):.4f}")
    lines.append("")
    
    report_text = "\n".join(lines)
    print(report_text)
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report_text, encoding="utf-8")
        print(f"\nReport saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate SAM masking quality by comparing inference results with GT bounding boxes."
    )
    parser.add_argument(
        "--inference-dir",
        type=str,
        required=True,
        help="Directory containing SAMInference output (masked images + txt files)",
    )
    parser.add_argument(
        "--original-dir",
        type=str,
        required=True,
        help="Original image directory containing GT JSON labels",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save the evaluation report",
    )
    args = parser.parse_args()
    
    inference_root = Path(args.inference_dir)
    original_root = Path(args.original_dir)
    
    if not inference_root.exists():
        raise FileNotFoundError(f"Inference directory not found: {inference_root}")
    if not original_root.exists():
        raise FileNotFoundError(f"Original directory not found: {original_root}")
    
    print(f"Evaluating...")
    print(f"  Inference dir: {inference_root}")
    print(f"  Original dir:  {original_root}")
    print("")
    
    splits = evaluate_directory(inference_root, original_root)
    
    output_path = None
    if args.output:
        output_path = Path(args.output)
        # 디렉토리면 기본 파일명 추가
        if output_path.is_dir() or (not output_path.suffix and output_path.exists()):
            output_path = output_path / "sam_quality_report.txt"
    
    print_report(splits, output_path)


if __name__ == "__main__":
    main()
