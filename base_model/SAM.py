import os
import json
import argparse
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

try:
    from .SAMSelector import (
        AutoMaskSelector,
        move_result_mask_to_front,
        select_best_mask_from_ultralytics_result,
    )
except ImportError:
    from SAMSelector import (
        AutoMaskSelector,
        move_result_mask_to_front,
        select_best_mask_from_ultralytics_result,
    )

class SAM2MaskGenerator:
    """
    Wraps Ultralytics SAM2.1 to produce *all* candidate masks per image
    using the 'segment everything' mode (no prompts → automatic).
    """

    def __init__(
        self,
        model_path: str = 'sam2.1_t.pt',
        mode: str = "center_point",
        selector_checkpoint_path: str = "weights/samselector/best_checkpoint.pth",
        selector_device: Optional[str] = None,
    ):
        from ultralytics import SAM

        self.model = SAM(model_path)
        self.mode = mode
        self.selector: Optional[AutoMaskSelector] = None

        if self.mode == "autoselector":
            self.selector = AutoMaskSelector(
                checkpoint_path=selector_checkpoint_path,
                device=selector_device,
            )

    @staticmethod
    def _mask_bbox(mask_bool: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        ys, xs = np.where(mask_bool)
        if len(xs) == 0:
            return None
        x1 = int(xs.min())
        y1 = int(ys.min())
        x2 = int(xs.max()) + 1
        y2 = int(ys.max()) + 1
        return x1, y1, x2, y2

    @torch.no_grad()
    def generate_with_region(self, image_path: str, bbox=None) -> Tuple[np.ndarray, Dict[str, float], np.ndarray]:
        """
        Run SAM2.1 and return masked RGB image + selected mask region info + binary mask.

        region_info keys:
            - x1, y1, x2, y2 (selected mask bbox)
            - area_pixels
            - area_ratio
        """
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        h, w = img.shape[:2]

        if self.mode == "center_point":
            center_x, center_y = w // 2, h // 2
            results = self.model(image_path, points=[[center_x, center_y]], labels=[1], verbose=False)
        elif self.mode == "box":
            if bbox is None:
                raise ValueError("bbox must be provided when mode='box'.")
            results = self.model(image_path, bboxes=[bbox], verbose=False)
        elif self.mode == "autoselector":
            if bbox is None:
                raise ValueError("bbox must be provided when mode='autoselector' for mask ranking.")
            if self.selector is None:
                raise RuntimeError("Auto selector is not initialized.")

            results = self.model(image_path, verbose=False)
            best_idx, _ = select_best_mask_from_ultralytics_result(
                selector=self.selector,
                result=results[0],
                bbox_xyxy=tuple(bbox),
            )
            move_result_mask_to_front(results[0], best_idx)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        if not results or results[0].masks is None or results[0].masks.data is None:
            raise RuntimeError("SAM did not return valid masks.")

        mask = results[0].masks.data[0].detach().cpu().numpy().astype(np.uint8)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_bool = mask > 0

        bg_color = (128, 128, 128)
        final_img = np.full_like(img, bg_color)
        final_img[mask_bool] = img[mask_bool]

        mb = self._mask_bbox(mask_bool)
        area_pixels = float(mask_bool.sum())
        area_ratio = area_pixels / float(max(h * w, 1))

        if mb is None:
            region_info: Dict[str, float] = {
                "x1": -1.0,
                "y1": -1.0,
                "x2": -1.0,
                "y2": -1.0,
                "area_pixels": area_pixels,
                "area_ratio": area_ratio,
            }
        else:
            x1, y1, x2, y2 = mb
            region_info = {
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "area_pixels": area_pixels,
                "area_ratio": area_ratio,
            }

        return final_img, region_info, mask_bool

    @torch.no_grad()
    def generate(self, image_path: str, bbox=None) -> np.ndarray:
        """Backward-compatible API: return only masked image."""
        final_img, _, _ = self.generate_with_region(image_path=image_path, bbox=bbox)
        return final_img