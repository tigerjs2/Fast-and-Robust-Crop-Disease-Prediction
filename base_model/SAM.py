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
# from SAMSelector import MaskSelector

class SAM2MaskGenerator:
    """
    Wraps Ultralytics SAM2.1 to produce *all* candidate masks per image
    using the 'segment everything' mode (no prompts → automatic).
    """

    def __init__(self, model_path='sam2.1_t.pt', mode="center_point"):
        #
        from ultralytics import SAM

        self.model = SAM(model_path)

        @torch.no_grad()
        def generate(self, image_path: str, bbox = None) -> List[np.ndarray]:
            """
            Run SAM2.1 in 'segment everything' mode.
            Returns: list of binary masks, each shape (H, W), dtype bool.
            """
            img = cv2.imread(image_path)

            if self.mode == "center_point": # 중앙 픽셀을 포함한 마스크 선택
                h, w = img.shape[:2]
                center_x, center_y = w // 2, h // 2 
                results = self.model(image_path, points=[[center_x, center_y]], labels=[1], verbose=False)
            elif self.mode == "box": # 바운딩 박스 + 10pix 패딩 영역을 포함한 마스크
                results = self.model(image_path, bboxes=[bbox], verbose=False)
            elif self.mode == "autoselector": # MLP Network 기반 마스크 셀렉터
                results = self.model(image_path, verbose=False)
                # TODO: MLP Network 기반 마스크 셀렉터로 results에서 최적 마스크 선택해서 results의 0번으로 위치시키기


            mask = results[0].masks.data[0].cpu().numpy()

            # 추출된 마스크 보정
            
            mask = mask.astype(np.uint8)
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

            bool_mask = mask > 0
            # 마스크 밖의 영역 회색으로 치환
            bg_color = (128, 128, 128)
            final_img = np.full_like(img, bg_color)
            final_img[bool_mask] = img[bool_mask]

            return final_img