import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import SAM

from torch.export import export
from executorch.exir import to_edge
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

class SAM2BoxDecoder512(nn.Module):
    """
    256x256 백본 출력(16x16)과 프롬프트 인코더의 정적 출력(64x64) 간의 
    해상도 충돌을 동적 보간(Interpolation)으로 해결한 래퍼 클래스입니다.
    """
    def __init__(self, model):
        super().__init__()
        self.prompt_encoder = model.sam_prompt_encoder
        self.mask_decoder = model.sam_mask_decoder
        self.conv_s0 = model.sam_mask_decoder.conv_s0
        self.conv_s1 = model.sam_mask_decoder.conv_s1

    def forward(self, image_embed, feat_s0_256c, feat_s1_256c, box_coords):
        # 1. 프롬프트 인코딩 (참고: 원본 Prompt Encoder가 1024 기준이므로 
        #    입력된 256 기준 좌표를 내부적으로 1024 스케일로 매핑하여 정확도를 유지합니다)
        scaled_box_coords = box_coords * 4.0
        # scaled_box_coords = box_coords * 2.0  # (orig 512 -> 1024)
        
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=scaled_box_coords,
            masks=None
        )

        # 2. 위치 인코딩 추출 (기본 64x64 출력)
        image_pe = self.prompt_encoder.get_dense_pe()

        # --------------------------------------------------------------------
        # [핵심 패치] image_embed 크기(16x16)에 맞춰 PE 및 Dense 텐서 강제 축소
        # --------------------------------------------------------------------
        target_size = image_embed.shape[-2:] # (16, 16)
        # target_size = image_embed.shape[-2:] # (32, 32)  # (orig 512)
        
        image_pe = F.interpolate(
            image_pe, size=target_size, mode="bilinear", align_corners=False
        )
        dense_embeddings = F.interpolate(
            dense_embeddings, size=target_size, mode="bilinear", align_corners=False
        )

        # 3. 채널 정합 (256c -> 32c / 64c)
        feat_s0_adapted = self.conv_s0(feat_s0_256c) # 64x64 텐서
        feat_s1_adapted = self.conv_s1(feat_s1_256c) # 32x32 텐서
        # feat_s0_adapted = self.conv_s0(feat_s0_256c) # 128x128 텐서  # (orig 512)
        # feat_s1_adapted = self.conv_s1(feat_s1_256c) # 64x64 텐서   # (orig 512)

        # 4. 마스크 디코딩 실행 (해상도가 완벽히 정합되어 에러 통과)
        pred_masks, pred_scores, _, _ = self.mask_decoder(
            image_embeddings=image_embed,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True, 
            repeat_image=True,      
            high_res_features=[feat_s0_adapted, feat_s1_adapted]
        )
        
        return pred_masks, pred_scores

def main():
    pt_path = "weights/sam2.1_t.pt"
    pte_box_decoder_path = "mobile_assets/sam2.1_t_box_decoder_256.pte"
    
    print("[Step 1] SAM 2.1 모델 로드 및 256 최적화 Box 디코더 생성...")
    sam_wrapper = SAM(pt_path)
    raw_model = sam_wrapper.model.eval()
    
    decoder_wrapper = SAM2BoxDecoder512(raw_model).eval()
    
    # ------------------------------------------------------------------------
    # [입력 준비] 256x256 통과 특성맵 파싱
    # ------------------------------------------------------------------------
    print("[Step 2] 256x256 백본 텐서 파싱 중...")
    dummy_image = torch.randn(1, 3, 256, 256)
    # dummy_image = torch.randn(1, 3, 512, 512)  # (orig 512)
    
    with torch.no_grad():
        encoder_output = raw_model.image_encoder(dummy_image)
    
    if isinstance(encoder_output, dict):
        image_embed = encoder_output.get("image_embed") or encoder_output.get("vision_features")
        raw_high_res = encoder_output.get("high_res_feats") or encoder_output.get("backbone_fpn")
    else:
        image_embed = encoder_output[0]
        raw_high_res = encoder_output[1:] if len(encoder_output) > 1 else []

    print(f"  -> 메인 임베딩 텐서 형태: {image_embed.shape}") # [1, 256, 16, 16]
    # print(f"  -> 메인 임베딩 텐서 형태: {image_embed.shape}") # [1, 256, 32, 32]  # (orig 512)

    feat_s0, feat_s1 = None, None
    for feat in raw_high_res:
        h, w = feat.shape[-2], feat.shape[-1]
        if h == 64 and w == 64:
            feat_s0 = feat
        elif h == 32 and w == 32:
            feat_s1 = feat
        # if h == 128 and w == 128:
        # 	feat_s0 = feat  # (orig 512)
        # elif h == 64 and w == 64:
        # 	feat_s1 = feat  # (orig 512)

    if feat_s0 is None or feat_s1 is None:
        raise RuntimeError("백본 출력에서 64x64 또는 32x32 해상도 텐서를 찾을 수 없습니다.")

    print(f"  -> 주입될 보조 특징맵 1 (64x64): {feat_s0.shape}")
    print(f"  -> 주입될 보조 특징맵 2 (32x32): {feat_s1.shape}")
    # print(f"  -> 주입될 보조 특징맵 1 (128x128): {feat_s0.shape}")  # (orig 512)
    # print(f"  -> 주입될 보조 특징맵 2 (64x64): {feat_s1.shape}")   # (orig 512)

    # 256 화면 기준 더미 바운딩 박스
    dummy_box = torch.tensor([[[25.0, 25.0, 225.0, 225.0]]], dtype=torch.float32)
    # dummy_box = torch.tensor([[[50.0, 50.0, 450.0, 450.0]]], dtype=torch.float32)  # (orig 512)

    decoder_inputs = (image_embed, feat_s0, feat_s1, dummy_box)
    
    # ------------------------------------------------------------------------
    # Step 3: ExecuTorch Tracing 및 직렬화
    # ------------------------------------------------------------------------
    print("\n[Step 3] Box 전용 디코더 EXIR 그래프 캡처 중 (torch.export)...")
    exported_program = export(decoder_wrapper, decoder_inputs)
    
    print("[Step 4] 모바일 최적화 하향(Edge IR) 및 XNNPACK CPU 백엔드 위임...")
    edge_program = to_edge(exported_program)
    try:
        edge_program = edge_program.to_backend(XnnpackPartitioner())
    except AssertionError as exc:
        print(f"[Warn] XNNPACK partition failed, exporting without backend: {exc}")
    
    print("[Step 5] 최종 512 Box 디코더 바이너리(.pte) 저장 중...")
    exec_program = edge_program.to_executorch()
    
    with open(pte_box_decoder_path, "wb") as f:
        f.write(exec_program.buffer)
        
    print(f"\n[성공] 256x256 환경에 완벽히 정합된 디코더 생성 완료!\n경로: {os.path.abspath(pte_box_decoder_path)}")

if __name__ == "__main__":
    main()