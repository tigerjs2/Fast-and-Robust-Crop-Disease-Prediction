import os
import torch
from ultralytics import SAM

from torch.export import export
from executorch.exir import to_edge
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

def main():
    pt_path = "weights/sam2.1_t.pt"
    pte_encoder_path = "mobile_assets/sam2.1_t_encoder1024.pte"
    
    print("[Step 1] SAM 2.1 모델 로드 및 Image Encoder 분리...")
    sam_wrapper = SAM(pt_path)
    raw_model = sam_wrapper.model.eval()
    image_encoder = raw_model.image_encoder
    
    # [핵심 수정] 모바일 최적화를 위해 1024x1024 입력 텐서 주입
    encoder_input = (torch.randn(1, 3, 1024, 1024),)
    
    print("[Step 2] 1024x1024 규격 Image Encoder EXIR 캡처 중 (torch.export)...")
    exported_program = export(image_encoder, encoder_input)
    
    print("[Step 3] 모바일 하향(Edge IR) 및 XNNPACK CPU 백엔드 위임...")
    edge_program = to_edge(exported_program)
    edge_program = edge_program.to_backend(XnnpackPartitioner())
    
    print("[Step 4] 최종 인코더 바이너리(.pte) 저장 중...")
    exec_program = edge_program.to_executorch()
    
    with open(pte_encoder_path, "wb") as f:
        f.write(exec_program.buffer)
        
    print(f"\n[성공] 1024x1024 Image Encoder 변환 완료!\n경로: {os.path.abspath(pte_encoder_path)}")

if __name__ == "__main__":
    main()