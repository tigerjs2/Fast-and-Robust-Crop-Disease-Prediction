import os
import torch
from ultralytics import SAM

# 기본 ExecuTorch 모듈
from torch.export import export
from executorch.exir import to_edge
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

# [양자화 전용 추가 모듈] PyTorch 2.x PT2E 양자화 파이프라인
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)

def main():
    pt_path = "weights/sam2.1_t.pt"
    pte_quantized_path = "mobile_assets/sam2.1_t_encoder_512_int8.pte"
    
    print("[Step 1] SAM 2.1 모델 로드 및 내부 Image Encoder 추출...")
    sam_wrapper = SAM(pt_path)
    raw_model = sam_wrapper.model.eval()
    image_encoder = raw_model.image_encoder
    
    # 512x512 해상도 맞춤형 더미 입력 생성
    example_input = (torch.randn(1, 3, 512, 512),)
    
    print("[Step 2] 1차 PyTorch Graph 캡처 (Pre-Autograd ATen Dialect)...")
    # 그래프 조작을 위해 export() 결과에서 순수 FX GraphModule 구조를 꺼냅니다.
    exported_program = export(image_encoder, example_input).module()
    
    # ------------------------------------------------------------------------
    # [Step 3] torchao + XNNPACK 백엔드 기반 가중치 INT8 양자화
    # ------------------------------------------------------------------------
    print("[Step 3] 가중치 INT8 양자화 연산 그래프 생성 중...")
    
    # ExecuTorch CPU(XNNPACK) 가속과 100% 호환되는 대칭형 양자화 설정 적용
    quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
    
    # 그래프에 양자화 관측 노드 삽입
    prepared_program = prepare_pt2e(exported_program, quantizer)
    
    # 동적 캘리브레이션 트리거 (더미 데이터를 주입해 활성화 값 범위 스케일 파악)
    prepared_program(*example_input)
    
    # 가중치를 실제 정수(INT8) 형식으로 결합 및 완전 치환
    quantized_program = convert_pt2e(prepared_program)
    print("  -> 양자화 그래프 변환 성공.")

    # ------------------------------------------------------------------------
    # Step 4: ExecuTorch 하향 컴파일 및 백본 위임
    # ------------------------------------------------------------------------
    print("\n[Step 4] 모바일 타겟 하향(Edge IR) 및 XNNPACK 백엔드 컴파일...")
    # 양자화 노드가 굳어진 최종 그래프를 모바일 배포용으로 완벽히 재캡처
    final_aten_dialect = export(quantized_program, example_input)
    
    edge_program = to_edge(final_aten_dialect)
    edge_program = edge_program.to_backend(XnnpackPartitioner())
    
    print("[Step 5] 최종 바이너리 직렬화 및 .pte 파일 저장...")
    exec_program = edge_program.to_executorch()
    
    with open(pte_quantized_path, "wb") as f:
        f.write(exec_program.buffer)
        
    print(f"\n[최종 성공] 512x512 INT8 양자화 인코더 파일이 추출되었습니다!")
    print(f"생성된 파일 경로: {os.path.abspath(pte_quantized_path)}")

if __name__ == "__main__":
    main()