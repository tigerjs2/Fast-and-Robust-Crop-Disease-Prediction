# SERVER_SPEC.md

`scripts/inference_single.py` 분석 정리 및 FastAPI 서버 래핑 설계 제안.

## 1. 입력값

### 이미지
- **전달 방식**: 파일 경로 (`image_path: str`) → 내부에서 `SAM2MaskGenerator.generate_with_region`이 경로로부터 OpenCV로 로딩.
- **포맷**: OpenCV가 읽을 수 있는 일반 이미지 파일 (JPG/PNG 등). 내부에서 BGR로 읽힌 뒤 SAM 처리 후 RGB로 변환됨.

### bbox
- **형식**: `xyxy` — `(x1, y1, x2, y2)` 정수 튜플 (픽셀 좌표, 원본 이미지 기준).
- **용도**: SAM2를 `mode="box"`로 동작시켜 해당 박스 안의 객체 마스크 생성을 유도.

### crop_name
- 문자열. 내부에서 `strip().lower()`로 정규화.
- `data/TextEmbeddings/` 디렉터리에 해당 작물의 `(5, D)` CLIP 텍스트 임베딩 파일이 존재해야 함.

### 전처리 파이프라인
1. `SAM2MaskGenerator(mode="box")`로 bbox에서 마스크 생성 → 배경 제거된 BGR 이미지 `masked_bgr` 반환.
2. `cv2.cvtColor(BGR2RGB)`로 RGB 변환.
3. `build_image_transform(image_size=checkpoint.config.image_size or 224, apply_augmentation=False)` 적용.
4. `unsqueeze(0).to(device)` → `(1, 3, H, W)` 텐서.
5. 텍스트 쿼리: `load_crop_text_embeddings(..., expected_num_queries=5)` → `(1, 5, D)`로 unsqueeze.

## 2. 출력값

`run_single_inference` 반환 dict:

| 키 | 타입 | 설명 |
|---|---|---|
| `pred_index` | `int` | 예측 클래스 인덱스 |
| `pred_class` | `str` | 예측 클래스 이름 |
| `pred_confidence` | `float` | softmax 확률 (top-1) |
| `class_names` | `List[str]` | 전체 클래스 이름 |
| `probs` | `np.ndarray` (shape `(C,)`) | 클래스별 softmax 확률 |
| `masked_bgr` | `np.ndarray` (HxWx3, uint8, BGR) | 배경 제거된 이미지 |
| `region_info` | `Dict[str, float]` | `x1,y1,x2,y2,area_pixels,area_ratio` |
| `mask_bool` | `np.ndarray` (HxW, bool) | SAM 바이너리 마스크 |

> JSON 직렬화 시 `probs`, `masked_bgr`, `mask_bool`은 변환 필요 (list / base64 / 파일 등).

## 3. 의존 라이브러리 (requirements)

런타임 코어:
- `torch` (CUDA 지원 빌드 권장)
- `torchvision` (transforms)
- `numpy`
- `opencv-python`
- `sam2` (Meta SAM2 패키지 — `base_model/SAM.py`에서 사용)
- `transformers` / `open_clip_torch` 등 텍스트 임베딩 생성 시 (런타임에는 사전 생성된 `.pt`만 로드하므로 불필요할 수 있음 — 확인 필요)

FastAPI 서버 추가:
- `fastapi`
- `uvicorn[standard]`
- `python-multipart` (파일 업로드)
- `pydantic`

## 4. 모델 파일 로딩 방식

- **SAM2**: `weights/sam2.1_t.pt` → `SAM2MaskGenerator(model_path=..., mode="box")` (호출 시점에 매번 인스턴스 생성됨 → 서버화 시 1회 로드 권장).
- **2-stage classifier**: `weights/mob3_1/best_classifier.pth` → `torch.load(..., map_location=device)`로 dict 로드.
  - dict 내부: `class_names`, `config` (backbone_name, image_size, embed_dim, num_heads, attn_dropout 등), state_dict.
  - `build_model_from_checkpoint(checkpoint, num_classes, device)` 후 `model.eval()`.
- **텍스트 임베딩**: `data/TextEmbeddings/{crop}.pt` 등을 `load_crop_text_embeddings`로 dict 로드 (`(5, D)` 텐서 × N작물).

> 현재 구현은 매 요청마다 전체 로드 → **서버에서는 모듈 전역에 캐시**해야 함.

## 5. FastAPI 서버 엔드포인트 설계 제안

### 5.1 부팅 시 1회 로드 (lifespan / startup)
```
state = {
    "device": torch.device(...),
    "checkpoint": ...,
    "model": SecondStageClassifier(...).eval(),
    "class_names": [...],
    "transform": build_image_transform(image_size, apply_augmentation=False),
    "sam_generator": SAM2MaskGenerator(model_path=..., mode="box"),
    "text_embeddings": load_crop_text_embeddings(..., crop_names=ALL_CROPS),
}
```

### 5.2 엔드포인트

#### `GET /health`
- 헬스체크. `{"status": "ok", "device": "cuda"}`.

#### `GET /crops`
- 사용 가능한 작물 목록 반환. `text_embeddings.keys()`.

#### `GET /classes`
- `class_names` 반환.

#### `POST /predict` (메인)
- **Content-Type**: `multipart/form-data`
- **Form fields**:
  - `image`: `UploadFile` (jpg/png)
  - `crop`: `str`
  - `bbox`: `str` — `"x1,y1,x2,y2"` 또는 JSON 배열
  - (선택) `return_mask`: `bool` (기본 false)
  - (선택) `return_masked_image`: `bool` (기본 false)
- **처리**:
  1. 업로드 파일을 임시 경로에 저장 (또는 메모리에서 `cv2.imdecode`로 처리하도록 `SAM` 래퍼 확장 권장).
  2. `run_single_inference`의 로직을 캐시된 state로 재구성한 함수 호출.
- **응답 (JSON)**:
  ```json
  {
    "pred_index": 3,
    "pred_class": "downy_mildew",
    "pred_confidence": 0.9123,
    "probs": [0.01, 0.02, ...],
    "region_info": {"x1":..,"y1":..,"x2":..,"y2":..,"area_pixels":..,"area_ratio":..},
    "masked_image_png_base64": "..."   // return_masked_image=true 일 때만
  }
  ```

#### (선택) `POST /predict/json`
- bbox + base64 이미지를 JSON으로 받는 변형 (모바일 친화).

### 5.3 Pydantic 스키마 예시
```python
class BBox(BaseModel):
    x1: int; y1: int; x2: int; y2: int

class PredictResponse(BaseModel):
    pred_index: int
    pred_class: str
    pred_confidence: float
    probs: list[float]
    region_info: dict[str, float]
    masked_image_png_base64: str | None = None
```

### 5.4 권장 사항
- GPU 사용 시 요청 동시성 제어를 위해 `asyncio.Lock` 또는 워커 1개 + 큐.
- 업로드 이미지 크기 제한 (`MAX_UPLOAD_SIZE`).
- SAM 래퍼가 경로만 받으므로, 메모리 ndarray도 받을 수 있도록 작은 어댑터 추가 권장 (디스크 I/O 절감).
- 에러 케이스:
  - 알 수 없는 crop → 400 + 사용 가능한 목록.
  - bbox 형식 오류 → 422.
  - 이미지 디코딩 실패 → 400.
