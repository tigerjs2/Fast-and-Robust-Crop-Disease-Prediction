from transformers import CLIPTokenizer, CLIPTextModel
import torch
import argparse

def main(args):
    # 1. 텍스트용 구성 요소만 로드
    model_id = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(model_id)
    text_encoder = CLIPTextModel.from_pretrained(model_id)
    vegetable = args.vegetable.lower()
    # 2. 임베딩할 텍스트 리스트
    sentences = [f"a {vegetable} leaf", f"an abnormal {vegetable} leaf", f"a {vegetable} leaf with disease", f"a {vegetable} leaf with risk", f"a {vegetable} leaf with odd parts"]

    # 3. 토큰화 (Padding과 Truncation 설정)
    inputs = tokenizer(sentences, padding=True, return_tensors="pt")

    # 4. 임베딩 추출
    with torch.no_grad():
        outputs = text_encoder(**inputs)
    # 5. 결과값 선택
    # last_hidden_state: 문장의 각 단어(토큰)별 벡터 (batch_size, sequence_length, hidden_size)
    # pooler_output: 문장 전체를 대표하는 하나의 벡터 (batch_size, hidden_size)
    text_embeddings = outputs.pooler_output
    # 6. 임베딩 저장
    torch.save(text_embeddings, args.out)
    print(f"Text embeddings for '{vegetable}' saved to {args.out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text Embedding Program.")
    parser.add_argument("--vegetable", type=str, required=True, help="The type of vegetable to generate text embeddings for.")
    parser.add_argument("--out", type=str, required=True, help="Output file to save the embeddings.")
    args = parser.parse_args()
    main(args)    