import argparse
import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ================================
# argparse 인자 정의
# ================================
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True, help="모델 경로 또는 HuggingFace 모델 이름")
parser.add_argument(
    "--test_input",
    type=str,
    default="나 친구랑 막 싸웠어",
    help="모델에 입력할 사용자 텍스트 (기본값 있음)"
)
args = parser.parse_args()

# ================================
# 모델 및 토크나이저 로드
# ================================
tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    trust_remote_code=True,
    torch_dtype="auto",
    device_map={"": "cuda:0"},
)
model.eval()

# ================================
# 모델 출력 생성 함수
# ================================
def get_model_output(input_text, model, tokenizer):
    messages = [{"role": "user", "content": input_text}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.6,
            top_p=0.95,
            do_sample=True,
        )
    output_ids = outputs[0][inputs["input_ids"].shape[1]:]
    decoded = tokenizer.decode(output_ids, skip_special_tokens=False).strip()
    decoded = decoded.split("<|im_end|>")[0].strip()
    return decoded

# ================================
# 추론 실행
# ================================
print("\n🔍 모델 응답:")
print(get_model_output(args.test_input, model, tokenizer))