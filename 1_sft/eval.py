import argparse
import json
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ================================
# argparse 인자 정의
# ================================
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True, help="모델 경로 또는 HuggingFace 모델 이름")
parser.add_argument(
    "--test_input",
    type=str,
    default="이젹의길동의슈단이신츌귀몰ᄒᆞ야팔도의횡ᄒᆡᆼᄒᆞ되능히알ᄌᆡ업ᄂᆞᆫ지라",
    help="추론 예시 입력 (기본값은 옛 한글 문장)"
)
parser.add_argument("--jsonl_path", type=str, default="honggildong_test.jsonl", help="테스트 jsonl 경로")
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
# JSONL 불러오기
# ================================
def load_jsonl(filename):
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

# ================================
# 단어 정답률 계산 함수
# ================================
def get_word_overlap(pred, ref):
    ref_words = set(ref.strip().split())
    pred_words = set(pred.strip().split())
    common = ref_words & pred_words
    return len(common) / len(ref_words) if ref_words else 0

# ================================
# 모델 출력 생성 함수
# ================================
def get_model_output(input_text, model, tokenizer):
    prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
입력의 옛 한글 내용을 현대의 한글로 변환해줘

### Input:
{input_text}

### Response:
"""
    messages = [{"role": "user", "content": prompt}]
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
            temperature=0.01,
            top_p=0.95,
            do_sample=True,
        )
    output_ids = outputs[0][inputs["input_ids"].shape[1]:]
    decoded = tokenizer.decode(output_ids, skip_special_tokens=False).strip()
    decoded = decoded.split("<|im_end|>")[0].strip()
    return decoded

# ================================
# 평가 루프
# ================================
results = []
for item in tqdm(load_jsonl(args.jsonl_path)):
    input_text = item["input"]
    target_text = item["output"]
    decoded = get_model_output(input_text, model, tokenizer)
    acc = get_word_overlap(decoded, target_text)
    results.append(acc)

accuracy = sum(results) / len(results)
print(f"\n✅ 전체 단어 포함 기준 Accuracy: {accuracy:.3f}")

# ================================
# 추론 예시 (선택적)
# ================================
if args.test_input:
    print("\n🔍 추론 예시:")
    print("Input:", args.test_input)
    print("Output:", get_model_output(args.test_input, model, tokenizer))
