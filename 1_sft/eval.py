from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
from tqdm import tqdm

model_path = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype="auto",
    device_map={"": "cuda:0"},
)

model.eval()

# json 파일을 불러오는 함수
def load_jsonl(filename):
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

# 예측 단어 중 정답과 겹치는 단어의 비율 계산 함수
def get_word_overlap(pred, ref):
    ref_words = set(ref.strip().split())
    pred_words = set(pred.strip().split())
    common = ref_words & pred_words
    return len(common) / len(ref_words)

# 모델의 출력을 얻는 함수
def get_model_output(input_text, model, tokenizer):
    # 프롬프트 템플릿 적용
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

# -- 평가 루프 --
results = []
jsonl_path = "honggildong_test.jsonl"

for item in tqdm(load_jsonl(jsonl_path)):
    input_text = item["input"]
    target_text = item["output"]
    decoded = get_model_output(input_text, model, tokenizer)
    acc = get_word_overlap(decoded, target_text)
    results.append(acc)

accuracy = sum(results) / len(results)
print(f"전체 단어 포함 기준 Accuracy: {accuracy:.3f}")

# -- 추론 예시 --
test_input = "이젹의길동의슈단이신츌귀몰ᄒᆞ야팔도의횡ᄒᆡᆼᄒᆞ되능히알ᄌᆡ업ᄂᆞᆫ지라"
print(get_model_output(test_input, model, tokenizer))