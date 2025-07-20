from datasets import load_dataset
import json
import random

# ✅ 데이터셋 불러오기
dataset = load_dataset("mks0813/mbti-f-t-style-responses")
data = list(dataset["train"])  # list로 변환

# ✅ 섞기
random.shuffle(data)

n_total = len(data)
n_test = max(1, int(n_total * 0.01))  # 최소 1개는 test로

test_data = data[:n_test]
train_data = data[n_test:]

# ✅ JSONL 저장 함수
def save_jsonl(filename, data):
    with open(filename, "w", encoding="utf-8") as f:
        for row in data:
            convsersation = row.get("conversation", "").strip()
            f_style_response = row.get("f_style_response", "").strip()
            t_style_response = row.get("t_style_response", "").strip()
            json_obj = {
                "question": convsersation,
                "chosen": f_style_response,
                "rejected": t_style_response
            }
            f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")

# ✅ 파일 저장
save_jsonl("./mbti_train.jsonl", train_data)
save_jsonl("./mbti_test.jsonl", test_data)

print(f"✅ train: {len(train_data)}개, test: {len(test_data)}개 저장 완료!")