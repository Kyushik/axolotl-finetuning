from datasets import load_dataset
import json
import random

# ✅ 데이터셋 불러오기
dataset = load_dataset("bebechien/HongGildongJeon")
data = list(dataset["train"])  # list로 변환

# ✅ 섞기
random.shuffle(data)
n_total = len(data)
n_test = max(1, int(n_total * 0.05))  # 최소 1개는 test로

test_data = data[:n_test]
train_data = data[n_test:]

# ✅ JSONL 저장 함수
def save_jsonl(filename, data):
	with open(filename, "w", encoding="utf-8") as f:
		for row in data:
			original_text = row.get("original", "").strip()
			modern_translation_text = row.get("modern translation", "").strip()
			
			json_obj = {
				"instruction": "입력의 옛 한글 내용을 현대의 한글로 변환해줘",
				"input": original_text,
				"output": modern_translation_text
			}
			
			f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")

# ✅ 파일 저장
save_jsonl("./honggildong_train.jsonl", train_data)
save_jsonl("./honggildong_test.jsonl", test_data)

print(f"✅ train: {len(train_data)}개, test: {len(test_data)}개 저장 완료!")