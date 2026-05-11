# swift.py
# 将数据集转换为适合swift微调的格式

import json

input_file = "qrl_labeled.jsonl"
output_file = "swift_data.jsonl"

with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
    for line in fin:
        item = json.loads(line.strip())
        query = item["query"]
        response = item["response"]
        label = item["label"]  # 0 或 1
        # 构建 swift 所需的格式
        swift_sample = {
            "messages": [{"role": "user", "content": query}],
            "positive_messages": [[{"role": "user", "content": response}]],
            "label": label
        }
        fout.write(json.dumps(swift_sample, ensure_ascii=False) + "\n")