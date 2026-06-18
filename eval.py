import json
import time

with open("golden_set.json", "r") as f:
    test_cases = json.load(f)

results = []

for test in test_cases:

    question = test["question"]

    start = time.time()

    # Placeholder answer
    answer = "Test Answer"

    latency = round((time.time() - start) * 1000, 2)

    results.append({
        "question": question,
        "answer": answer,
        "latency_ms": latency
    })

print("\nEvaluation Results\n")

for r in results:
    print(r)