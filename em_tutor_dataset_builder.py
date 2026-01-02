import json

def validate_example(ex):
    required = ["id", "topic", "level", "prompt", "response"]
    for k in required:
        if k not in ex:
            raise ValueError(f"Missing key: {k}")
        if not isinstance(ex[k], str) or not ex[k].strip():
            raise ValueError(f"Key '{k}' must be a non-empty string")

def write_jsonl(path, examples):
    seen_ids = set()
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            validate_example(ex)
            if ex["id"] in seen_ids:
                raise ValueError(f"Duplicate id: {ex['id']}")
            seen_ids.add(ex["id"])
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    examples = [
        {
            "id": "em_0001",
            "topic": "Coulomb",
            "level": "intro",
            "prompt": (
                "You are a physics E&M tutor. Solve step-by-step and include a unit check.\n\n"
                "Problem: Two point charges q1 = +2.0 µC and q2 = -3.0 µC are separated by 0.50 m.\n"
                "Find the magnitude of the force on q1 due to q2.\n"
                "Use k = 8.99e9 N·m^2/C^2."
            ),
            "response": (
                "We use Coulomb’s law: F = k |q1 q2| / r^2.\n"
                "Convert: q1 = 2.0e-6 C, q2 = 3.0e-6 C, r = 0.50 m.\n"
                "Compute: F = (8.99e9) * (2.0e-6 * 3.0e-6) / (0.50^2)\n"
                "Numerator: 8.99e9 * 6.0e-12 = 5.394e-2\n"
                "Denominator: 0.25\n"
                "F = 0.05394 / 0.25 = 0.216 N.\n"
                "Unit check: (N·m^2/C^2)*(C^2)/m^2 = N.\n"
                "Answer: 0.216 N (attractive since charges are opposite)."
            ),
        }
    ]

    write_jsonl("em_tutor_train.jsonl", examples)
    print("Wrote em_tutor_train.jsonl with", len(examples), "examples")
