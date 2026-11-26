import json
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv()

client = genai.Client()
MODEL_ID = "gemini-3.5-flash-lite"


def load_prompt(name: str, version: str, **variables) -> str:
    prompt_path = Path(__file__).resolve().parent.parent / "prompts" / f"{name}_{version}.txt"
    template = prompt_path.read_text(encoding="utf-8")
    return template.format(**variables)


def load_golden_dataset() -> list[dict]:
    dataset_path = Path(__file__).resolve().parent / "golden.jsonl"
    with open(dataset_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def call_with_retry(prompt: str, retries: int = 5) -> str:
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0),
            )
            return response.text.strip()
        except errors.ClientError as e:
            if e.code == 429:
                print("  [Rate limit 15 RPM] waiting 60s...")
                time.sleep(60)
            else:
                raise
        except errors.ServerError:
            wait = 2 ** attempt
            print(f"  [Server busy] retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError("Model remained unavailable after retries.")


def run_eval(prompt_version: str) -> tuple[float, list[dict]]:
    dataset = load_golden_dataset()
    correct = 0
    failures = []

    print(f"Running evaluation for {prompt_version} ({len(dataset)} tickets)...")
    for item in dataset:
        prompt = load_prompt("classifier", prompt_version, ticket=item["ticket"])
        prediction = call_with_retry(prompt)
        time.sleep(4.2)  # Stay under 15 requests/minute free-tier limit

        if prediction == item["expected"]:
            correct += 1
        else:
            failures.append({
                "id": item["id"],
                "ticket": item["ticket"],
                "expected": item["expected"],
                "got": prediction,
            })

    total = len(dataset)
    accuracy = (correct / total) * 100
    print(f"Prompt {prompt_version}: {correct}/{total} correct ({accuracy:.0f}%)")
    return accuracy, failures


if __name__ == "__main__":
    acc_v1, fail_v1 = run_eval("v1")
    acc_v2, fail_v2 = run_eval("v2")

    if fail_v2:
        failed_ids = ", ".join(f"#{f['id']}" for f in fail_v2)
        print(f"Failing cases: {failed_ids}\n")
        for f in fail_v2:
            print(f"  Case #{f['id']}: '{f['ticket']}'")
            print(f"    Expected: {f['expected']} | Got: {f['got']}")
    else:
        print("Failing cases: None")

    THRESHOLD = 85.0
    if acc_v2 < THRESHOLD:
        print(f"\n[CI FAILED] Prompt v2 accuracy ({acc_v2:.0f}%) is below {THRESHOLD:.0f}% threshold!")
        sys.exit(1)
    else:
        print(f"\n[CI PASSED] Prompt v2 accuracy ({acc_v2:.0f}%) meets >= {THRESHOLD:.0f}% threshold.")
