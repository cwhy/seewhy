"""Quick test of OpenRouter models — checks raw response and content field."""
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ai_eval.openrouter import OpenRouterClient
import httpx, os
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent.parent.parent / ".env")
API_KEY = os.environ.get("OPENROUTER_KEY", "")

PROMPT = 'Reply with JSON only: {"answer": "hello"}'
MODELS = ["inception/mercury-2", "openai/gpt-4.1-mini"]

def raw_request(model: str) -> dict:
    resp = httpx.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={"model": model, "messages": [{"role": "user", "content": PROMPT}],
              "temperature": 0.1, "max_tokens": 64},
        timeout=30,
    )
    return resp.json()

for model in MODELS:
    print(f"\n{'='*60}")
    print(f"Model: {model}")
    try:
        data = raw_request(model)
        if "choices" in data:
            msg = data["choices"][0]["message"]
            print(f"  message keys:  {list(msg.keys())}")
            print(f"  content:       {msg.get('content')!r}")
            for k, v in msg.items():
                if k != "content":
                    print(f"  {k}: {str(v)[:120]!r}")
        elif "error" in data:
            print(f"  ERROR: {data['error']}")
        else:
            print(f"  unexpected: {data}")
    except Exception as e:
        print(f"  EXCEPTION: {e}")

print("\ndone")
