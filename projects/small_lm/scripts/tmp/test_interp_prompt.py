"""Test the actual interpretation prompt with a fake model answer."""
import sys, json, time
from pathlib import Path

SMALL_LM = Path(__file__).parent.parent.parent
sys.path.insert(0, str(SMALL_LM))
sys.path.insert(0, str(SMALL_LM.parent.parent))

from ai_eval.openrouter import OpenRouterClient
from lib.prompts import fill, INTERP_SYSTEM, INTERP_USER

identity  = (SMALL_LM / "data" / "identity" / "lighthouse.md").read_text()
question  = "What was the most memorable storm you ever rode out here?"
answer    = "<n1e.ce<n5i3g7k6j8l3g7k.c6j-b6j3g0d0d3g"

facts_block = "(none yet)"
user_msg    = fill(INTERP_USER, identity=identity, facts_block=facts_block,
                   question=question, answer=answer)

print(f"Prompt length: {len(user_msg)} chars", flush=True)
print(f"System length: {len(INTERP_SYSTEM)} chars", flush=True)

for model in ["inception/mercury-2", "openai/gpt-4.1-mini"]:
    print(f"\n{'='*60}", flush=True)
    print(f"Model: {model}", flush=True)
    try:
        client = OpenRouterClient(model=model, timeout=30.0)
        t0 = time.perf_counter()
        raw = client.chat(user_msg, system=INTERP_SYSTEM, temperature=0.3, max_tokens=256, retries=0)
        print(f"  time: {time.perf_counter()-t0:.1f}s", flush=True)
        print(f"  raw: {raw!r}", flush=True)
        try:
            data = json.loads(raw)
            print(f"  exact_sentence: {data.get('exact_sentence')!r}", flush=True)
            print(f"  new_facts: {data.get('new_facts')}", flush=True)
        except json.JSONDecodeError as e:
            print(f"  JSON parse failed: {e}", flush=True)
        client.close()
    except Exception as e:
        print(f"  EXCEPTION: {type(e).__name__}: {e}", flush=True)

print("\ndone", flush=True)
