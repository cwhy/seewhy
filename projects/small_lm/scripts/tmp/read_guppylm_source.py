"""
Read the cached GuppyLM generate_data.py source from the agent output JSON.

Usage:
    uv run python projects/small_lm/scripts/tmp/read_guppylm_source.py
"""

import json
from pathlib import Path

SAVED = Path("/home/newuser/.claude/projects/-home-newuser-Projects-seewhy-projects/f6876b89-832e-4ace-9417-eb32e7c7a230/tool-results/toolu_01J1cA2YEBiH1ecUSjT9YDvY.json")

with open(SAVED) as f:
    data = json.load(f)

for item in data:
    if item.get("type") == "text":
        print(item["text"])
        break
