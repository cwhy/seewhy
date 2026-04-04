"""
Wait for exp_ema to finish (log contains "Done."), then:
  Step 1: run exp_noaug_512, exp_noaug_1024, exp_noaug_2048
  Step 2: run gen_viz_latent_comparison.py
Prints final probe_acc for each experiment and the saved URL.
"""

import subprocess
import sys
import time
import os

LOG_PATH = "/home/newuser/Projects/seewhy/projects/ssl/logs/exp_ema.log"
POLL_INTERVAL = 15  # seconds
PROJECT_DIR = "/home/newuser/Projects/seewhy"


def tail_file(path):
    try:
        with open(path, "r") as f:
            return f.read()
    except FileNotFoundError:
        return ""


def wait_for_done():
    print("Waiting for exp_ema to finish...", flush=True)
    while True:
        content = tail_file(LOG_PATH)
        if "Done." in content:
            print("exp_ema is done. Proceeding.", flush=True)
            return
        # Also check for error/crash indicators
        lines = content.strip().splitlines()
        last_line = lines[-1] if lines else ""
        print(f"  exp_ema still running... last line: {last_line}", flush=True)
        time.sleep(POLL_INTERVAL)


def run_step1():
    print("\n--- Step 1: Running no-augmentation ablations ---", flush=True)
    cmd = [
        "uv", "run", "python",
        "projects/ssl/scripts/run_experiments.py",
        "exp_noaug_512", "exp_noaug_1024", "exp_noaug_2048"
    ]
    result = subprocess.run(cmd, cwd=PROJECT_DIR, capture_output=True, text=True)
    print("STDOUT:\n", result.stdout, flush=True)
    if result.stderr:
        print("STDERR:\n", result.stderr, flush=True)
    if result.returncode != 0:
        print(f"ERROR: run_experiments.py exited with code {result.returncode}", flush=True)
        sys.exit(result.returncode)
    return result.stdout, result.stderr


def extract_probe_acc(exp_name):
    """Read the experiment log and extract the final probe_acc."""
    log_path = f"/home/newuser/Projects/seewhy/projects/ssl/logs/{exp_name}.log"
    try:
        with open(log_path, "r") as f:
            content = f.read()
    except FileNotFoundError:
        return f"(log not found: {log_path})"
    # Find all probe_acc= occurrences, take the last one
    import re
    matches = re.findall(r"probe_acc=([\d.]+%)", content)
    if matches:
        return matches[-1]
    # Maybe it's reported as a decimal
    matches = re.findall(r"probe_acc=([\d.]+)", content)
    if matches:
        return matches[-1]
    return "(not found in log)"


def run_step2():
    print("\n--- Step 2: Generating visualization ---", flush=True)
    cmd = [
        "uv", "run", "python",
        "projects/ssl/scripts/gen_viz_latent_comparison.py"
    ]
    result = subprocess.run(cmd, cwd=PROJECT_DIR, capture_output=True, text=True)
    print("STDOUT:\n", result.stdout, flush=True)
    if result.stderr:
        print("STDERR:\n", result.stderr, flush=True)
    if result.returncode != 0:
        print(f"ERROR: gen_viz_latent_comparison.py exited with code {result.returncode}", flush=True)
    return result.stdout, result.stderr, result.returncode


def extract_url(text):
    """Extract a saved URL from script output."""
    import re
    # Common patterns: "Saved to https://...", "URL: https://...", just a bare https:// line
    patterns = [
        r"https?://\S+",
        r"Saved[:\s]+(\S+)",
        r"url[:\s]+(\S+)",
        r"URL[:\s]+(\S+)",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m.group(0) if pat.startswith("https") else m.group(1)
    return "(URL not found in output)"


if __name__ == "__main__":
    wait_for_done()

    stdout1, stderr1 = run_step1()

    # Extract probe_acc for each experiment
    for exp in ["exp_noaug_512", "exp_noaug_1024", "exp_noaug_2048"]:
        acc = extract_probe_acc(exp)
        print(f"  {exp}: final probe_acc = {acc}", flush=True)

    stdout2, stderr2, rc2 = run_step2()

    combined_output = stdout2 + "\n" + stderr2
    url = extract_url(combined_output)
    print(f"\nVisualization saved URL: {url}", flush=True)
    print("\n--- All done ---", flush=True)
