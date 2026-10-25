"""Collect this CI run's measured numbers into docs/ci-results.{json,md}
and the GitHub job summary. Run from the repo root after the test,
benchmark and fault-injection steps have written their results-* files.
"""
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load(name):
    return json.loads((ROOT / name).read_text())


def main():
    bench60 = _load("results-bench-w60.json")
    bench120 = _load("results-bench-w120.json")
    pytest_log = (ROOT / "results-pytest.txt").read_text()
    fault_log = (ROOT / "results-faulttest.txt").read_text()

    m = re.search(r"(\d+) passed", pytest_log)
    tests_passed = int(m.group(1)) if m else None
    fault_ok = "ALL PASS" in fault_log

    results = {
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "runner": "GitHub Actions ubuntu-latest, 1 core (taskset -c 0)",
        "workflow_run": os.environ.get("GITHUB_SERVER_URL", "https://github.com")
        + "/" + os.environ.get("GITHUB_REPOSITORY", "")
        + "/actions/runs/" + os.environ.get("GITHUB_RUN_ID", ""),
        "tests_passed": tests_passed,
        "fault_injection_all_pass": fault_ok,
        "W60": bench60,
        "W120": bench120,
    }
    (ROOT / "docs" / "ci-results.json").write_text(json.dumps(results, indent=2))

    def row(w):
        p, f = w["primary"], w["fallback"]
        return (f"{p['macro_F1']:.4f} | {f['macro_F1']:.4f} | "
                f"{p['latency_ms_mean']:.1f} ms | {p['latency_ms_p95']:.1f} ms | "
                f"{p['n_windows']}")

    md = f"""# NPPAD-Edge — results measured in GitHub Actions CI

Measured {results['measured_at_utc']} on {results['runner']}.
Run: {results['workflow_run']}

- Test suite: **{tests_passed} passed**
- Fault-injection demo: **{"ALL PASS" if fault_ok else "FAILURES — see run log"}**
- Benchmark: held-out test windows, single core, bundle verified
  (SHA-256) before loading.

| Window | Primary macro-F1 | Fallback macro-F1 | Primary latency mean | Primary latency p95 | Windows |
|---|---|---|---|---|---|
| W = 60 s | {row(bench60)} |
| W = 120 s | {row(bench120)} |

Benchmark-process peak RSS (includes the replay harness): W = 60: {bench60['peak_rss_mb']:.0f} MB · W = 120: {bench120['peak_rss_mb']:.0f} MB (budget 512 MB).

Reference (full-precision study): W = 60 macro-F1 0.9838, W = 120
macro-F1 0.9876. Local-machine figures (contended shared VM) are in
`BENCHMARKS.md`; CI numbers are measured on different hardware and may
differ slightly.
"""
    (ROOT / "docs" / "ci-results.md").write_text(md)

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NPPAD-Edge — CI-measured results</title>
<style>body{{font:16px/1.6 -apple-system,"Segoe UI",Roboto,sans-serif;margin:2rem auto;max-width:820px;padding:0 1rem;color:#1a2332}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #d9e1ec;padding:.5rem .65rem;text-align:left}}
th{{background:#e8f4ef}}a{{color:#0b6e4f}}</style></head><body>
<h1>NPPAD-Edge — results measured in GitHub Actions CI</h1>
<p>Measured {results['measured_at_utc']} on {results['runner']}.<br>
Run: <a href="{results['workflow_run']}">{results['workflow_run']}</a></p>
<ul>
<li>Test suite: <strong>{tests_passed} passed</strong></li>
<li>Fault-injection demo: <strong>{"ALL PASS" if fault_ok else "FAILURES — see run log"}</strong></li>
<li>Benchmark: held-out test windows, single core, bundle SHA-256-verified before loading.</li>
</ul>
<table>
<tr><th>Window</th><th>Primary macro-F1</th><th>Fallback macro-F1</th><th>Primary latency mean</th><th>Primary latency p95</th><th>Windows</th></tr>
<tr><td>W = 60 s</td><td>{row(bench60)}</td></tr>
<tr><td>W = 120 s</td><td>{row(bench120)}</td></tr>
</table>
<p>Benchmark-process peak RSS (includes the replay harness): W = 60: {bench60['peak_rss_mb']:.0f} MB · W = 120: {bench120['peak_rss_mb']:.0f} MB (budget 512 MB).</p>
<p>Reference (full-precision study): W = 60 macro-F1 0.9838, W = 120 macro-F1 0.9876.
Local-machine figures are in <a href="https://github.com/saurabhkulkarni77/nppad-edge/blob/main/BENCHMARKS.md">BENCHMARKS.md</a>;
CI numbers are measured on different hardware and may differ slightly.</p>
<p><a href="index.html">← Back to the NPPAD-Edge demo page</a></p>
</body></html>
"""
    # row() yields pipe-separated cells; split them into <td>s for the table.
    def cells(r):
        return "".join(f"<td>{c.strip()}</td>" for c in r.split("|"))

    html = html.replace(f"<td>{row(bench60)}</td>", cells(row(bench60)))
    html = html.replace(f"<td>{row(bench120)}</td>", cells(row(bench120)))
    (ROOT / "docs" / "ci-results.html").write_text(html)

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as fh:
            fh.write(md)
    print(md)


if __name__ == "__main__":
    main()
