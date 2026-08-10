import argparse
import html
import json
from pathlib import Path
from typing import Dict, List


def load_jsonl(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def template_ok(text: str) -> bool:
    return all(token in text for token in ("Diagnosis:", "Fix:", "1.", "2."))


def has_prompt_leak(text: str) -> bool:
    return "### Instruction:" in text or "### Input:" in text or "### Response:" in text


def esc(text: str) -> str:
    return html.escape(text).replace("\n", "<br>")


def pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0.0%"
    return f"{(100.0 * numerator / denominator):.1f}%"


def avg_len(items: List[str]) -> float:
    if not items:
        return 0.0
    return sum(len(i) for i in items) / len(items)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate HTML summary for before/after fine-tuning outputs")
    parser.add_argument("--input_jsonl", default="outputs/runpod-finetune-outputs/before_after_outputs.jsonl")
    parser.add_argument("--output_html", default="outputs/runpod-finetune-outputs/before_after_summary.html")
    args = parser.parse_args()

    rows = load_jsonl(args.input_jsonl)
    if not rows:
        raise ValueError(f"No rows found in {args.input_jsonl}")

    total = len(rows)
    before_texts = [str(r.get("before", "")) for r in rows]
    after_texts = [str(r.get("after", "")) for r in rows]

    before_template_ok = sum(1 for t in before_texts if template_ok(t))
    after_template_ok = sum(1 for t in after_texts if template_ok(t))
    before_leak = sum(1 for t in before_texts if has_prompt_leak(t))
    after_leak = sum(1 for t in after_texts if has_prompt_leak(t))

    before_avg_chars = avg_len(before_texts)
    after_avg_chars = avg_len(after_texts)

    output_path = Path(args.output_html)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    row_html = []
    for idx, row in enumerate(rows, start=1):
        ticket = str(row.get("input", ""))
        before = str(row.get("before", ""))
        after = str(row.get("after", ""))

        before_ok = template_ok(before)
        after_ok = template_ok(after)
        before_bad = has_prompt_leak(before)
        after_bad = has_prompt_leak(after)

        row_html.append(
            """
            <tr>
              <td class=\"num\">{idx}</td>
              <td>{ticket}</td>
              <td>
                <div class=\"badges\">{before_ok_badge} {before_leak_badge}</div>
                <div class=\"cell-text\">{before}</div>
              </td>
              <td>
                <div class=\"badges\">{after_ok_badge} {after_leak_badge}</div>
                <div class=\"cell-text\">{after}</div>
              </td>
            </tr>
            """.format(
                idx=idx,
                ticket=esc(ticket),
                before=esc(before),
                after=esc(after),
                before_ok_badge="<span class='badge ok'>Template OK</span>" if before_ok else "<span class='badge warn'>Template Miss</span>",
                after_ok_badge="<span class='badge ok'>Template OK</span>" if after_ok else "<span class='badge warn'>Template Miss</span>",
                before_leak_badge="<span class='badge bad'>Prompt Leak</span>" if before_bad else "<span class='badge ok'>No Leak</span>",
                after_leak_badge="<span class='badge bad'>Prompt Leak</span>" if after_bad else "<span class='badge ok'>No Leak</span>",
            )
        )

    doc = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Fine-Tuning Before/After Summary</title>
  <style>
    :root {{
      --bg: #f5f7fb;
      --card: #ffffff;
      --text: #0f172a;
      --muted: #475569;
      --border: #dbe2ea;
      --ok: #15803d;
      --warn: #b45309;
      --bad: #b91c1c;
      --accent: #0f766e;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: radial-gradient(1000px 400px at 0% 0%, #e6fffa 0%, var(--bg) 60%);
      color: var(--text);
    }}
    .wrap {{ max-width: 1300px; margin: 32px auto; padding: 0 16px; }}
    .header {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 20px;
      margin-bottom: 16px;
    }}
    h1 {{ margin: 0 0 8px; font-size: 1.5rem; }}
    .subtitle {{ color: var(--muted); margin: 0; }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }}
    .metric {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px;
    }}
    .metric .k {{ color: var(--muted); font-size: 0.88rem; }}
    .metric .v {{ font-weight: 700; font-size: 1.2rem; margin-top: 4px; }}
    .table-wrap {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      overflow: auto;
    }}
    table {{ width: 100%; border-collapse: collapse; min-width: 1100px; }}
    th, td {{ border-bottom: 1px solid var(--border); padding: 12px; vertical-align: top; }}
    th {{ text-align: left; background: #f8fafc; position: sticky; top: 0; z-index: 1; }}
    .num {{ width: 52px; color: var(--muted); text-align: right; }}
    .cell-text {{ font-size: 0.92rem; line-height: 1.4; }}
    .badges {{ margin-bottom: 8px; }}
    .badge {{
      display: inline-block;
      margin-right: 6px;
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 0.75rem;
      border: 1px solid transparent;
    }}
    .ok {{ color: var(--ok); border-color: #86efac; background: #f0fdf4; }}
    .warn {{ color: var(--warn); border-color: #fcd34d; background: #fffbeb; }}
    .bad {{ color: var(--bad); border-color: #fca5a5; background: #fef2f2; }}
    .foot {{ color: var(--muted); font-size: 0.85rem; margin-top: 10px; }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <section class=\"header\">
      <h1>Fine-Tuning Before/After Summary</h1>
      <p class=\"subtitle\">Generated from {html.escape(args.input_jsonl)}</p>
    </section>

    <section class=\"metrics\">
      <div class=\"metric\"><div class=\"k\">Evaluation Examples</div><div class=\"v\">{total}</div></div>
      <div class=\"metric\"><div class=\"k\">Template Adherence (Before)</div><div class=\"v\">{before_template_ok}/{total} ({pct(before_template_ok, total)})</div></div>
      <div class=\"metric\"><div class=\"k\">Template Adherence (After)</div><div class=\"v\">{after_template_ok}/{total} ({pct(after_template_ok, total)})</div></div>
      <div class=\"metric\"><div class=\"k\">Prompt Leakage (Before)</div><div class=\"v\">{before_leak}/{total} ({pct(before_leak, total)})</div></div>
      <div class=\"metric\"><div class=\"k\">Prompt Leakage (After)</div><div class=\"v\">{after_leak}/{total} ({pct(after_leak, total)})</div></div>
      <div class=\"metric\"><div class=\"k\">Avg Response Length (Before)</div><div class=\"v\">{before_avg_chars:.1f} chars</div></div>
      <div class=\"metric\"><div class=\"k\">Avg Response Length (After)</div><div class=\"v\">{after_avg_chars:.1f} chars</div></div>
    </section>

    <section class=\"table-wrap\">
      <table>
        <thead>
          <tr>
            <th class=\"num\">#</th>
            <th>Ticket</th>
            <th>Before</th>
            <th>After</th>
          </tr>
        </thead>
        <tbody>
          {''.join(row_html)}
        </tbody>
      </table>
    </section>

    <p class=\"foot\">Tip: If Prompt Leakage is high, reduce generation spillover by adding stopping criteria or post-processing output trimming.</p>
  </div>
</body>
</html>
"""

    output_path.write_text(doc, encoding="utf-8")
    print(f"Saved HTML summary: {output_path}")


if __name__ == "__main__":
    main()
