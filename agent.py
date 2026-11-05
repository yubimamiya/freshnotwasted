"""Assistant layer: answers manager questions about the plan. Uses Claude when credentials exist, else simple rules."""
import json

SYSTEM = ("You help a grocery store manager run the fresh departments. Answer in 2-5 short sentences of plain language, "
          "with concrete numbers from the plan data. Never invent numbers that are not in the data.")

def _rules(q, s, d):
    ql = q.lower()
    for r in s["rows"]:
        if r["product"].split()[0].lower() in ql:
            why = ", ".join(f"{g} {v:+d}%" for g, v in r["drivers"]) or "normal pattern"
            return (f"{r['product']}: expect about {r['fc']} units (likely {r['lo']}–{r['hi']}). "
                    f"{r['usable']} usable on hand, so order {r['cases']} case(s) = {r['order']} units. Main factors: {why}.")
    if "storm" in ql:
        st = [a["title"] for a in s["actions"] if a["group"] == "Storm"]
        return "Storm actions: " + "; ".join(st[:5]) if st else "No storm warning in this scenario. Switch the scenario to 'Winter storm' to see the storm plan."
    if "donat" in ql:
        return "; ".join(f"{x['units']} {x['product']} to {x['partner']} ({x['pickup']})" for x in s["donations"]) or "No surplus is forecast for donation today."
    if "staff" in ql or "labor" in ql or "hour" in ql:
        return f"{s['kpi']['hours']} associate-hours today: " + ", ".join(f"{k} {v}h" for k, v in s["labor"].items() if v)
    if "trend" in ql:
        return "; ".join(f"{t['topic']} (index {t['now']}, {t['wk']:+d} this week) affects {', '.join(t['products'])}" for t in d["trends"])
    return "Try asking about a product (e.g. 'strawberries'), the storm, donations, staff hours, or trends."

def _context(s, d):
    return {"storm_scenario": d["storm"], "kpi": s["kpi"], "actions": s["actions"][:25], "labor_hours": s["labor"],
            "donations": s["donations"], "markdowns": s["markdowns"], "backtest_vs_current_rule": {
                d["backtest"][k + "_label"]: {("FreshCast" if pol == "tool" else "current rule"): {
                    "food_thrown_away_units": v["waste"], "food_thrown_away_usd": v["waste_usd"], "food_thrown_away_kg": v["waste_kg"],
                    "food_donated_kg": v["donated_kg"], "sales_lost_to_empty_shelves_units": v["lost"],
                    "percent_of_demand_met": v["fill"], "gross_profit_usd": v["profit"]}
                    for pol, v in d["backtest"][k].items()} for k in ("recent", "storm")},
            "note": "backtest covers all 3 stores; 'donations'/'markdowns' lists are only today's plan for this store",
            "current_rule": "baseline = order (same weekday last week's sales x 1.1) minus usable stock, rounded to cases; no markdowns/donations; ignores holidays, storms, trends",
            "products": [{k: r[k] for k in ("product", "on_hand", "usable", "fc", "lo", "hi", "order", "cases", "drivers", "err")} for r in s["rows"]],
            "trends": [{k: t[k] for k in ("topic", "now", "wk", "products")} for t in d["trends"]], "events": d["events"]}

def _claude_cli(q, s, d):
    """Answer through the locally installed, signed-in Claude Code CLI (no API key needed)."""
    import shutil, subprocess
    exe = shutil.which("claude") or shutil.which("claude.cmd")
    if not exe: return None
    prompt = (SYSTEM + " Do not use any tools; answer directly from the data below. Plain text, no markdown headings.\n\n"
              f"Plan data (JSON):\n{json.dumps(_context(s, d))}\n\nQuestion: {q}")
    r = subprocess.run([exe, "-p", "--output-format", "text", "--max-turns", "1"], input=prompt, capture_output=True,
                       text=True, encoding="utf-8", timeout=120)
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None

def answer(q, s, d):
    try:
        txt = _claude_cli(q, s, d)
        if txt: return {"source": "Claude (via Claude Code on this computer)", "text": txt}
    except Exception:
        pass
    try:
        import anthropic
        client = anthropic.Anthropic()
        ctx = {"storm_scenario": d["storm"], "kpi": s["kpi"], "actions": s["actions"][:25], "labor_hours": s["labor"],
               "donations": s["donations"], "markdowns": s["markdowns"],
               "products": [{k: r[k] for k in ("product", "on_hand", "usable", "fc", "lo", "hi", "order", "cases", "drivers")} for r in s["rows"]],
               "trends": [{k: t[k] for k in ("topic", "now", "wk", "products")} for t in d["trends"]], "events": d["events"]}
        r = client.messages.create(model="claude-opus-5-5", max_tokens=1500, output_config={"effort": "low"}, system=SYSTEM,
                                   messages=[{"role": "user", "content": f"Plan data:\n{json.dumps(ctx)}\n\nQuestion: {q}"}])
        if r.stop_reason == "refusal": raise RuntimeError
        return {"source": "Claude", "text": next(b.text for b in r.content if b.type == "text")}
    except Exception:
        return {"source": "built-in rules (no API key set)", "text": _rules(q, s, d)}
