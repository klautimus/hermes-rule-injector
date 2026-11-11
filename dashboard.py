"""Optional web dashboard for Rule Injector.

A small FastAPI app that reads the rule set and the activation trace from
disk. Because it reads files rather than in-process state, it works as a
standalone viewer whether or not a gateway is running.

    pip install fastapi uvicorn
    python3 dashboard.py --port 9123

Then open http://127.0.0.1:9123

Endpoints:
    GET /                       server-rendered page
    GET /api/status             counts and paths
    GET /api/rules              the rule set, without the inject bodies
    GET /api/trace?limit=N      the last N activation entries
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RULES = os.path.join(HERE, "rules.json")
DEFAULT_TRACE = os.path.expanduser("~/.hermes/logs/rule-injector-trace.jsonl")


def load_rules(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    rules = data.get("rules", data) if isinstance(data, dict) else data
    return rules if isinstance(rules, list) else []


def read_trace(path: str, limit: int = 50) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    entries: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries[-limit:][::-1]


def summarise(rules: List[Dict[str, Any]]) -> Dict[str, Any]:
    hooks: Dict[str, int] = {}
    for r in rules:
        hook = r.get("hook") or "none"
        hooks[hook] = hooks.get(hook, 0) + 1
    return {
        "total": len(rules),
        "enabled": sum(1 for r in rules if r.get("enabled")),
        "safety_critical": sum(1 for r in rules if r.get("safety_critical")),
        "by_hook": dict(sorted(hooks.items(), key=lambda kv: -kv[1])),
    }


def _clip(v: Any, n: int) -> str:
    """Truncate the RAW value, so escaping can never be cut mid-entity."""
    s = str(v)
    return s if len(s) <= n else s[: n - 1] + "\u2026"


def _esc(v: Any) -> str:
    s = str(v)
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace('"', "&quot;"))


def render_page(rules: List[Dict[str, Any]], trace: List[Dict[str, Any]]) -> str:
    s = summarise(rules)
    rows = []
    for r in sorted(rules, key=lambda r: (r.get("hook", ""), r.get("priority", 999))):
        on = r.get("enabled")
        badge = "on" if on else "off"
        rows.append(
            "<tr><td class=\"mono nowrap\">{id}</td><td class=nowrap>{hook}</td>"
            "<td class=num>{prio}</td>"
            "<td><span class='badge {badge}'>{state}</span></td><td>{name}</td></tr>".format(
                id=_esc(r.get("id", "")), hook=_esc(r.get("hook", "")),
                prio=_esc(r.get("priority", "")), badge=badge,
                state="enabled" if on else "disabled",
                name=_esc(r.get("name", ""))[:90],
            )
        )
    trows = []
    for e in trace[:30]:
        trows.append(
            "<tr><td class=\"mono nowrap\">{ts}</td><td class=\"mono nowrap\">{rule}</td>"
            "<td class=nowrap>{hook}</td><td class=\"mono ctx\">{ctx}</td></tr>".format(
                ts=_esc(str(e.get("timestamp", ""))[:19]), rule=_esc(e.get("rule_id", "")),
                hook=_esc(e.get("hook", "")),
                ctx=_esc(_clip(e.get("context_summary", ""), 90)),
            )
        )
    hook_cards = "".join(
        "<div class=card><div class=label>{h}</div><div class=value>{n}</div></div>".format(
            h=_esc(h), n=_esc(n))
        for h, n in s["by_hook"].items()
    )
    empty_rules = "<tr><td colspan=5 class=dim>no rules loaded</td></tr>"
    empty_trace = "<tr><td colspan=4 class=dim>no activations logged yet</td></tr>"
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Rule Injector</title>
<meta http-equiv="refresh" content="10">
<style>
  body{{font-family:system-ui,-apple-system,sans-serif;margin:0;background:#0d1117;color:#e6edf3}}
  header{{padding:18px 28px;background:#161b22;border-bottom:1px solid #30363d;display:flex;
         align-items:baseline;gap:14px}}
  h1{{margin:0;font-size:18px;font-weight:600}}
  .sub{{color:#8b949e;font-size:12px}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;padding:20px 28px}}
  .card{{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:14px}}
  .label{{font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:.05em}}
  .value{{font-size:26px;font-weight:600;margin-top:4px;font-family:ui-monospace,monospace}}
  section{{padding:0 28px 28px}}
  h2{{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:#8b949e;margin:18px 0 8px}}
  table{{width:100%;border-collapse:collapse;font-size:13px}}
  th,td{{padding:6px 8px;text-align:left;border-bottom:1px solid #21262d}}
  th{{color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:.05em;font-weight:500}}
  .mono{{font-family:ui-monospace,monospace}}
  .nowrap{{white-space:nowrap}}
  .ctx{{max-width:560px;overflow:hidden;text-overflow:ellipsis;color:#8b949e}}
  .num{{text-align:right;font-family:ui-monospace,monospace}}
  .badge{{padding:2px 7px;border-radius:3px;font-size:11px;font-weight:600}}
  .badge.on{{background:#1f6f37;color:#e6edf3}} .badge.off{{background:#6e7681;color:#fff}}
  .dim{{color:#6e7681;font-style:italic}}
</style></head><body>
<header><h1>Rule Injector</h1>
<span class=sub>{_esc(os.path.relpath(DEFAULT_RULES, os.getcwd()) if os.getcwd() else "rules.json")}
 &middot; auto-refresh 10s</span></header>
<div class=grid>
  <div class=card><div class=label>rules</div><div class=value>{s["total"]}</div></div>
  <div class=card><div class=label>enabled</div><div class=value>{s["enabled"]}</div></div>
  <div class=card><div class=label>safety critical</div><div class=value>{s["safety_critical"]}</div></div>
  <div class=card><div class=label>logged activations</div><div class=value>{len(trace)}</div></div>
</div>
<div class=grid>{hook_cards}</div>
<section>
  <h2>Recent activations</h2>
  <table><tr><th>time (utc)</th><th>rule</th><th>hook</th><th>context</th></tr>
  {"".join(trows) or empty_trace}</table>
  <h2>Rule set</h2>
  <table><tr><th>id</th><th>hook</th><th>prio</th><th>state</th><th>name</th></tr>
  {"".join(rows) or empty_rules}</table>
</section></body></html>"""


def build_app(rules_path: str = DEFAULT_RULES, trace_path: str = DEFAULT_TRACE):
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse

    app = FastAPI(title="Rule Injector")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return render_page(load_rules(rules_path), read_trace(trace_path, 30))

    @app.get("/api/status")
    def status() -> Any:
        rules = load_rules(rules_path)
        trace = read_trace(trace_path, 1)
        return JSONResponse({
            **summarise(rules),
            "rules_path": rules_path,
            "trace_path": trace_path,
            "trace_present": os.path.exists(trace_path),
            "last_activation": trace[0] if trace else None,
        })

    @app.get("/api/rules")
    def rules() -> Any:
        return JSONResponse([
            {k: r.get(k) for k in
             ("id", "name", "hook", "priority", "enabled", "safety_critical", "domain")}
            for r in load_rules(rules_path)
        ])

    @app.get("/api/trace")
    def trace(limit: int = 50) -> Any:
        return JSONResponse(read_trace(trace_path, max(1, min(limit, 1000))))

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Rule Injector dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=9123, type=int)
    parser.add_argument("--rules", default=DEFAULT_RULES)
    parser.add_argument("--trace", default=DEFAULT_TRACE)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(build_app(args.rules, args.trace), host=args.host, port=args.port,
                log_level="info")


if __name__ == "__main__":
    main()
