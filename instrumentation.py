# instrumentation.py
from __future__ import annotations
import json, os, re
from datetime import datetime
from typing import Any, Dict

ACTION_RE = re.compile(r'^\s*(Fold|Call|Raise\s+(\d+(?:\.\d+)?)\s*bb)\s*$', re.I)

def ensure_parent(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    ensure_parent(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def utcnow_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def safe_first_line(s: str | None) -> str:
    if not s:
        return ""
    return s.splitlines()[0][:500]  # cap just in case

def parse_action_one_line(s: str | None) -> Dict[str, Any]:
    """Non-invasive parse; you DON'T have to use this for your analysis."""
    first = safe_first_line(s)
    m = ACTION_RE.match(first)
    if not m:
        return {"valid": False, "raw_first": first}
    act = m.group(1).strip()
    out = {"valid": True, "action": "RAISE" if act.lower().startswith("raise") else act.upper()}
    if out["action"] == "RAISE":
        out["x_bb"] = float(m.group(2))
    return out

# in names_matter_poker_study.py (or a helpers file)
def extract_first_token_logprobs(provider_response) -> dict | None:
    """
    Tries to find the first generated token's text and logprob.
    Returns dict like {"text": "...", "logprob": -0.42, "topk": [{"t": "Fold", "lp": -0.9}, ...]}
    or None if unavailable.
    """
    try:
        # Example: OpenAI responses API (chat.completions-like) with logprobs=True
        # response.choices[0].logprobs.content[0].top_logprobs -> list of {token, logprob}
        ch0 = provider_response.choices[0]
        lp_block = getattr(ch0, "logprobs", None)
        if lp_block and getattr(lp_block, "content", None):
            first_piece = lp_block.content[0]
            top = getattr(first_piece, "top_logprobs", None) or []
            out = {
                "text": getattr(first_piece, "token", None),
                "logprob": getattr(first_piece, "logprob", None),
                "topk": [{"t": getattr(t, "token", None), "lp": getattr(t, "logprob", None)} for t in top]
            }
            return out
    except Exception:
        pass
    try:
        # Example: Together generate endpoint with "logprobs": [{"token": "...", "logprob": ...}, ...]
        if "output" in provider_response and "logprobs" in provider_response["output"]:
            lps = provider_response["output"]["logprobs"]
            if lps:
                first = lps[0]
                # Some models include top-k; others don't
                out = {"text": first.get("token"), "logprob": first.get("logprob"), "topk": None}
                return out
    except Exception:
        pass
    return None
