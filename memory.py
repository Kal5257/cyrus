import json, time, re
from pathlib import Path

DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)
FACTS_PATH = DATA_DIR / "facts.json"
HIST_PATH  = DATA_DIR / "history.jsonl"

DEFAULT_FACTS = {
    "profile": {"name": "Kal", "pronouns": "she/her"},  
    "preferences": {},         
    "facts": []                
}

# ---------- load/save ----------
def load_facts():
    if FACTS_PATH.exists():
        try:
            return json.loads(FACTS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return DEFAULT_FACTS.copy()

def save_facts(facts):
    FACTS_PATH.write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8")

def append_history(message_obj: dict):
    with HIST_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(message_obj, ensure_ascii=False) + "\n")

def load_recent_history(limit=50):
    if not HIST_PATH.exists():
        return []
    lines = HIST_PATH.read_text(encoding="utf-8").splitlines()[-limit:]
    return [json.loads(x) for x in lines if x.strip()]

# ---------- mutate/query facts ----------
def add_fact(facts, text):
    text = text.strip()
    if not text:
        return False
    # de-dupe simple
    if any(f["text"].lower() == text.lower() for f in facts["facts"]):
        return False
    facts["facts"].append({"text": text, "added_at": int(time.time())})
    return True

def forget_fact(facts, needle):
    needle_l = needle.lower().strip()
    before = len(facts["facts"])
    facts["facts"] = [f for f in facts["facts"] if needle_l not in f["text"].lower()]
    return len(facts["facts"]) != before

def summarize_facts(facts, max_items=10):
    core = facts.get("profile", {})
    pref = facts.get("preferences", {})
    lines = []
    if core: lines.append(f"Profile: {core}")
    if pref: lines.append(f"Preferences: {pref}")
    if facts.get("facts"):
        # newest first
        items = sorted(facts["facts"], key=lambda f: f["added_at"], reverse=True)[:max_items]
        lines.append("Recent facts: " + "; ".join(f["text"] for f in items))
    return "\n".join(lines).strip()

# ---------- simple intent hooks ----------

REMEMBER_CMD = re.compile(r"^\s*(?:please\s*)?remember\s*[:\-]?\s*(?P<payload>.+?)\s*$", re.I)
FORGET_CMD   = re.compile(r"^\s*(?:please\s*)?forget\s*[:\-]?\s*(?P<payload>.+?)\s*$", re.I)



def parse_memory_command(text: str):
    """Return ('remember', payload) or ('forget', payload) or None.
       Only triggers if the message STARTS with remember/forget (imperative)."""
    s = (text or "").strip()
    if not s:
        return None

    # Exact command form only (no mid-sentence triggers)
    m = REMEMBER_CMD.match(s)
    if m:
        payload = m.group("payload").strip()
        # Optional safety: ignore questions like "remember when...?"
        if payload.endswith("?"):
            return None
        return ("remember", payload)

    m = FORGET_CMD.match(s)
    if m:
        payload = m.group("payload").strip()
        if payload.endswith("?"):
            return None
        return ("forget", payload)

    return None
