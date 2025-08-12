# memory.py
import json, time, re
from pathlib import Path

DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)
FACTS_PATH = DATA_DIR / "facts.json"
HIST_PATH  = DATA_DIR / "history.jsonl"

DEFAULT_FACTS = {
    "profile": {"name": "Kal", "pronouns": "she/her"},  # seed defaults; edit as you like
    "preferences": {},          # e.g., {"voice": "ryan-high", "music": "lofi"}
    "facts": []                 # list of {"text": "...", "added_at": 1690000000}
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

REMEMBER_PAT = re.compile(r"\b(remember|note|save)\b(?:\s+that)?\s+(?P<payload>.+)$", re.I)
FORGET_PAT   = re.compile(r"\b(forget|delete|remove)\b\s+(?P<payload>.+)$", re.I)

def parse_memory_command(text: str):
    """Return ('remember', payload) or ('forget', payload) or None."""
    s = (text or "").strip()
    if not s:
        return None
    # search anywhere in the sentence (not just at start)
    if m := REMEMBER_PAT.search(s):
        return ("remember", m.group("payload").strip())
    if m := FORGET_PAT.search(s):
        return ("forget", m.group("payload").strip())
    return None
