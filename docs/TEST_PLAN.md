# Cyrus AI Assistant – Test Plan

_Last updated: 2025-08-26_

---

## 1. Purpose

This document defines the testing approach for the **Cyrus voice assistant** project.  
The goal is to ensure stability, correctness, and professional handling of memory, conversation flow, and voice features whenever code changes are made.

---

## 2. Scope

**In scope:**
- Memory handling:
  - Correct saving/forgetting of facts.
  - Facts disclosed **only** when explicitly asked.
- Greeting & etiquette:
  - Neutral greetings without unsolicited memory recall.
- Command handling:
  - `reset` clears context but preserves persisted facts.
  - `goodbye` exits cleanly.
- Persistence:
  - Facts saved to `facts.json` and reload across sessions.
- Utility functions:
  - `is_greeting`, `asks_for_memory`, resampling, sentence splitting.
- Integration flow:
  - Single-turn conversations (input → STT → core logic → reply).
  - System prompt enforcement.

**Out of scope (for now):**
- Audio hardware reliability (mic/speaker drivers).
- Accuracy of external models (Whisper STT, Ollama LLM, Piper TTS).
- Performance/load testing.

---

## 3. Test Strategy

### 3.1 Levels of Testing
- **Unit tests**  
  Verify isolated functions (e.g., `parse_memory_command`, `forget_fact`, `is_greeting`).
- **Integration tests**  
  Simulate multi-turn flows, stubbing external models (Ollama).
- **Snapshot tests**  
  Compare assistant replies for known prompts against “golden” outputs.
- **Manual tests**  
  Quick voice checks (STT + TTS pipeline sanity).

### 3.2 Test Types
- **Functional** – confirm expected behavior.
- **Regression** – ensure fixes don’t reintroduce unsolicited memory.
- **Persistence** – verify memory survives restarts.
- **Negative** – ensure invalid/irrelevant inputs don’t trigger memory saves.

---

## 4. Test Environment

- **OS:** Windows 10/11 (local), Windows runners (GitHub Actions).  
- **Python:** 3.11.x (venv).  
- **Dependencies:** `pytest`, `pytest-cov`, `pytest-html`, `requests-mock`, `freezegun`.  
- **External services:** Ollama running locally (stubbed in integration tests).

---

## 5. Test Data

- Example facts:  
  - “My favorite food is pizza.”  
  - “I love the color black.”  
- Greeting inputs: “Hi”, “Hello”, “Good morning”.  
- Memory queries: “What do you remember?”, “What’s my favorite food?”  
- Non-commands: “Can you not always mention what you remember…”.

---

## 6. Entry / Exit Criteria

**Entry criteria:**
- Latest code committed to repo.
- All dependencies installed in virtual environment.
- Test data prepared.

**Exit criteria:**
- 100% of unit and integration tests pass.
- No critical or high-severity bugs open.
- Coverage of memory logic ≥ 80%.
- Known issues documented.

---

## 7. Deliverables

- Automated test results:
  - `reports/junit.xml` (machine-readable).
  - `reports/report.html` (human-readable).
- Coverage report (`pytest-cov`).
- Updated `docs/TEST_RESULTS.md` after each meaningful change.

---

## 8. Risks & Mitigation

- **LLM nondeterminism** – use stubbed responses or post-filtered outputs in tests.  
- **Audio flakiness** – skip in automated runs, check manually.  
- **Human error in fact logging** – enforce strict regex parsing for `remember/forget`.

---

## 9. Schedule

- **Unit + integration tests:** run locally before commit.  
- **Full suite:** run automatically on push/PR via GitHub Actions.  
- **Manual checks:** run before tagged releases.

---

## 10. Roles & Responsibilities

- **Developer (Kal)**: write/maintain tests, run locally.  
- **CI/CD pipeline**: run tests on every commit/PR.  
- **Reviewer (self or peers)**: verify `TEST_RESULTS.md` is updated with outcome.

---

## 11. References

- [Pytest Documentation](https://docs.pytest.org/)  
- [GitHub Actions for Python](https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-python)  
- Cyrus project codebase (`main.py`, `memory.py`).

---
