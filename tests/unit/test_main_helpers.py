import json
import types
import main

def test_split_sentences_basic():
    text = "Hello. How are you? I'm fine!"
    parts = main._split_sentences(text)
    assert parts == ["Hello.", "How are you?", "I'm fine!"]

def test_flatten_messages_for_prompt():
    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hey"},
        {"role": "user", "content": "Bye"},
    ]
    out = main._flatten_messages_for_prompt(messages)
    # Must preserve roles and end with "ASSISTANT:" prompt
    assert out.splitlines()[-1] == "ASSISTANT:"
    assert "SYSTEM: SYS" in out
    assert "USER: Hi" in out
    assert "ASSISTANT: Hey" in out
    assert "USER: Bye" in out

def test_parse_streaming_collects_chunks(monkeypatch):
    # Fake streaming response object with iter_lines()
    chunks = [
        json.dumps({"message": {"content": "Hello "}}),
        json.dumps({"message": {"content": "Kal!"}}),
        json.dumps({"done": True})
    ]
    resp = types.SimpleNamespace(iter_lines=lambda decode_unicode: chunks)
    combined = main._parse_streaming(resp)
    assert combined == "Hello Kal!"

def test_ask_ollama_with_history_uses_chat_nonstream(monkeypatch):
    # Return 200 with chat JSON
    class Resp:
        status_code = 200
        def json(self):
            return {"message": {"content": "chat-ok"}}

    def fake_post(url, json=None, timeout=None, stream=False):
        return Resp()

    monkeypatch.setattr(main.requests, "post", fake_post)
    result = main.ask_ollama_with_history([{"role": "user", "content": "Hi"}], model="anything")
    assert result == "chat-ok"

def test_ask_ollama_with_history_fallback_generate(monkeypatch):
    # First call (chat) raises exception -> fallback to /api/generate
    class RespGen:
        status_code = 200
        def json(self):
            return {"response": "gen-ok"}

    # Switch behavior based on URL
    def fake_post(url, json=None, timeout=None, stream=False):
        if url.endswith("/api/chat"):
            raise main.requests.RequestException("fail chat")
        if url.endswith("/api/generate"):
            return RespGen()
        raise AssertionError("Unexpected URL " + url)

    monkeypatch.setattr(main.requests, "post", fake_post)
    result = main.ask_ollama_with_history([{"role": "user", "content": "Hi"}], model="anything")
    assert result == "gen-ok"
