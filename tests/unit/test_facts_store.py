import memory

def test_add_save_load_forget_roundtrip():
    facts = memory.load_facts()

    # add and save
    assert memory.add_fact(facts, "pizza is great")
    memory.save_facts(facts)

    # reload and verify
    reloaded = memory.load_facts()
    assert any("pizza is great" in f["text"] for f in reloaded["facts"])

    # forget and save
    assert memory.forget_fact(reloaded, "pizza is great")
    memory.save_facts(reloaded)

    # reload and verify removal
    reloaded2 = memory.load_facts()
    assert not any("pizza is great" in f["text"] for f in reloaded2["facts"])

def test_summarize_facts_lists_newest_first():
    facts = memory.load_facts()
    memory.add_fact(facts, "A")
    memory.add_fact(facts, "B")  # newer
    summary = memory.summarize_facts(facts, max_items=10)

    # Expect a "Recent facts:" line with B before A
    assert "Recent facts:" in summary
    assert summary.index("B") < summary.index("A")
