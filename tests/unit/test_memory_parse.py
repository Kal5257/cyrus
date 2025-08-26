import memory

def test_parse_memory_command_anchored_only():
    # Positive: anchored commands (with/without "please", with ":" optional)
    assert memory.parse_memory_command("remember pizza is my favorite food") == ("remember", "pizza is my favorite food")
    assert memory.parse_memory_command("please remember: drink water") == ("remember", "drink water")
    assert memory.parse_memory_command("forget pizza is my favorite food") == ("forget", "pizza is my favorite food")
    assert memory.parse_memory_command("forget: call mom on Sunday") == ("forget", "call mom on Sunday")

    # Negative: should NOT trigger mid-sentence mentions or questions
    assert memory.parse_memory_command("can you remember this later?") is None
    assert memory.parse_memory_command("I want you to forget when I said that") is None
    assert memory.parse_memory_command("remember?") is None