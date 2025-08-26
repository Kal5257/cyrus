import memory

def test_parse_memory_command_anchored():
    # should trigger only on commands at start
    assert memory.parse_memory_command("remember pizza is my favorite food") == ("remember", "pizza is my favorite food")
    assert memory.parse_memory_command("please remember: drink water") == ("remember", "drink water")
    assert memory.parse_memory_command("forget pizza is my favorite food") == ("forget", "pizza is my favorite food")

    # should NOT trigger on mid-sentence mentions
    assert memory.parse_memory_command("can you remember this later?") is None
    assert memory.parse_memory_command("I want you to forget when I said that") is None
    assert memory.parse_memory_command("remember?") is None