from pitaya.runner.parsing.codex_parser import CodexOutputParser


def test_codex_parser_handles_thread_and_turn_events() -> None:
    parser = CodexOutputParser()
    events = []
    lines = [
        '{"type":"thread.started","thread_id":"thread-123"}',
        '{"type":"turn.started"}',
        '{"type":"item.completed","item":{"id":"item_0","details":{"type":"reasoning","text":"Analyzing files"}}}',
        '{"type":"item.started","item":{"id":"item_1","details":{"type":"command_execution","command":"bash -lc ls","status":"in_progress"}}}',
        '{"type":"item.completed","item":{"id":"item_1","details":{"type":"command_execution","command":"bash -lc ls","status":"completed","exit_code":0,"aggregated_output":"README.md"}}}',
        '{"type":"item.completed","item":{"id":"item_2","details":{"type":"agent_message","text":"All good"}}}',
        '{"type":"turn.completed","usage":{"input_tokens":100,"cached_input_tokens":10,"output_tokens":5}}',
    ]

    for line in lines:
        evt = parser.parse_line(line)
        if evt:
            events.append(evt)

    summary = parser.get_summary()

    assert summary["session_id"] == "thread-123"
    assert summary["final_message"] == "All good"
    assert summary["metrics"] == {
        "input_tokens": 110,
        "output_tokens": 5,
        "total_tokens": 115,
    }

    assistant_events = [e for e in events if e["type"] == "assistant"]
    assert any("Analyzing files" in e["content"] for e in assistant_events)
    assert any(
        e.get("command") == "bash -lc ls" for e in events if e["type"] == "tool_result"
    )


def test_codex_parser_records_errors() -> None:
    parser = CodexOutputParser()
    parser.parse_line('{"type":"turn.failed","error":{"message":"quota exceeded"}}')
    summary = parser.get_summary()
    assert summary["error"] == "quota exceeded"


def test_codex_parser_flattened_items() -> None:
    parser = CodexOutputParser()
    parser.parse_line(
        '{"type":"item.completed","item":{"id":"item_0","type":"agent_message","text":"flat message"}}'
    )
    summary = parser.get_summary()
    assert summary["final_message"] == "flat message"
