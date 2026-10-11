import json

from app.ml.datasets.loader import (
    detect_dataset_format,
    format_as_prompt,
    load_and_normalize,
    normalize_record,
)
from app.ml.datasets.validation import (
    compute_quality_report,
    compute_statistics,
    split_records,
    validate_records,
)
from app.models.enums import DatasetFormat


def test_detect_instruction_format():
    records = [{"instruction": "Do X", "output": "Did X"}]
    assert detect_dataset_format(records) == DatasetFormat.INSTRUCTION


def test_detect_conversational_format():
    records = [{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}]
    assert detect_dataset_format(records) == DatasetFormat.CONVERSATIONAL


def test_normalize_instruction_record():
    record = normalize_record({"instruction": "Explain X", "input": "", "output": "X is..."}, DatasetFormat.INSTRUCTION)
    assert record == {"instruction": "Explain X", "input": "", "output": "X is..."}


def test_normalize_drops_incomplete_record():
    record = normalize_record({"instruction": "", "output": "answer"}, DatasetFormat.INSTRUCTION)
    assert record is None


def test_normalize_conversational_record():
    raw = {"messages": [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4"}]}
    record = normalize_record(raw, DatasetFormat.CONVERSATIONAL)
    assert record["instruction"] == "What is 2+2?"
    assert record["output"] == "4"


def test_load_and_normalize_jsonl(tmp_path):
    path = tmp_path / "data.jsonl"
    lines = [
        {"instruction": "Q1", "output": "A1"},
        {"instruction": "Q2", "output": "A2"},
        {"instruction": "", "output": "A3"},  # will be dropped
    ]
    with open(path, "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")

    fmt, normalized, dropped = load_and_normalize(str(path), "jsonl")
    assert fmt == DatasetFormat.INSTRUCTION
    assert len(normalized) == 2
    assert dropped == 1


def test_format_as_prompt_includes_sections():
    prompt = format_as_prompt({"instruction": "Explain X", "input": "", "output": "X is Y."})
    assert "### Instruction:" in prompt
    assert "### Response:" in prompt
    assert "X is Y." in prompt


def test_validate_records_flags_empty_dataset():
    errors = validate_records([])
    assert len(errors) > 0


def test_split_records_covers_all_records():
    records = [{"instruction": f"q{i}", "input": "", "output": f"a{i}"} for i in range(20)]
    train, val, test = split_records(records, seed=1)
    assert len(train) + len(val) + len(test) == 20
    assert len(train) > 0


def test_json_wrapped_in_examples_key_is_detected(tmp_path):
    """Regression test: exports that wrap records under keys other than
    'data' (e.g. 'examples', 'records', 'items') must still be detected,
    not fail with 'Could not detect dataset format'."""
    path = tmp_path / "data.json"
    payload = {"examples": [{"instruction": "Q1", "output": "A1"}, {"instruction": "Q2", "output": "A2"}]}
    path.write_text(json.dumps(payload))

    fmt, normalized, dropped = load_and_normalize(str(path), "json")
    assert fmt == DatasetFormat.INSTRUCTION
    assert len(normalized) == 2
    assert dropped == 0


def test_single_unwrapped_record_object_is_detected(tmp_path):
    """A JSON file containing exactly one record object at the top level
    (not wrapped in a list) should be treated as a one-record dataset."""
    path = tmp_path / "data.json"
    path.write_text(json.dumps({"instruction": "Q1", "output": "A1"}))

    fmt, normalized, dropped = load_and_normalize(str(path), "json")
    assert fmt == DatasetFormat.INSTRUCTION
    assert len(normalized) == 1


def test_non_object_json_list_raises_clean_error(tmp_path):
    """Regression test: a JSON file that is just a list of raw values (not
    objects) must raise a clear ValueError, not crash with a raw
    'argument of type int is not iterable' TypeError."""
    path = tmp_path / "data.json"
    path.write_text(json.dumps([1, 2, 3]))

    try:
        load_and_normalize(str(path), "json")
        assert False, "expected a ValueError"
    except ValueError as e:
        assert "Could not detect dataset format" in str(e)
    except TypeError:
        assert False, "raw TypeError leaked instead of a clean ValueError"


def test_jsonl_with_malformed_line_drops_it_instead_of_crashing(tmp_path):
    """Regression test: one malformed/non-object line in an otherwise valid
    JSONL file should be dropped and counted, not crash the whole upload."""
    path = tmp_path / "data.jsonl"
    lines = [
        json.dumps({"instruction": "Q1", "output": "A1"}),
        "42",  # malformed row: valid JSON, but not an object
        json.dumps({"instruction": "Q2", "output": "A2"}),
    ]
    path.write_text("\n".join(lines))

    fmt, normalized, dropped = load_and_normalize(str(path), "jsonl")
    assert fmt == DatasetFormat.INSTRUCTION
    assert len(normalized) == 2
    assert dropped == 1


def test_conversational_with_malformed_message_entry_is_handled(tmp_path):
    """A malformed entry inside 'messages' (not an object) should not crash
    normalization - the record is dropped if it can't be normalized."""
    path = tmp_path / "data.jsonl"
    payload = {
        "messages": [
            {"role": "user", "content": "hi"},
            "garbage-entry",
            {"role": "assistant", "content": "hello"},
        ]
    }
    path.write_text(json.dumps(payload))

    fmt, normalized, dropped = load_and_normalize(str(path), "jsonl")
    assert fmt == DatasetFormat.CONVERSATIONAL
    assert len(normalized) == 1


def test_quality_report_detects_duplicates():
    records = [{"instruction": "same", "input": "", "output": "same answer"}] * 5
    stats = compute_statistics(records)
    quality = compute_quality_report(records, stats, dropped_records=0)
    assert stats["duplicates"] == 4
    assert quality["score"] < 100
    assert any("duplicate" in w for w in quality["warnings"])