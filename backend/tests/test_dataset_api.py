import io
import json


def _upload_sample_dataset(client, headers):
    lines = [
        {"instruction": f"Question {i}", "input": "", "output": f"Answer {i}"}
        for i in range(10)
    ]
    content = "\n".join(json.dumps(r) for r in lines).encode("utf-8")
    files = {"file": ("sample.jsonl", io.BytesIO(content), "application/octet-stream")}
    data = {"name": "sample-dataset", "description": "test dataset"}
    return client.post("/api/datasets", files=files, data=data, headers=headers)


def test_upload_dataset(client, auth_headers):
    res = _upload_sample_dataset(client, auth_headers)
    assert res.status_code == 201
    body = res.json()
    assert body["record_count"] == 10
    assert body["dataset_format"] == "instruction"


def test_list_datasets(client, auth_headers):
    _upload_sample_dataset(client, auth_headers)
    res = client.get("/api/datasets", headers=auth_headers)
    assert res.status_code == 200
    assert len(res.json()) == 1


def test_dataset_versions_created_on_upload(client, auth_headers):
    upload_res = _upload_sample_dataset(client, auth_headers)
    dataset_id = upload_res.json()["id"]
    res = client.get(f"/api/datasets/{dataset_id}/versions", headers=auth_headers)
    assert res.status_code == 200
    versions = res.json()
    assert len(versions) == 1
    assert "score" in versions[0]["quality_report"]


def test_reject_unsupported_extension(client, auth_headers):
    files = {"file": ("bad.txt", io.BytesIO(b"not a dataset"), "text/plain")}
    data = {"name": "bad-dataset"}
    res = client.post("/api/datasets", files=files, data=data, headers=auth_headers)
    assert res.status_code == 422
