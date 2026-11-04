"""Ratings tests use temporary saved fixtures only, never providers or credentials."""

import copy
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from generation import AppError
from ratings import RatingStore, majority, validate_answer
from server import make_handler
from test_generation import manager, plan


def fixture(tmp_path):
    app, _, _ = manager(tmp_path)
    run = app.generate(plan(app)["id"], {"max_image_calls": 3})
    data = app.data(run["id"])
    # Two cases make order persistence test meaningful without new generated images.
    data["camera_roll"]["photos"] *= 2
    data["manifest"]["photos"] = [copy.deepcopy(data["manifest"]["photos"][0]) for _ in range(2)]
    data["manifest"]["photos"][1]["photo_index"] = 1
    store = RatingStore(tmp_path / "ratings", lambda _: (data, lambda relative: app.image_file(run["id"], relative)))
    return store, data, app


def new_round(store):
    return store.create_round({"album_id": "maya"})["round"]


def start(store, round_id, name="Ethan", exposed=False):
    return store.start_session(round_id, {"reviewer": name, "prior_exposure": exposed})


def answer(item, identity="same", presence="present", setting="yes", action="yes"):
    return {"identity": identity, "people_count": item["expected_people_count"], "setting": setting, "action": action,
            "objects": {obj["id"]: {"presence": presence, "count": 1, "appearance": "matches", "state": "correct"} for obj in item["objects"]}}


def complete(store, round_id, name, identity="same", presence="present", setting="yes", action="yes"):
    bundle = start(store, round_id, name)
    for item in bundle["items"]:
        store.save_answer(bundle["session"]["id"], {"item_id": item["id"], "answer": answer(item, identity, presence, setting, action)})
    return store.submit(bundle["session"]["id"])


def test_store_is_empty_until_round_created_and_snapshot_is_frozen(tmp_path):
    store, data, app = fixture(tmp_path)
    assert store.overview()["rounds"] == [] and not store.root.exists()
    round_data = new_round(store)
    assert round_data["item_count"] == 2 and not round_data["results_available"]
    assert new_round(store)["id"] == round_data["id"]
    bundle = start(store, round_data["id"])
    image_url = bundle["items"][0]["image_url"]
    frozen = store.asset_file(round_data["id"], image_url.rsplit("/", 1)[1]).read_bytes()
    source = app.image_file(data["run"]["id"], data["manifest"]["photos"][0]["final_path"])
    source.write_bytes(b"changed source")
    assert store.asset_file(round_data["id"], image_url.rsplit("/", 1)[1]).read_bytes() == frozen
    assert new_round(store)["id"] != round_data["id"]


def test_shuffle_is_persistent_and_casefold_reviewer_resumes(tmp_path):
    store, _, _ = fixture(tmp_path)
    round_id = new_round(store)["id"]
    bundle = start(store, round_id, " Ethan ", True)
    resumed = start(store, round_id, "eTHan", False)
    assert resumed["session"]["id"] == bundle["session"]["id"]
    assert resumed["session"]["prior_exposure"] is True
    assert [i["id"] for i in resumed["items"]] == [i["id"] for i in bundle["items"]]
    assert store.overview()["rounds"][0]["draft_count"] == 1
    raw = json.dumps(bundle)
    assert "fake-image" not in raw and "fake-text" not in raw and "source_photo_index" not in raw
    assert "scores" not in bundle and "review" not in bundle and "source_path" not in raw
    assert all("/ratings/assets/" in i["image_url"] for i in bundle["items"])


def test_incomplete_draft_can_resume_but_cannot_submit(tmp_path):
    store, _, _ = fixture(tmp_path)
    bundle = start(store, new_round(store)["id"])
    session_id, item = bundle["session"]["id"], bundle["items"][0]
    saved = store.save_answer(session_id, {"item_id": item["id"], "answer": {"identity": "unclear", "notes": "small face"}})
    assert saved["session"]["saved_count"] == 1 and saved["session"]["complete_count"] == 0
    restarted = RatingStore(store.root, store.dataset_loader)
    assert restarted.session(session_id)["answers"][item["id"]]["notes"] == "small face"
    with pytest.raises(AppError, match="required"):
        store.submit(session_id)
    assert store.overview()["rounds"][0]["submitted_count"] == 0


def test_submitted_ratings_are_immutable_and_own_export_hides_others(tmp_path):
    store, _, _ = fixture(tmp_path)
    round_id = new_round(store)["id"]
    done = complete(store, round_id, "Ethan")
    session_id, item = done["session"]["id"], done["items"][0]
    assert done["session"]["status"] == "submitted" and done["session"]["complete_count"] == 2
    with pytest.raises(AppError, match="cannot be overwritten"):
        store.save_answer(session_id, {"item_id": item["id"], "answer": answer(item, "different")})
    other = start(store, round_id, "Private reviewer")
    assert "Private reviewer" not in json.dumps(store.export_session(session_id))
    assert store.submit(session_id)["session"]["status"] == "submitted"
    with pytest.raises(AppError):
        store.export_session(other["session"]["id"])
    with pytest.raises(AppError):
        store.results(round_id)
    with pytest.raises(AppError):
        store.export_round(round_id)


@pytest.mark.parametrize("patch", [
    {"identity": "yes"}, {"identity": []}, {"people_count": True}, {"people_count": -1}, {"people_count": 21},
    {"setting": "maybe"}, {"action": None}, {"unknown": True}, {"notes": "x" * 2001},
    {"objects": {"invented": {"presence": "present"}}},
])
def test_invalid_fields_are_rejected(tmp_path, patch):
    store, _, _ = fixture(tmp_path)
    bundle = start(store, new_round(store)["id"])
    item = bundle["items"][0]
    with pytest.raises(AppError):
        store.save_answer(bundle["session"]["id"], {"item_id": item["id"], "answer": answer(item) | patch})
    assert not store.session(bundle["session"]["id"])["answers"]


def test_object_absence_and_uncertainty_have_explicit_not_applicable_fields(tmp_path):
    store, _, _ = fixture(tmp_path)
    item = start(store, new_round(store)["id"])["items"][0]
    for presence, count in (("absent", 0), ("unclear", None)):
        result = validate_answer(answer(item, presence=presence), item, True)
        rating = next(iter(result["objects"].values()))
        assert rating == {"presence": presence, "count": count, "appearance": "not_applicable", "state": "not_applicable"}
    bad = answer(item)
    next(iter(bad["objects"].values()))["count"] = 0
    with pytest.raises(AppError):
        validate_answer(bad, item, True)


def test_results_use_strict_majorities_full_denominators_and_conjunction(tmp_path):
    store, _, _ = fixture(tmp_path)
    round_id = new_round(store)["id"]
    complete(store, round_id, "A", "same", "present", "yes", "no")
    complete(store, round_id, "B", "different", "absent", "no", "yes")
    complete(store, round_id, "C", "unclear", "unclear", "yes", "yes")
    result = store.results(round_id)
    assert result["metrics"]["identity_match"] == {"numerator": 0, "denominator": 2}
    assert result["metrics"]["identity_unresolved"] == {"numerator": 2, "denominator": 2}
    assert result["metrics"]["object_visibility"] == {"numerator": 0, "denominator": 2}
    assert all(i["setting"] == "yes" and i["action"] == "yes" and i["scene_aligned"] is False for i in result["items"])
    assert result["metrics"]["scene_alignment"]["numerator"] == 0
    assert "No pipeline improvement" in result["limitations"]
    assert len(store.export_round(round_id)["sessions"]) == 3
    assert majority(["same", "same", "different", "unclear"]) == "unresolved"



def test_unknown_object_visibility_does_not_become_confirmed_no_duplication(tmp_path):
    store, _, _ = fixture(tmp_path)
    round_id = new_round(store)["id"]
    complete(store, round_id, "A", presence="unclear")
    complete(store, round_id, "B", presence="unclear")
    complete(store, round_id, "C", presence="present")
    result = store.results(round_id)
    assert all(obj["duplicate"] == "unclear" for item in result["items"] for obj in item["objects"])
    assert result["metrics"]["object_duplication"] == {"numerator": 0, "denominator": 2}


def test_missing_and_unclear_faces_are_not_dropped(tmp_path):
    store, _, _ = fixture(tmp_path)
    round_id = new_round(store)["id"]
    for name in ("A", "B", "C"):
        complete(store, round_id, name, "not_visible", "absent")
    result = store.results(round_id)
    assert result["metrics"]["identity_match"] == {"numerator": 0, "denominator": 2}
    assert result["metrics"]["identity_not_visible"] == {"numerator": 2, "denominator": 2}
    assert result["metrics"]["object_visibility"] == {"numerator": 0, "denominator": 2}


def test_protagonist_not_requested_is_not_an_identity_failure(tmp_path):
    store, data, _ = fixture(tmp_path)
    data["camera_roll"]["photos"][0] = {**data["camera_roll"]["photos"][0], "people": []}
    round_id = new_round(store)["id"]
    for name in ("A", "B", "C"):
        complete(store, round_id, name)
    result = store.results(round_id)
    assert result["metrics"]["identity_match"] == {"numerator": 1, "denominator": 1}
    assert result["metrics"]["scene_alignment"]["denominator"] == 2


def test_duplicate_concurrent_starts_create_one_reviewer(tmp_path):
    store, _, _ = fixture(tmp_path)
    round_id = new_round(store)["id"]
    with ThreadPoolExecutor(max_workers=5) as pool:
        sessions = list(pool.map(lambda _: start(store, round_id), range(5)))
    assert len({s["session"]["id"] for s in sessions}) == 1
    assert store.overview()["rounds"][0]["draft_count"] == 1


def request(store, app, path, body=None, headers=None):
    class Connection:
        def __init__(self, raw): self.input, self.output = io.BytesIO(raw), io.BytesIO()
        def makefile(self, *args, **kwargs): return self.input
        def sendall(self, content): self.output.write(content)
    method, content = ("GET", b"") if body is None else ("POST", json.dumps(body).encode())
    supplied = {"Host": "127.0.0.1:5812", "Content-Length": str(len(content)), "Content-Type": "application/json", "X-Demo-Token": "test-token", **(headers or {})}
    raw = f"{method} {path} HTTP/1.0\r\n" + "".join(f"{key}: {value}\r\n" for key, value in supplied.items()) + "\r\n"
    connection = Connection(raw.encode() + content)
    make_handler(app, "test-token", rating_store=store)(connection, ("127.0.0.1", 1234), SimpleNamespace(server_address=("127.0.0.1", 5812)))
    head, content = connection.output.getvalue().split(b"\r\n\r\n", 1)
    return int(head.split()[1]), content


def test_http_rating_flow_and_no_provider_calls(tmp_path):
    store, _, app = fixture(tmp_path)
    assert json.loads(request(store, app, "/api/ratings")[1])["rounds"] == []
    status, raw = request(store, app, "/api/ratings/rounds", {"album_id": "maya"})
    round_id = json.loads(raw)["round"]["id"]
    assert status == 201
    status, raw = request(store, app, f"/api/ratings/rounds/{round_id}/sessions", {"reviewer": "A", "prior_exposure": False})
    bundle = json.loads(raw)
    assert status == 201
    assert request(store, app, bundle["items"][0]["image_url"])[0] == 200
    assert request(store, app, f"/api/ratings/rounds/{round_id}/results")[0] == 409
    assert request(store, app, f"/api/ratings/sessions/{bundle['session']['id']}")[0] == 200
    for item in bundle["items"]:
        assert request(store, app, f"/api/ratings/sessions/{bundle['session']['id']}/answer", {"item_id": item["id"], "answer": answer(item)})[0] == 200
    assert request(store, app, f"/api/ratings/sessions/{bundle['session']['id']}/submit", {})[0] == 200
    assert request(store, app, f"/api/ratings/sessions/{bundle['session']['id']}/export")[0] == 200


@pytest.mark.parametrize("headers", [{"Origin": "https://evil.example"}, {"Host": "evil.example:5812"}, {"X-Demo-Token": "wrong"}])
def test_rating_writes_keep_server_origin_and_token_guards(tmp_path, headers):
    store, _, app = fixture(tmp_path)
    assert request(store, app, "/api/ratings/rounds", {"album_id": "maya"}, headers)[0] == 403
    assert not store.overview()["rounds"]


@pytest.mark.parametrize("path", ["/api/ratings/rounds/../../private", "/ratings/assets/../credentials.env", "/ratings/assets/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/../round.json", "/ratings/assets/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/%2e%2e%2fround.json"])
def test_rating_paths_reject_traversal(tmp_path, path):
    store, _, app = fixture(tmp_path)
    assert request(store, app, path)[0] == 404
