"""Frozen local image rounds and independent human judgments. No model calls."""

import hashlib
import json
import secrets
import threading
from pathlib import Path
from uuid import uuid4

from generation import AppError, HERE, ID_PATTERN, now, read_json, write_json

RUBRIC_VERSION = "human-v1"
MIN_REVIEWERS = 3
RUBRIC = {
    "version": RUBRIC_VERSION,
    "identity": "Compare the protagonist's face with the reference: same, different, unclear, or not_visible. Clothing, role and props alone do not establish identity.",
    "people_count": "Count visible people. Use null when the count cannot be judged confidently.",
    "objects": "For each required recurring object, judge presence, visible count, reference appearance, and requested placement/state. Absent or unclear presence makes the other judgments not applicable.",
    "scene": "Rate setting and main action separately: yes, partial, no, or unclear.",
    "majority": "A category requires strictly more than half of all submitted reviewers. Unclear, missing and unresolved judgments remain in denominators.",
    "independence": "Rate without viewing model scores, existing annotations or another reviewer's answers. Prior exposure is recorded; this is a development pilot, not a held-out study.",
}


def default_loader(manager, static_root):
    """Read only public saved Maya data or a completed local generation."""
    static_root = Path(static_root)
    def load(album_id):
        if album_id == "maya":
            source = (static_root / "data.js").read_text()
            data = json.loads(source.split("=", 1)[1].strip().removesuffix(";"))
            base = (static_root / "assets/images").resolve()
            def image_file(relative):
                path = (base / relative).resolve()
                if not path.is_relative_to(base) or not path.is_file():
                    raise AppError("A saved source image is unavailable.", "source_missing", 409)
                return path
            return data, image_file
        if manager.get(album_id)["status"] != "completed":
            raise AppError("Finish the album before starting a rating round.", "album_incomplete", 409)
        return manager.data(album_id), lambda relative: manager.image_file(album_id, relative)
    return load


def require_id(identifier):
    if not isinstance(identifier, str) or not ID_PATTERN.fullmatch(identifier):
        raise AppError("Rating record not found.", "not_found", 404)
    return identifier


def majority(values):
    for value in set(values):
        if values.count(value) > len(values) / 2:
            return value
    return "unresolved"


def validate_answer(answer, item, complete=False):
    """Drafts may be incomplete; submitted judgments must be explicit and valid."""
    if not isinstance(answer, dict) or set(answer) - {"identity", "people_count", "setting", "action", "objects", "notes"}:
        raise AppError("The rating contains unsupported fields.")
    clean = dict(answer)
    if not item["protagonist_required"]:
        clean["identity"] = "not_applicable"
    choices = {"identity": {"same", "different", "unclear", "not_visible"} if item["protagonist_required"] else {"not_applicable"},
               "setting": {"yes", "partial", "no", "unclear"}, "action": {"yes", "partial", "no", "unclear"}}
    for field, allowed in choices.items():
        if field in clean and (not isinstance(clean[field], str) or clean[field] not in allowed):
            raise AppError(f"Choose a valid {field} rating.")
        if complete and field not in clean:
            raise AppError("Finish every required rating before submitting.", "incomplete", 409)
    if "people_count" in clean and clean["people_count"] is not None and (type(clean["people_count"]) is not int or not 0 <= clean["people_count"] <= 20):
        raise AppError("Use a whole-number people count from zero to twenty, or unclear.")
    if complete and "people_count" not in clean:
        raise AppError("Rate the visible people count.", "incomplete", 409)
    if "notes" in clean and (not isinstance(clean["notes"], str) or len(clean["notes"]) > 2000):
        raise AppError("Keep notes within 2,000 characters.")
    objects = clean.get("objects", {})
    expected = {obj["id"] for obj in item["objects"]}
    if not isinstance(objects, dict) or set(objects) - expected:
        raise AppError("Rate only the required objects in this photograph.")
    if complete and set(objects) != expected:
        raise AppError("Rate every required object.", "incomplete", 409)
    clean["objects"] = {}
    for identifier, rating in objects.items():
        if not isinstance(rating, dict) or set(rating) - {"presence", "count", "appearance", "state"}:
            raise AppError("The object rating contains unsupported fields.")
        rating = dict(rating)
        allowed = {"presence": {"present", "absent", "unclear"},
                   "appearance": {"matches", "different", "unclear", "not_applicable"},
                   "state": {"correct", "partial", "wrong", "unclear", "not_applicable"}}
        for field, values in allowed.items():
            if field in rating and (not isinstance(rating[field], str) or rating[field] not in values):
                raise AppError(f"Choose a valid object {field} rating.")
        if "count" in rating and rating["count"] is not None and (type(rating["count"]) is not int or not 0 <= rating["count"] <= 20):
            raise AppError("Use a whole-number object count from zero to twenty, or unclear.")
        presence = rating.get("presence")
        if presence in ("absent", "unclear"):
            rating.update(count=0 if presence == "absent" else None, appearance="not_applicable", state="not_applicable")
        if complete:
            if presence is None or set(rating) != {"presence", "count", "appearance", "state"}:
                raise AppError("Finish every required object rating.", "incomplete", 409)
            if presence == "present" and (rating["count"] is None or rating["count"] < 1 or rating["appearance"] == "not_applicable" or rating["state"] == "not_applicable"):
                raise AppError("A visible object needs a positive count, appearance and state judgment.", "incomplete", 409)
        clean["objects"][identifier] = rating
    return clean


class RatingStore:
    def __init__(self, root=None, dataset_loader=None):
        self.root = Path(root) if root is not None else HERE / "private/ratings"
        self.dataset_loader = dataset_loader
        self.lock = threading.RLock()

    def _round(self, identifier):
        path = self.root / "rounds" / require_id(identifier) / "round.json"
        if not path.is_file():
            raise AppError("Rating round not found.", "not_found", 404)
        return read_json(path)

    def _session(self, identifier):
        path = self.root / "sessions" / (require_id(identifier) + ".json")
        if not path.is_file():
            raise AppError("Rating session not found.", "not_found", 404)
        return read_json(path)

    def _sessions(self, round_id):
        return [s for path in (self.root / "sessions").glob("*.json") if (s := read_json(path))["round_id"] == round_id]

    def _save_session(self, session):
        session["updated_at"] = now()
        write_json(self.root / "sessions" / (session["id"] + ".json"), session)

    def _meta(self, round_data):
        sessions = self._sessions(round_data["id"])
        submitted = sum(s["status"] == "submitted" for s in sessions)
        return {key: round_data[key] for key in ("id", "label", "album_id", "created_at", "rubric_version")} | {
            "item_count": len(round_data["items"]), "submitted_count": submitted,
            "draft_count": len(sessions) - submitted, "results_available": submitted >= MIN_REVIEWERS,
            "min_reviewers": MIN_REVIEWERS, "purpose": "development", "iteration_status": "measurement_in_progress"}

    def overview(self):
        with self.lock:
            rounds = [self._meta(read_json(path)) for path in (self.root / "rounds").glob("*/round.json")]
            return {"rounds": sorted(rounds, key=lambda r: r["created_at"], reverse=True), "rubric_version": RUBRIC_VERSION}

    def round(self, identifier):
        with self.lock:
            return {"round": self._meta(self._round(identifier))}

    def create_round(self, payload):
        album_id = payload.get("album_id")
        if album_id != "maya":
            require_id(album_id)
        label = payload.get("label")
        if label is not None and (not isinstance(label, str) or not 1 <= len(label.strip()) <= 100):
            raise AppError("Use a round label between one and 100 characters.")
        with self.lock:
            data, image_file = self.dataset_loader(album_id)
            roll, manifest = data["camera_roll"], data["manifest"]
            if not manifest["photos"] or any(not all(p["completed"] for p in photo["passes"]) for photo in manifest["photos"]):
                raise AppError("All final photographs must be saved before rating.", "album_incomplete", 409)
            entities = {v["id"]: v for v in roll["people"] + roll["objects"]}
            refs = {v["entity_id"]: v for v in manifest["references"]}
            protagonist = entities[roll["protagonist_id"]]
            assets, provenance, items = {}, [], []
            def asset(relative):
                content = image_file(relative).read_bytes()
                digest = hashlib.sha256(content).hexdigest()
                assets[digest] = content
                provenance.append({"source_path": relative, "sha256": digest})
                return digest
            for photo in manifest["photos"]:
                scene = roll["photos"][photo["photo_index"]]
                items.append({"image": asset(photo["final_path"]), "scene_request": scene["description"],
                    "protagonist": {"name": protagonist["name"], "description": protagonist["visual_description"], "image": asset(refs[protagonist["id"]]["image"]["path"])},
                    "expected_people_count": len(scene["people"]), "protagonist_required": any(p["person_id"] == roll["protagonist_id"] for p in scene["people"]),
                    "objects": [{"id": o["object_id"], "name": entities[o["object_id"]]["name"],
                                 "description": entities[o["object_id"]]["visual_description"], "expected_state": o["appearance"],
                                 "image": asset(refs[o["object_id"]]["image"]["path"])} for o in scene["objects"]],
                    "source_photo_index": photo["photo_index"]})
            fingerprint = hashlib.sha256(json.dumps({"items": items, "rubric": RUBRIC_VERSION}, sort_keys=True).encode()).hexdigest()
            for path in (self.root / "rounds").glob("*/round.json"):
                existing = read_json(path)
                if existing["album_id"] == album_id and existing["fingerprint"] == fingerprint:
                    return {"round": self._meta(existing)}
            identifier = uuid4().hex
            folder = self.root / "rounds" / identifier
            (folder / "assets").mkdir(parents=True)
            folder.chmod(0o700)
            names = {}
            for digest, content in assets.items():
                name = uuid4().hex + ".png"
                target = folder / "assets" / name
                target.write_bytes(content)
                target.chmod(0o600)
                names[digest] = name
            for item in items:
                item["id"] = uuid4().hex
                item["image"] = names[item["image"]]
                item["protagonist"]["image"] = names[item["protagonist"]["image"]]
                for obj in item["objects"]:
                    obj["image"] = names[obj["image"]]
            round_data = {"id": identifier, "album_id": album_id, "label": label.strip() if label else f"{protagonist['name']} · Final photographs",
                          "created_at": now(), "rubric_version": RUBRIC_VERSION, "fingerprint": fingerprint,
                          "items": items, "assets": names, "provenance": provenance,
                          "source": {"generation_settings": manifest.get("settings", {}), "source_commit": data.get("provenance", {}).get("source_commit"), "brief": data["story"]["seed"]}}
            write_json(folder / "round.json", round_data)
            return {"round": self._meta(round_data)}

    def _public_item(self, item, round_id):
        url = lambda name: f"/ratings/assets/{round_id}/{name}"
        return {"id": item["id"], "image_url": url(item["image"]), "scene_request": item["scene_request"],
                "expected_people_count": item["expected_people_count"], "protagonist_required": item["protagonist_required"],
                "protagonist": {"name": item["protagonist"]["name"], "description": item["protagonist"]["description"], "image_url": url(item["protagonist"]["image"])},
                "objects": [{k: v for k, v in obj.items() if k != "image"} | {"image_url": url(obj["image"])} for obj in item["objects"]]}

    def _bundle(self, session):
        round_data = self._round(session["round_id"])
        items = {item["id"]: item for item in round_data["items"]}
        complete_count = 0
        for identifier, answer in session["answers"].items():
            try:
                validate_answer(answer, items[identifier], complete=True)
                complete_count += 1
            except AppError:
                pass
        return {"session": {k: session[k] for k in ("id", "round_id", "reviewer", "prior_exposure", "status", "updated_at")} | {
                    "saved_count": len(session["answers"]), "complete_count": complete_count, "item_count": len(items)},
                "round": self._meta(round_data), "items": [self._public_item(items[i], round_data["id"]) for i in session["order"]],
                "answers": session["answers"], "rubric": RUBRIC}

    def session(self, identifier):
        with self.lock:
            return self._bundle(self._session(identifier))

    def start_session(self, round_id, payload):
        reviewer = payload.get("reviewer")
        if not isinstance(reviewer, str) or not 1 <= len(reviewer.strip()) <= 60 or any(ord(c) < 32 for c in reviewer):
            raise AppError("Enter a reviewer name or initials, up to 60 characters.")
        if type(payload.get("prior_exposure")) is not bool:
            raise AppError("State whether you have seen these images or their scores before.")
        reviewer = " ".join(reviewer.split())
        with self.lock:
            round_data = self._round(round_id)
            for existing in self._sessions(round_id):
                if existing["reviewer"].casefold() == reviewer.casefold():
                    return self._bundle(existing)
            order = [item["id"] for item in round_data["items"]]
            secrets.SystemRandom().shuffle(order)
            session = {"id": uuid4().hex, "round_id": round_id, "reviewer": reviewer,
                       "prior_exposure": payload["prior_exposure"], "status": "draft", "order": order, "answers": {}, "created_at": now()}
            self._save_session(session)
            return self._bundle(session)

    def save_answer(self, identifier, payload):
        with self.lock:
            session = self._session(identifier)
            if session["status"] != "draft":
                raise AppError("Submitted ratings are preserved and cannot be overwritten.", "submitted", 409)
            round_data = self._round(session["round_id"])
            item = next((i for i in round_data["items"] if i["id"] == payload.get("item_id")), None)
            if item is None:
                raise AppError("This photograph is not in this rating session.")
            session["answers"][item["id"]] = validate_answer(payload.get("answer"), item)
            self._save_session(session)
            return self._bundle(session)

    def submit(self, identifier):
        with self.lock:
            session = self._session(identifier)
            if session["status"] == "submitted":
                return self._bundle(session)
            round_data = self._round(session["round_id"])
            for item in round_data["items"]:
                validate_answer(session["answers"].get(item["id"], {}), item, complete=True)
            session.update(status="submitted", submitted_at=now())
            self._save_session(session)
            return self._bundle(session)

    def asset_file(self, round_id, filename):
        with self.lock:
            round_data = self._round(round_id)
            if filename not in round_data["assets"].values():
                raise AppError("Rating image not found.", "not_found", 404)
            return self.root / "rounds" / round_id / "assets" / filename

    def results(self, round_id):
        with self.lock:
            round_data = self._round(round_id)
            sessions = [s for s in self._sessions(round_id) if s["status"] == "submitted"]
            if len(sessions) < MIN_REVIEWERS:
                raise AppError("Team results unlock after three reviewers submit their own ratings. This avoids showing early votes to later reviewers.", "ratings_pending", 409)
            items = []
            for item in round_data["items"]:
                votes = [s["answers"][item["id"]] for s in sessions]
                counts = lambda values: {str(value): values.count(value) for value in set(values)}
                objects = []
                for obj in item["objects"]:
                    ratings = [vote["objects"][obj["id"]] for vote in votes]
                    objects.append({"id": obj["id"], "name": obj["name"],
                                    "presence": majority([r["presence"] for r in ratings]),
                                    "duplicate": majority(["unclear" if r["presence"] == "unclear" else r["presence"] == "present" and r["count"] > 1 for r in ratings]),
                                    "appearance": majority([r["appearance"] for r in ratings]),
                                    "state": majority([r["state"] for r in ratings]),
                                    "presence_votes": counts([r["presence"] for r in ratings])})
                items.append({"id": item["id"], "image_url": self._public_item(item, round_id)["image_url"],
                              "protagonist_required": item["protagonist_required"], "identity": majority([v["identity"] for v in votes]), "identity_votes": counts([v["identity"] for v in votes]),
                              "people_count": majority([v["people_count"] for v in votes]), "expected_people_count": item["expected_people_count"],
                              "setting": majority([v["setting"] for v in votes]), "action": majority([v["action"] for v in votes]),
                              "scene_aligned": majority([v["setting"] == "yes" and v["action"] == "yes" for v in votes]), "objects": objects})
            objects = [obj for item in items for obj in item["objects"]]
            identity_items = [item for item in items if item["protagonist_required"]]
            metric = lambda numerator, denominator: {"numerator": numerator, "denominator": denominator}
            metrics = {
                "identity_match": metric(sum(i["identity"] == "same" for i in identity_items), len(identity_items)),
                "identity_unclear": metric(sum(i["identity"] == "unclear" for i in identity_items), len(identity_items)),
                "identity_unresolved": metric(sum(i["identity"] == "unresolved" for i in identity_items), len(identity_items)),
                "identity_not_visible": metric(sum(i["identity"] == "not_visible" for i in identity_items), len(identity_items)),
                "object_visibility": metric(sum(o["presence"] == "present" for o in objects), len(objects)),
                "object_duplication": metric(sum(o["duplicate"] is True for o in objects), len(objects)),
                "object_state_wrong": metric(sum(o["state"] == "wrong" for o in objects), len(objects)),
                "scene_alignment": metric(sum(i["scene_aligned"] is True for i in items), len(items)),
            }
            return {"round": self._meta(round_data), "reviewer_count": len(sessions), "prior_exposure_count": sum(s["prior_exposure"] for s in sessions),
                    "metrics": metrics, "items": items, "rubric": RUBRIC,
                    "limitations": "Development-round human judgments, not a held-out benchmark. Strict majorities use every submitted reviewer; all protagonist-required scenes remain in the identity denominator, all scenes in alignment, and all required objects in object metrics. No pipeline improvement is established by this round alone."}

    def export_round(self, round_id):
        with self.lock:
            results = self.results(round_id)
            return {"results": results, "snapshot": self._round(round_id), "rubric": RUBRIC,
                    "sessions": [s for s in self._sessions(round_id) if s["status"] == "submitted"]}

    def export_session(self, identifier):
        with self.lock:
            session = self._session(identifier)
            if session["status"] != "submitted":
                raise AppError("Submit your completed ratings before exporting them.", "incomplete", 409)
            return self._bundle(session)
