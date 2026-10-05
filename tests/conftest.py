import json

import pytest

from ai_camera_roll import LLM


class FakeLLM(LLM):
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def _complete(self, *, system, prompt):
        self.calls.append({"system": system, "prompt": prompt})
        response = next(self.responses)
        return response if isinstance(response, str) else json.dumps(response)


@pytest.fixture
def fake_llm():
    return FakeLLM


@pytest.fixture
def story_payload():
    def make(years=5):
        return {
            "background": "Maya, 24, lives in Philadelphia and cycles in her green cap.",
            "years": [
                {"year": year, "narrative": f"In year {year}, Maya rides with Eli."}
                for year in range(1, years + 1)
            ],
        }

    return make


@pytest.fixture
def roll_payload():
    def make(years=5, photos_per_year=10):
        return {
            "protagonist_id": "person_maya",
            "people": [
                {
                    "id": "person_maya",
                    "name": "Maya",
                    "relationship": "self",
                    "age_at_start": 24,
                    "visual_description": "Short curly brown hair and round glasses.",
                }
            ],
            "objects": [
                {
                    "id": "object_cap",
                    "name": "Green cap",
                    "visual_description": "Faded green canvas with a mountain patch.",
                    "significance": "A gift from Eli, worn on weekend bike rides.",
                }
            ],
            "photos": [
                {
                    "year": year,
                    "month": 1 + index * 12 // photos_per_year,
                    "description": "Maya's cap rests on a bench beside a river path.",
                    "story_context": "A quiet stop on her regular weekend ride.",
                    "people": [],
                    "objects": [
                        {"object_id": "object_cap", "appearance": "Lying on the bench."}
                    ],
                }
                for year in range(1, years + 1)
                for index in range(photos_per_year)
            ],
        }

    return make
