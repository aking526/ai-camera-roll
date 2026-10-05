"""The story, shared identities, and scene-specific appearances."""

from typing import Self

from pydantic import BaseModel, Field, model_validator


class StoryYear(BaseModel):
    year: int = Field(ge=1, description="Relative year, starting at 1.")
    narrative: str


class _StoryContent(BaseModel):
    """LLM-authored content; the original seed is attached by the pipeline."""

    background: str
    years: list[StoryYear] = Field(min_length=1)

    @model_validator(mode="after")
    def check_years(self) -> Self:
        if [chapter.year for chapter in self.years] != list(range(1, len(self.years) + 1)):
            raise ValueError("Story years must be consecutive and ordered, starting at 1.")
        return self


class LifeStory(_StoryContent):
    seed: str = Field(min_length=1)


class Person(BaseModel):
    id: str
    name: str
    relationship: str = Field(description="Relationship to the protagonist, or 'self'.")
    age_at_start: int = Field(ge=0, description="Age at the beginning of year 1.")
    visual_description: str = Field(description="Baseline physical appearance.")


class PersonalObject(BaseModel):
    id: str
    name: str
    visual_description: str = Field(
        description="Recognizable color, material, shape, and distinctive details."
    )
    significance: str = Field(description="Why this possession matters to the person.")


class PersonAppearance(BaseModel):
    person_id: str
    appearance: str = Field(
        description="Visible action, clothing, pose, and any changes from the baseline."
    )


class ObjectAppearance(BaseModel):
    object_id: str
    appearance: str = Field(description="Placement, use, and condition in this photo.")


class PhotoIdea(BaseModel):
    year: int = Field(ge=1)
    month: int = Field(ge=1, le=12)
    description: str = Field(description="A concrete, photographable scene.")
    story_context: str = Field(description="How the moment relates to the life story.")
    people: list[PersonAppearance] = Field(
        description="Only people visible in the frame; may be empty."
    )
    objects: list[ObjectAppearance] = Field(min_length=1)


class CameraRoll(BaseModel):
    protagonist_id: str
    people: list[Person] = Field(min_length=1)
    objects: list[PersonalObject] = Field(min_length=1)
    photos: list[PhotoIdea] = Field(min_length=1)

    @model_validator(mode="after")
    def check_references_and_order(self) -> Self:
        person_ids = {person.id for person in self.people}
        object_ids = {obj.id for obj in self.objects}
        if len(person_ids | object_ids) != len(self.people) + len(self.objects):
            raise ValueError("People and objects must have unique IDs across the catalog.")
        if self.protagonist_id not in person_ids:
            raise ValueError(f"Unknown protagonist ID: {self.protagonist_id}")
        for index, photo in enumerate(self.photos, start=1):
            unknown_people = {p.person_id for p in photo.people} - person_ids
            unknown_objects = {obj.object_id for obj in photo.objects} - object_ids
            if unknown_people or unknown_objects:
                raise ValueError(
                    f"Photo {index} references unknown IDs: "
                    f"{sorted(unknown_people | unknown_objects)}"
                )
        dates = [(photo.year, photo.month) for photo in self.photos]
        if dates != sorted(dates):
            raise ValueError("Photos must be ordered by year and month.")
        return self


class GenerationResult(BaseModel):
    story: LifeStory
    camera_roll: CameraRoll
