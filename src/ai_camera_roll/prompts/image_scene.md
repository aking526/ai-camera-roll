Convert a story-backed PhotoIdea into image-model prompts for the supplied ordered
passes. Return exactly one prompt per pass in the same order. Input slot mappings
are fixed by the caller: do not reorder slots or invent additional references.

The final result is one candid, everyday phone photograph. Preserve the concrete
scene, season, setting, composition, lighting, visible people and objects, and
scene-specific appearances. The story and catalogs are the source of truth.
Use story_context to guide the moment; do not literally render emotions, captions,
biography, or events outside the frame. Account for aging and narrative changes.
Reference portraits establish faces and physical identity, not fixed outfits or
poses. Object references establish recognizable design, not placement or temporary
condition. Include each new entity exactly once in its specified scene role.

For the first pass, compose the scene using only the supplied new_ids. Explicitly
describe each reference by its numbered image slot (image 0, image 1, etc.) and
name. Preserve each person's identity and each object's distinctive appearance,
while applying the requested pose, clothes, placement, and condition. If additional
people or personal objects are deferred to later passes, omit them for now and
leave their intended positions available; the final photo will be completed by
editing. Ordinary untracked background details are permitted if the scene needs
them. Do not copy the white reference backgrounds into the scene.

For each later pass, image 0 is the current scene. Write a precise editing prompt
that adds or restores the entities shown in the other supplied images into their
correct roles. Preserve the current image's camera viewpoint, composition,
lighting, background, and all already_established_ids. Do not add another copy
of an entity if it is already visible: correct its appearance using the reference.
Explain each new entity's position and interactions with existing subjects.
For example, a cap should be worn by its owner rather than floating in the frame.
Change only what is needed to complete this pass. Do not preserve a studio pose
or white background from an asset reference. The last pass must realize the full
supplied PhotoIdea without extra people or extra cataloged possessions.
