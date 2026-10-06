const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

// Exercise the viewer's rendering and selection logic without a browser or network.
function viewer() {
  const people = [
    { id: "person_maya", name: "Maya" },
    { id: "person_jonah", name: "Jonah" },
  ];
  const objects = [
    { id: "object_cap", name: "Green cap" },
    { id: "object_book", name: "Sketchbook" },
    { id: "object_mug", name: "Blue mug" },
  ];
  const initial = [
    { slot: 0, kind: "people", entity_id: "person_maya", is_new: true },
    { slot: 1, kind: "people", entity_id: "person_jonah", is_new: true },
    { slot: 2, kind: "objects", entity_id: "object_cap", is_new: true },
    { slot: 3, kind: "objects", entity_id: "object_book", is_new: true },
  ];
  const fixture = {
    result: {
      camera_roll: {
        people,
        objects,
        photos: [{
          year: 1, month: 1, description: "Maya and Jonah at a studio desk.",
          story_context: "Late-night studio work.",
          people: people.map((item) => ({ person_id: item.id, appearance: "At the desk." })),
          objects: objects.map((item) => ({ object_id: item.id, appearance: "On the desk." })),
        }],
      },
    },
    images: {
      references: [
        ...people.map((item) => ({ ...item, entity_id: item.id, kind: "people" })),
        ...objects.map((item) => ({ ...item, entity_id: item.id, kind: "objects" })),
      ].map((asset) => ({
        ...asset, completed: true, url: `/images/${asset.id}.png`, prompt: `Reference prompt for ${asset.name}`,
      })),
      photos: [{
        photo_index: 0, complete: true, final_url: "/images/final.png",
        passes: [
          {
            pass_index: 0, completed: true, url: "/images/pass_00.png", inputs: initial,
            carried_forward: [], prompt: "Create studio scene.\nUse Maya and Jonah's references.",
          },
          {
            pass_index: 1, completed: true, url: "/images/pass_01.png",
            prompt: "Edit the current scene by adding the blue mug.",
            inputs: [
              { slot: 0, kind: "scene", source_pass_index: 0 },
              { slot: 1, kind: "objects", entity_id: "object_mug", is_new: true },
            ],
            carried_forward: initial.map(({ kind, entity_id }) => ({ kind, entity_id })),
          },
        ],
      }],
    },
  };
  const elements = new Map();
  const element = (selector) => {
    if (!elements.has(selector)) elements.set(selector, {
      innerHTML: "", textContent: "", value: "all", open: false, scrollTop: 0,
      addEventListener() {},
      showModal() { this.open = true; },
      close() { this.open = false; },
    });
    return elements.get(selector);
  };
  const buttons = [0, 1].map((index) => ({
    dataset: { pass: String(index) },
    attributes: {},
    setAttribute(name, value) { this.attributes[name] = value; },
  }));
  const document = {
    activeElement: buttons[0],
    querySelector: element,
    querySelectorAll: (selector) => selector === "[data-pass]" ? buttons : [],
    addEventListener() {},
  };
  const context = vm.createContext({ document, fixture, fetch: () => new Promise(() => {}) });
  vm.runInContext(fs.readFileSync(path.join(__dirname, "../src/ai_camera_roll/web/app.js"), "utf8"), context);
  vm.runInContext("data = fixture.result; images = fixture.images; selected = 0;", context);
  return { fixture, document, element, buttons, run: (code) => vm.runInContext(code, context) };
}

test("initial pass shows each supplied reference as new", () => {
  const app = viewer();
  const html = app.run("passReferences(photoImages(0), photoImages(0).passes[0])");
  for (const name of ["Maya", "Jonah", "Green cap", "Sketchbook"]) assert.ok(html.includes(name));
  assert.ok(!html.includes("Blue mug"));
  assert.equal((html.match(/data-new="true"/g) || []).length, 4);
  assert.ok(!html.includes("Previous scene"));
  assert.ok(!html.includes("Carried forward"));
});

test("edit pass separates the scene, new reference, and carried identities", () => {
  const app = viewer();
  const html = app.run("passReferences(photoImages(0), photoImages(0).passes[1])");
  assert.ok(html.includes("Scene from Pass 1"));
  assert.ok(html.includes("Blue mug"));
  assert.equal((html.match(/data-new="true"/g) || []).length, 1);
  assert.ok(html.includes("Carried forward in the scene image"));
  assert.ok(html.includes("Green cap"));
  assert.ok(html.includes("Sketchbook"));
});

test("pass selection updates references while preserving popup scroll and focus", () => {
  const app = viewer();
  assert.equal(app.run("selectEvent(0); selectedPass"), 1);
  const detail = app.element("#event-detail").innerHTML;
  app.element("#moment-dialog").scrollTop = 250;
  const focus = app.document.activeElement;
  app.run("selectPass(0)");
  assert.equal(app.element("#pass-caption").textContent, "Pass 1");
  assert.ok(app.element("#photo-preview").innerHTML.includes("/images/pass_00.png"));
  assert.ok(!app.element("#pass-references").innerHTML.includes("Blue mug"));
  assert.equal(app.buttons[0].attributes["aria-pressed"], "true");
  assert.equal(app.buttons[1].attributes["aria-pressed"], "false");
  assert.equal(app.element("#event-detail").innerHTML, detail);
  assert.equal(app.element("#moment-dialog").scrollTop, 250);
  assert.equal(app.document.activeElement, focus);
  assert.equal(app.run("selectEvent(0); selectedPass"), 1);
});

test("pending passes can display their planned references", () => {
  const app = viewer();
  const photo = app.fixture.images.photos[0];
  photo.complete = false;
  photo.final_url = null;
  photo.passes[1].completed = false;
  photo.passes[1].url = null;
  app.run("selectPass(1)");
  assert.equal(app.element("#pass-caption").textContent, "Pass 2 · Pending");
  assert.ok(app.element("#photo-preview").innerHTML.includes("Pass not generated yet"));
  assert.ok(app.element("#pass-references").innerHTML.includes("Blue mug"));
  assert.ok(!app.run("photoGallery(0)").includes("disabled"));
});

test("resupplied references are labeled as reused", () => {
  const app = viewer();
  Object.assign(app.fixture.images.photos[0].passes[1].inputs[1], {
    entity_id: "object_cap", is_new: false,
  });
  const html = app.run("passReferences(photoImages(0), photoImages(0).passes[1])");
  assert.ok(html.includes("Reused reference"));
  assert.ok(!html.includes("New this pass"));
});

test("prompts are collapsed initially and safely preserve their text", () => {
  const app = viewer();
  const prompt = 'Use <Maya> & "Jonah".\nDo not render <script>alert(1)</script>.';
  app.fixture.images.photos[0].passes[1].prompt = prompt;
  app.run("selectEvent(0)");
  const html = app.element("#event-detail").innerHTML;
  assert.ok(html.includes("Show prompt"));
  assert.ok(!/<details[^>]*\bopen\b/.test(html));
  assert.ok(html.includes("&lt;Maya&gt; &amp; &quot;Jonah&quot;.\n"));
  assert.ok(!html.includes("<script>"));
});

test("switching passes updates the prompt and retains its expanded state", () => {
  const app = viewer();
  app.element("#pass-prompt details").open = true;
  app.run("selectPass(0)");
  const html = app.element("#pass-prompt").innerHTML;
  assert.ok(/<details[^>]*\bopen\b/.test(html));
  assert.ok(html.includes("Create studio scene.\n"));
  assert.ok(!html.includes("adding the blue mug"));
});

test("missing prompts show an explanatory message", () => {
  const app = viewer();
  app.fixture.images.photos[0].passes[1].prompt = null;
  app.run("selectPass(1)");
  assert.ok(app.element("#pass-prompt").innerHTML.includes("No prompt recorded"));
});

test("reference popups include their saved generation prompt", () => {
  const app = viewer();
  app.run('showEntity("people", "person_maya")');
  const html = app.element("#dialog-content").innerHTML;
  assert.ok(html.includes("Reference prompt for Maya"));
  assert.ok(!/<details[^>]*\bopen\b/.test(html));
});
