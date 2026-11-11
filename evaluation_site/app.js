"use strict";

const DATA = window.EVAL_DATA;
const state = { condition: DATA.default_condition, rows: [], section: "faces", index: -1 };
const CONDITION = Object.fromEntries(DATA.conditions.map((c) => [c.key, c]));

/* ---------------------------------------------------------------- helpers */

function el(tag, props = {}, children = []) {
  const node = document.createElement(tag);
  Object.assign(node, props);
  for (const child of [].concat(children)) {
    if (child) node.append(child);
  }
  return node;
}

function scoreText(value) {
  if (value === null || value === undefined) return "—";
  return value.toFixed(3);
}

function scoreCell(row) {
  const td = el("td", { className: "score" });
  if (row.similarity === null || row.similarity === undefined) {
    td.append(el("span", { className: "flag", textContent: "not detected" }));
    return td;
  }
  const value = el("span", { textContent: scoreText(row.similarity) });
  if (row.similarity < 0) value.className = "neg";
  td.append(value);
  if (row.person_id === null) {
    td.append(el("span", { className: "flag", textContent: "no reference" }));
  } else if (row.candidate_count > 1) {
    td.append(el("span", { className: "flag", textContent: `${row.candidate_count} candidates` }));
  }
  return td;
}

function imageCell(src, alt, index, enabled) {
  const td = el("td", { className: "img" });
  if (!src) {
    td.append(el("span", { className: "none", textContent: "none" }));
    return td;
  }
  const img = new Image();
  img.src = src;
  img.alt = alt;
  img.loading = "lazy";
  const button = el("button", { type: "button", className: "thumb", title: `Enlarge ${alt}` }, img);
  button.setAttribute("aria-label", `Enlarge ${alt}`);
  if (enabled) {
    button.dataset.index = String(index);
  } else {
    button.disabled = true;
  }
  td.append(button);
  return td;
}

/* ------------------------------------------------------------- conditions */

function buildConditionSwitch() {
  const host = document.getElementById("conditions");
  for (const condition of DATA.conditions) {
    const button = el("button", { type: "button", textContent: condition.label });
    button.setAttribute("aria-pressed", String(condition.key === state.condition));
    button.addEventListener("click", () => {
      if (state.condition === condition.key) return;
      state.condition = condition.key;
      for (const other of host.children) {
        other.setAttribute("aria-pressed", String(other === button));
      }
      render();
    });
    host.append(button);
  }
}

/* ------------------------------------------------------------- step lists */

function renderSteps() {
  const face = DATA.face.method;
  const object = DATA.object.method;
  document.getElementById("faces-steps").append(
    el("li", {}, [document.createTextNode("Detect faces · ")]
      .concat([el("code", { textContent: `${face.detector} @ ${face.det_size.join("×")}, threshold ${face.detection_threshold}` })])),
    el("li", {}, [document.createTextNode("Align and crop · "), el("code", { textContent: `${face.crop.join("×")} via landmarks` })]),
    el("li", {}, [document.createTextNode("Embed · "), el("code", { textContent: face.recognition })]),
    el("li", { textContent: "Cosine similarity against the reference embedding" }),
  );
  document.getElementById("objects-steps").append(
    el("li", {}, [document.createTextNode("Detect · "), el("code", { textContent: object.detector })]),
    el("li", {}, [document.createTextNode(
      `NMS at IoU ${object.nms_iou}, keep boxes over ${object.box_threshold}, select highest confidence`,
    )]),
    el("li", {}, [document.createTextNode("Resize · "), el("code", { textContent: object.resize })]),
    el("li", {}, [document.createTextNode("Embed · "), el("code", { textContent: object.encoder })]),
    el("li", { textContent: "Cosine similarity against the reference embedding" }),
  );
}

/* ---------------------------------------------------------- table render */

function renderFaces(rows) {
  const body = document.getElementById("faces-body");
  body.replaceChildren();
  rows.forEach((row, index) => {
    const tr = el("tr", { className: index === 0 || rows[index - 1].photo_index !== row.photo_index ? "scene-start" : "" });
    tr.dataset.index = String(index);
    tr.append(
      el("td", { className: "scene" }, [
        el("div", { textContent: row.scene }),
        el("span", { className: "month", textContent: `photo ${row.photo_index + 1}` }),
      ]),
      el("td", { className: "pass", textContent: String(row.pass) }),
      el("td", { className: "name", textContent: row.person }),
      imageCell(row.reference_image, `${row.person} reference portrait`, index, false),
      imageCell(row.detected_image, `${row.person} detected face in ${row.scene}`, index, true),
      scoreCell(row),
    );
    body.append(tr);
  });
  const detected = rows.filter((r) => r.detected_image).length;
  document.getElementById("faces-count").textContent =
    `${rows.length} rows · ${detected} with a paired detection`;
}

function renderObjects(rows) {
  const body = document.getElementById("objects-body");
  body.replaceChildren();
  rows.forEach((row, index) => {
    const tr = el("tr", { className: index === 0 || rows[index - 1].photo_index !== row.photo_index ? "scene-start" : "" });
    tr.dataset.index = String(index);
    tr.append(
      el("td", { className: "scene" }, [
        el("div", { textContent: row.scene }),
        el("span", { className: "month", textContent: `photo ${row.photo_index + 1}` }),
      ]),
      el("td", { className: "name" }, [
        el("div", { textContent: row.object }),
        el("span", { className: "query" }, [el("code", { textContent: row.query })]),
      ]),
      imageCell(row.reference_image, `${row.object} reference`, index, true),
      imageCell(row.detected_image, `${row.object} detected in ${row.scene}`, index, true),
      scoreCell(row),
    );
    body.append(tr);
  });
  const detected = rows.filter((r) => r.detected_image).length;
  document.getElementById("objects-count").textContent =
    `${rows.length} rows · ${detected} with a detection`;
}

function render() {
  const condition = CONDITION[state.condition];
  document.getElementById("condition-note").textContent = condition.note;
  const faces = DATA.face.rows.filter((r) => r.condition === state.condition);
  const objects = DATA.object.rows.filter((r) => r.condition === state.condition);
  renderFaces(faces);
  renderObjects(objects);
}

/* ---------------------------------------------------------------- modal */

const dialog = document.getElementById("modal");

function pairFigure(src, caption, faceCrop) {
  const figure = el("figure");
  const frame = el("div", { className: "frame" });
  if (src) {
    const img = new Image();
    img.src = src;
    img.alt = caption;
    if (faceCrop) img.className = "face";
    frame.append(img);
  } else {
    frame.append(el("span", { className: "none", textContent: "no image" }));
  }
  figure.append(frame, el("figcaption", { textContent: caption }));
  return figure;
}

function details(entries) {
  const dl = el("dl");
  for (const [term, value] of entries) {
    if (value === null || value === undefined || value === "") continue;
    dl.append(el("dt", { textContent: term }), el("dd", { textContent: String(value) }));
  }
  return el("details", { open: true }, [el("summary", { textContent: "Detector details" }), dl]);
}

function faceModal(row) {
  const body = el("div");
  body.append(
    el("div", { className: "pair" }, [
      pairFigure(row.reference_image, "Reference portrait · aligned crop given to ArcFace", true),
      pairFigure(row.detected_image, "Detected face · aligned crop given to ArcFace", true),
    ]),
    el("div", { className: "modal-score" }, [
      document.createTextNode(`cosine ${scoreText(row.similarity)}`),
      el("span", { textContent: row.similarity === null ? "no pairing for this character" : "higher is closer in ArcFace space" }),
    ]),
    details([
      ["Scene", `${row.scene} · pass ${row.pass}`],
      ["Requested character", row.person],
      ["Detector confidence", row.detector_confidence === null ? null : row.detector_confidence.toFixed(3)],
      ["Face index in pass", row.face_index === null ? null : `${row.face_index} (numbered left to right)`],
      ["Bounding box", row.bbox === null ? null : row.bbox.map((v) => v.toFixed(0)).join(", ")],
      ["Detected faces in pass", `${row.detected_faces} for ${row.expected_faces} requested`],
      ["Unassigned faces in pass", row.unassigned.length ? row.unassigned.join(", ") : "none"],
    ]),
    el("figure", { className: "context" }, [
      (() => {
        const img = new Image();
        img.src = row.detections_image;
        img.alt = `Detected faces in ${row.scene}`;
        return img;
      })(),
      el("figcaption", { textContent: "What the detector saw: every face in this pass, numbered by index." }),
    ]),
  );
  return body;
}

function objectModal(row) {
  const body = el("div");
  const alternatives = el("ul", { className: "alts" });
  row.candidates.forEach((candidate, rank) => {
    const li = el("li", { className: rank === 0 ? "selected" : "" });
    const img = new Image();
    img.src = candidate.image;
    img.alt = `${row.object} candidate ${rank + 1}`;
    li.append(img, el("div", { textContent: `${scoreText(candidate.similarity)} · conf ${candidate.confidence.toFixed(2)}` }));
    alternatives.append(li);
  });
  body.append(
    el("div", { className: "pair" }, [
      pairFigure(row.reference_image, `Canonical reference · cropped for “${row.query}”`, false),
      pairFigure(row.detected_image, "Selected candidate · highest detector confidence", false),
    ]),
    el("div", { className: "modal-score" }, [
      document.createTextNode(`cosine ${scoreText(row.similarity)}`),
      el("span", { textContent: row.detected_image ? "higher is closer in DINOv2 space" : "no detection, so no similarity" }),
    ]),
    details([
      ["Scene", row.scene],
      ["Requested object", row.object],
      ["Detector query", row.query],
      ["Requested state", row.requirement],
      ["Detector confidence", row.detector_confidence === null ? null : row.detector_confidence.toFixed(3)],
      ["Candidate boxes", `${row.candidate_count} (highest confidence is scored)`],
    ]),
    row.candidates.length > 1
      ? el("div", {}, [el("p", { className: "foot", textContent: "All candidates for this query, best confidence first. Only the first is scored." }), alternatives])
      : el("span"),
    el("figure", { className: "context" }, [
      (() => {
        const img = new Image();
        img.src = row.detections_image;
        img.alt = `Detected objects in ${row.scene}`;
        return img;
      })(),
      el("figcaption", { textContent: "What the detector saw: yellow boxes are the scored candidate, blue are alternatives." }),
    ]),
  );
  return body;
}

function openModal(section, index) {
  state.section = section;
  state.index = index;
  state.rows = section === "faces"
    ? DATA.face.rows.filter((r) => r.condition === state.condition)
    : DATA.object.rows.filter((r) => r.condition === state.condition);
  const row = state.rows[index];
  if (!row) return;
  const title = section === "faces"
    ? `${row.person} · ${row.scene} · pass ${row.pass}`
    : `${row.object} · ${row.scene}`;
  document.getElementById("modal-title").textContent = title;
  document.getElementById("modal-body").replaceChildren(section === "faces" ? faceModal(row) : objectModal(row));
  document.getElementById("prev").disabled = index <= 0;
  document.getElementById("next").disabled = index >= state.rows.length - 1;
  if (!dialog.open) dialog.showModal();
}

function step(delta) {
  const next = state.index + delta;
  if (next < 0 || next >= state.rows.length) return;
  openModal(state.section, next);
}

document.addEventListener("click", (event) => {
  const button = event.target.closest("[data-index]");
  if (!button) return;
  const section = button.closest("section").id;
  const index = Number(button.dataset.index);
  if (section === "faces" || section === "objects") openModal(section, index);
});

document.getElementById("prev").addEventListener("click", () => step(-1));
document.getElementById("next").addEventListener("click", () => step(1));
document.getElementById("close").addEventListener("click", () => dialog.close());
dialog.addEventListener("click", (event) => {
  if (event.target === dialog) dialog.close();
});
document.addEventListener("keydown", (event) => {
  if (!dialog.open) return;
  if (event.key === "ArrowLeft") step(-1);
  if (event.key === "ArrowRight") step(1);
});

/* ------------------------------------------------------------------ boot */

document.getElementById("built").textContent = `Built ${DATA.generated_utc}`;
document.getElementById("experiment").textContent = DATA.experiment;
buildConditionSwitch();
renderSteps();
render();
