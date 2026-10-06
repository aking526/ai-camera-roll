const $ = (selector) => document.querySelector(selector);
const months = [
  "January",
  "February",
  "March",
  "April",
  "May",
  "June",
  "July",
  "August",
  "September",
  "October",
  "November",
  "December",
];
let data = null;
let images = { photos: [], references: [] };
let view = "timeline";
let selected = -1;
let selectedPass = -1;
let requestVersion = 0;
const esc = (value) =>
  String(value).replace(
    /[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );
const initials = (name) =>
  name
    .split(/\s+/)
    .slice(0, 2)
    .map((part) => part[0])
    .join("");
const date = (photo) => `${months[photo.month - 1]} · Year ${photo.year}`;
const sceneTitle = (photo) => photo.description.split(/(?<=[.!?])\s/)[0];
const catalog = (kind) => data.camera_roll[kind];
const entity = (kind, id) => catalog(kind).find((item) => item.id === id);
const key = (kind) => (kind === "people" ? "person_id" : "object_id");
const appearances = (kind, id) =>
  data.camera_roll.photos
    .map((photo, index) => ({ photo, index }))
    .filter(({ photo }) => photo[kind].some((item) => item[key(kind)] === id));
const photoImages = (index) => images.photos.find((photo) => photo.photo_index === index);
const referenceImage = (kind, id) =>
  images.references.find((asset) => asset.kind === kind && asset.entity_id === id);
const passUrl = (photo, pass) =>
  photo.complete && pass.pass_index === photo.passes.length - 1
    ? photo.final_url || pass.url
    : pass.url;
const latestPass = (photo) =>
  photo ? [...photo.passes].reverse().find((pass) => passUrl(photo, pass)) : null;
const photoPreview = (index) => {
  const photo = photoImages(index);
  return photo?.final_url || latestPass(photo)?.url;
};
const passLabel = (photo, pass) =>
  `Pass ${pass.pass_index + 1}${
    photo.complete && pass.pass_index === photo.passes.length - 1 ? " · Final" : ""
  }`;

function imageFrame(url, alt, className = "", placeholder = "Image not generated yet", eager = false) {
  return `<span class="image-frame ${className}">${
    url
      ? `<img src="${esc(url)}" alt="${esc(alt)}" loading="${eager ? "eager" : "lazy"}" decoding="async" />`
      : ""
  }<span class="image-placeholder" ${url ? "hidden" : ""}>${esc(placeholder)}</span></span>`;
}

function referencePreview(kind, id, className = "") {
  const item = entity(kind, id);
  const asset = referenceImage(kind, id);
  return `<button class="reference-preview ${className}" data-kind="${kind}" data-entity="${esc(
    id
  )}" aria-label="View ${esc(item.name)} reference image">${imageFrame(
    asset?.url,
    `${item.name} reference image`,
    "",
    asset?.completed ? "Reference image unavailable" : "Reference not generated yet"
  )}<span class="reference-caption">Reference image ↗</span></button>`;
}

function photoGallery(index) {
  const photo = photoImages(index);
  const pass = photo?.passes.find((step) => step.pass_index === selectedPass);
  const missing = photo?.passes.some((step) => step.completed)
    ? "Photo image unavailable"
    : "Photo not generated yet";
  return `<section class="photo-gallery" aria-label="Photo and generation passes">
    <figure class="photo-figure"><div id="photo-preview">${imageFrame(
      pass ? passUrl(photo, pass) : null,
      `${date(data.camera_roll.photos[index])}, ${pass ? passLabel(photo, pass) : "photo"}: ${data.camera_roll.photos[index].description}`,
      "photo-image",
      missing,
      true
    )}</div><figcaption id="pass-caption" class="image-caption" aria-live="polite">${
    pass ? esc(passLabel(photo, pass)) : "Generated photo"
  }</figcaption></figure>
    ${
      photo && !photo.complete
        ? '<p class="schema-note">Generation is incomplete. Showing available passes.</p>'
        : ""
    }
    ${
      photo?.passes.length > 1
        ? `<div class="pass-strip" role="group" aria-label="Image evolution">${photo.passes
            .map((step) => {
              const url = passUrl(photo, step);
              const state = step.completed ? "Unavailable" : "Pending";
              return `<button class="pass-button" data-pass="${step.pass_index}" aria-pressed="${
                step.pass_index === selectedPass
              }" aria-label="View ${esc(passLabel(photo, step))}${url ? "" : `, ${state}`}" ${
                url ? "" : "disabled"
              }>${imageFrame(url, "", "", state)}<span>${esc(passLabel(photo, step))}</span></button>`;
            })
            .join("")}</div>`
        : ""
    }</section>`;
}

function selectPass(index) {
  const photo = photoImages(selected);
  const pass = photo?.passes.find((step) => step.pass_index === index);
  if (!pass || !passUrl(photo, pass)) return;
  selectedPass = index;
  // Update only the image so the thumbnail's focus and popup scroll are retained.
  $("#photo-preview").innerHTML = imageFrame(
    passUrl(photo, pass),
    `${date(data.camera_roll.photos[selected])}, ${passLabel(photo, pass)}: ${data.camera_roll.photos[selected].description}`,
    "photo-image",
    "Photo image unavailable",
    true
  );
  $("#pass-caption").textContent = passLabel(photo, pass);
  document.querySelectorAll("[data-pass]").forEach((button) =>
    button.setAttribute("aria-pressed", String(Number(button.dataset.pass) === index))
  );
}

const modelName = (kind) => (kind === "people" ? "Person" : "PersonalObject");
const recordPath = (kind, id) =>
  `camera_roll.${kind}[${catalog(kind).findIndex((item) => item.id === id)}]`;
const recordMeta = (path, type) =>
  `<div class="record-meta"><code>${esc(path)}</code><span class="type-label">${esc(
    type
  )}</span></div>`;
const referenceLink = (kind, id) =>
  `<button class="reference-link" data-kind="${kind}" data-entity="${esc(id)}" title="Open ${esc(
    recordPath(kind, id)
  )}"><code>${esc(id)}</code> ↗</button>`;

function fields(record) {
  return `<dl class="record-fields">${Object.entries(record)
    .filter(([, value]) => value === null || typeof value !== "object")
    .map(([name, value]) => {
      const type = value === null ? "null" : Number.isInteger(value) ? "integer" : typeof value;
      const kind =
        name === "person_id" || name === "protagonist_id"
          ? "people"
          : name === "object_id"
          ? "objects"
          : null;
      const content = kind
        ? referenceLink(kind, value)
        : name === "id"
        ? `<code>${esc(value)}</code>`
        : esc(value);
      return `<div class="record-field"><dt><code>${esc(
        name
      )}</code><span class="type-label">${type}</span></dt><dd>${content}</dd></div>`;
    })
    .join("")}</dl>`;
}

function arrayHeading(name, type, count) {
  return `<h3 class="array-heading"><code>${esc(name)}</code><span class="type-label">${esc(
    type
  )}[]</span><span class="metadata">${count} ${count === 1 ? "item" : "items"}</span></h3>`;
}

function error(message) {
  $("#notice").textContent = message;
  $("#notice").hidden = !message;
}

async function load(responsePromise, name) {
  const version = ++requestVersion;
  error("");
  try {
    const response = await responsePromise;
    const payload = await response.json();
    if (version !== requestVersion) return;
    if (!response.ok) throw new Error(payload.error || "Unable to open this camera roll.");
    data = payload.result;
    images = payload.images || { photos: [], references: [] };
    selected = -1;
    selectedPass = -1;
    document.querySelectorAll("dialog[open]").forEach((dialog) => dialog.close());
    error(images.notice || "");
    const protagonist = entity("people", data.camera_roll.protagonist_id);
    $("#open-file").title = `Current file: ${name || payload.name}`;
    document.title = `${protagonist.name} · Camera Roll`;
    $("#read-story").disabled = false;
    $("#roll-reference").innerHTML = `<code>protagonist_id</code> ${referenceLink(
      "people",
      data.camera_roll.protagonist_id
    )}`;
    $("#count-timeline").textContent = data.camera_roll.photos.length;
    $("#count-people").textContent = catalog("people").length;
    $("#count-objects").textContent = catalog("objects").length;
    $("#year-filter").innerHTML =
      '<option value="all">All years</option>' +
      [...new Set(data.camera_roll.photos.map((p) => p.year))]
        .map((year) => `<option value="${year}">Year ${year}</option>`)
        .join("");
    $("#entity-filter").innerHTML =
      '<option value="all">Everyone & everything</option>' +
      ["people", "objects"]
        .map(
          (kind) =>
            `<optgroup label="${kind === "people" ? "People" : "Objects"}">${catalog(kind)
              .map((item) => `<option value="${esc(item.id)}">${esc(item.name)}</option>`)
              .join("")}</optgroup>`
        )
        .join("");
    $("#catalog-search").value = "";
    switchView("timeline");
    $("#timeline-scroll").scrollLeft = 0;
  } catch (err) {
    if (version === requestVersion) error(err.message || "Unable to connect to the local viewer.");
  }
}

function visibleEvents() {
  const year = $("#year-filter").value;
  const id = $("#entity-filter").value;
  return data.camera_roll.photos
    .map((photo, index) => ({ photo, index }))
    .filter(
      ({ photo }) =>
        (year === "all" || photo.year === Number(year)) &&
        (id === "all" ||
          photo.people.some((p) => p.person_id === id) ||
          photo.objects.some((o) => o.object_id === id))
    );
}

function chips(photo, kind) {
  if (!photo[kind].length) return '<p class="empty-connection">No people in frame</p>';
  return photo[kind]
    .map((appearance) => {
      const item = entity(kind, appearance[key(kind)]);
      return `<button class="entity-chip ${kind}" data-kind="${kind}" data-entity="${esc(
        item.id
      )}" title="View ${esc(item.name)}"><span class="chip-icon" aria-hidden="true">${
        kind === "people" ? esc(initials(item.name)) : "◇"
      }</span><span class="chip-label">${esc(item.name)}<code>${esc(
        item.id
      )}</code></span></button>`;
    })
    .join("");
}

function renderTimeline() {
  if (!data) return;
  const events = visibleEvents();
  if (!events.some((e) => e.index === selected)) selected = -1;
  $(
    "#event-count"
  ).innerHTML = `<code>camera_roll.photos</code> <span class="type-label">PhotoIdea[]</span> · ${events.length} / ${data.camera_roll.photos.length}`;
  const height = Math.max(
    130,
    Math.max(0, ...events.map(({ photo }) => photo.people.length)) * 76 + 28
  );
  $("#timeline").style.setProperty("--people-height", `${height}px`);
  const hasImages = events.some(({ index }) => photoPreview(index));
  $("#timeline").style.setProperty("--event-height", hasImages ? "310px" : "136px");
  $("#timeline").innerHTML = events.length
    ? events
        .map(
          ({ photo, index }) => `
    <div class="event-column">
      <div class="connections people ${photo.people.length ? "" : "none"}">${chips(
            photo,
            "people"
          )}</div>
      <div class="event-slot"><button class="event-button ${
        selected === index ? "selected" : ""
      }" data-event="${index}" aria-haspopup="dialog" aria-controls="moment-dialog" aria-label="Event ${
            index + 1
          }, ${esc(date(photo))}">
        ${
          hasImages
            ? imageFrame(
                photoPreview(index),
                `${date(photo)}: ${photo.description}`,
                "event-thumbnail",
                photoImages(index)?.passes.some((pass) => pass.completed)
                  ? "Photo image unavailable"
                  : "Photo not generated yet"
              )
            : ""
        }
        <span class="event-meta"><span>${esc(
          date(photo)
        )}</span><span class="event-number">${String(index + 1).padStart(2, "0")}</span></span>
        <span class="event-title">${esc(sceneTitle(photo))}</span>
        <code class="event-path">photos[${index}]</code>
      </button></div>
      <div class="connections objects">${chips(photo, "objects")}</div>
    </div>`
        )
        .join("")
    : '<div class="empty">No moments match these filters. Try another year or connection.</div>';
}

function appearanceDetails(photo, kind) {
  return photo[kind]
    .map((appearance, index) => {
      return `<section class="appearance-record ${kind}">${recordMeta(
        `${kind}[${index}]`,
        kind === "people" ? "PersonAppearance" : "ObjectAppearance"
      )}<div class="appearance-content">${referencePreview(
        kind,
        appearance[key(kind)],
        "appearance-reference"
      )}${fields(appearance)}</div></section>`;
    })
    .join("");
}

function renderDetail() {
  const photo = data.camera_roll.photos[selected];
  if (!photo) return;
  const focusedStep = document.activeElement?.dataset.step;
  const events = visibleEvents();
  const position = events.findIndex((e) => e.index === selected);
  $("#event-detail").innerHTML = `
    <div class="detail-top"><span class="eyebrow">MOMENT ${String(selected + 1).padStart(
      2,
      "0"
    )} / ${
    data.camera_roll.photos.length
  }</span><div class="pager"><button data-step="-1" aria-label="Previous event" ${
    position === 0 ? "disabled" : ""
  }>←</button><button data-step="1" aria-label="Next event" ${
    position === events.length - 1 ? "disabled" : ""
  }>→</button></div></div>
    <h2 id="moment-heading">${esc(date(photo))}</h2>
    ${photoGallery(selected)}
    ${recordMeta(`camera_roll.photos[${selected}]`, "PhotoIdea")}
    ${fields(photo)}
    ${arrayHeading("people", "PersonAppearance", photo.people.length)}
    <p class="schema-note">Each person_id references a shared camera_roll.people record. appearance describes this moment.</p>
    ${
      photo.people.length
        ? appearanceDetails(photo, "people")
        : '<p class="schema-note"><code>[]</code> — No people visible in this frame.</p>'
    }
    ${arrayHeading("objects", "ObjectAppearance", photo.objects.length)}
    <p class="schema-note">Each object_id references a shared camera_roll.objects record. appearance describes its use or condition here.</p>
    ${appearanceDetails(photo, "objects")}`;
  const dialog = $("#moment-dialog");
  if (!dialog.open) dialog.showModal();
  if (focusedStep) {
    const button = $(`[data-step="${focusedStep}"]`);
    (button.disabled
      ? $("#event-detail .pager button:not(:disabled)") || $("#close-moment")
      : button
    ).focus();
  }
  dialog.scrollTop = 0;
}

function switchView(next) {
  view = next;
  document.querySelectorAll("[data-view]").forEach((button) => {
    if (button.dataset.view === next) button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
  });
  $("#timeline-view").hidden = next !== "timeline";
  $("#catalog-view").hidden = next === "timeline";
  if (next === "timeline") renderTimeline();
  else {
    $("#catalog-search").value = "";
    renderCatalog();
  }
}

function renderCatalog() {
  $(
    "#catalog-description"
  ).innerHTML = `<code>camera_roll.${view}</code> <span class="type-label">${modelName(
    view
  )}[]</span> · Shared records`;
  if (!data) {
    $("#catalog").innerHTML = '<p class="empty">Open a camera roll to explore this collection.</p>';
    return;
  }
  const query = $("#catalog-search").value.trim().toLocaleLowerCase();
  const items = catalog(view).filter((item) =>
    Object.values(item).join(" ").toLocaleLowerCase().includes(query)
  );
  $("#catalog").innerHTML = items.length
    ? items
        .map((item) => {
          const count = appearances(view, item.id).length;
          return `<article class="catalog-card">
          ${referencePreview(view, item.id, "catalog-reference")}
          ${recordMeta(recordPath(view, item.id), modelName(view))}
          ${fields(item)}
          <button class="foot" data-kind="${view}" data-entity="${esc(item.id)}"><span>${count} ${
            count === 1 ? "appearance" : "appearances"
          }</span><span>View references ↗</span></button></article>`;
        })
        .join("")
    : '<p class="empty">No matches. Try another name or description.</p>';
}

function showDialog(label, content) {
  $("#dialog-label").textContent = label;
  $("#dialog-content").innerHTML = content;
  $("#reader").showModal();
  $("#reader").scrollTop = 0;
}

function showEntity(kind, id) {
  const item = entity(kind, id);
  const events = appearances(kind, id);
  showDialog(
    `${modelName(kind)} · Shared record`,
    `<h2>${esc(item.name)}</h2>
    <figure class="reference-detail">${imageFrame(
      referenceImage(kind, id)?.url,
      `${item.name} reference image`,
      "",
      referenceImage(kind, id)?.completed ? "Reference image unavailable" : "Reference not generated yet",
      true
    )}<figcaption class="image-caption">Reference image</figcaption></figure>
    ${recordMeta(recordPath(kind, id), modelName(kind))}
    ${fields(item)}
    <h3>Referenced by ${events.length} ${events.length === 1 ? "photo" : "photos"}</h3>
    <p class="schema-note">Derived links below point to scene-specific appearances; they are not additional fields on this record.</p>
    ${
      events
        .map(
          ({ photo, index }) =>
            `<button class="related-event" data-jump="${index}"><strong>${esc(
              date(photo)
            )} ↗</strong><code class="reference-path">camera_roll.photos[${index}].${kind}[${photo[
              kind
            ].findIndex(
              (a) => a[key(kind)] === id
            )}]</code><span class="field-caption">appearance</span>${esc(
              photo[kind].find((a) => a[key(kind)] === id).appearance
            )}</button>`
        )
        .join("") || '<p class="prose">No visible appearances in this roll.</p>'
    }`
  );
}

function selectEvent(index, scroll = false) {
  selected = index;
  selectedPass = latestPass(photoImages(index))?.pass_index ?? -1;
  document.querySelectorAll("[data-event]").forEach((button) => {
    const active = Number(button.dataset.event) === index;
    button.classList.toggle("selected", active);
    if (active && scroll) button.scrollIntoView({ block: "nearest", inline: "center" });
  });
  renderDetail();
}

document.addEventListener("click", (event) => {
  const target = event.target.closest("button");
  if (!target) return;
  if (target.dataset.view) switchView(target.dataset.view);
  if (target.dataset.event !== undefined) selectEvent(Number(target.dataset.event));
  if (target.dataset.pass !== undefined) selectPass(Number(target.dataset.pass));
  if (target.dataset.entity) showEntity(target.dataset.kind, target.dataset.entity);
  if (target.dataset.step) {
    const events = visibleEvents();
    const next =
      events[events.findIndex((e) => e.index === selected) + Number(target.dataset.step)];
    if (next) selectEvent(next.index, true);
  }
  if (target.dataset.jump !== undefined) {
    $("#reader").close();
    $("#moment-dialog").close();
    $("#year-filter").value = "all";
    $("#entity-filter").value = "all";
    selected = Number(target.dataset.jump);
    switchView("timeline");
    $(`[data-event="${selected}"]`).focus({ preventScroll: true });
    selectEvent(selected, true);
  }
});
// Files may disappear or be damaged after the manifest is loaded.
document.addEventListener("error", (event) => {
  const image = event.target;
  if (!(image instanceof HTMLImageElement) || !image.closest(".image-frame")) return;
  image.hidden = true;
  const placeholder = image.closest(".image-frame").querySelector(".image-placeholder");
  placeholder.textContent = "Image unavailable";
  placeholder.hidden = false;
}, true);
$("#year-filter").addEventListener("change", renderTimeline);
$("#entity-filter").addEventListener("change", renderTimeline);
$("#catalog-search").addEventListener("input", renderCatalog);
$("#close-dialog").addEventListener("click", () => $("#reader").close());
$("#close-moment").addEventListener("click", () => $("#moment-dialog").close());
document.querySelectorAll("dialog").forEach((dialog) =>
  dialog.addEventListener("click", (event) => {
    if (event.target === dialog) {
      const r = dialog.getBoundingClientRect();
      if (
        event.clientX < r.left ||
        event.clientX > r.right ||
        event.clientY < r.top ||
        event.clientY > r.bottom
      )
        dialog.close();
    }
  })
);
$("#read-story").addEventListener("click", () => {
  if (!data) return;
  showDialog(
    "LifeStory",
    `${recordMeta("story", "LifeStory")}${fields(data.story)}
    ${arrayHeading("years", "StoryYear", data.story.years.length)}${data.story.years
      .map(
        (year, index) =>
          `<section class="appearance-record">${recordMeta(
            `story.years[${index}]`,
            "StoryYear"
          )}${fields(year)}</section>`
      )
      .join("")}`
  );
});
$("#open-file").addEventListener("click", () => $("#file-input").click());
$("#file-input").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  event.target.value = "";
  if (file.size > 5 * 1024 * 1024) {
    error("Choose a YAML file smaller than 5 MB.");
    return;
  }
  await load(
    fetch("/api/preview", {
      method: "POST",
      headers: { "Content-Type": "application/yaml" },
      body: file,
    }),
    file.name
  );
});
load(fetch("/api/result"));
