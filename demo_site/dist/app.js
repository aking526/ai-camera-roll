'use strict';

(() => {
  let data = window.CAMERA_ROLL_DATA;
  if (!data) {
    document.getElementById('album-grid').innerHTML = '<p class="load-error">The saved album could not be loaded. Reopen this demo using Open_Demo.command.</p>';
    return;
  }
  let roll = data.camera_roll;
  let photos = data.manifest.photos;
  let entities = Object.fromEntries([...roll.people, ...roll.objects].map(item => [item.id, item]));
  let refs = Object.fromEntries(data.manifest.references.map(item => [item.entity_id, item]));
  let albumId = 'maya';
  let firstName = 'Maya';
  const months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
  // Editorial album titles summarize the saved fictional story, not a new generation.
  let titles = ['After hours', 'Around the table', 'A slow afternoon', 'Make a wish', 'Back in the kitchen'];
  let locations = ['Architecture studio', 'Fishtown apartment', 'Along the Schuylkill', 'Penn Treaty Park', 'The apartment kitchen'];
  const steps = ['The brief', 'The story', 'Scene & cast', 'Reference images', 'Generation passes', 'Scores & results'];
  const headings = ['It starts with a few details.', 'A year begins to take shape.', 'Turn the story into a moment.', 'Give familiar things a face.', 'Build the photograph, pass by pass.', 'Face consistency scores.'];
  const descriptions = [
    'The original biography is the creative brief. Its details anchor the fictional life.',
    'The first language-model stage expands the brief into a background and a yearly narrative.',
    'The next text stage extracts dated photo ideas and shared people and object catalogs together.',
    'A prompt compiler writes an image prompt for each reference. Those images guide recurring appearances.',
    'These are the actual saved image-generation records. Switch passes to inspect their inputs and prompts.',
    'Inspect the saved face comparisons, then use human ratings to judge whether the identity holds.'
  ];
  const dialog = document.getElementById('detail-dialog');
  const content = document.getElementById('dialog-content');
  let state = { mode: '', scene: 0, step: 0, pass: 1, compare: false };
  let opener = null;
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[c]));
  const image = path => `${data.image_base || 'assets/images/'}${path}`;
  const evaluation = path => `${data.evaluation_base || 'assets/evaluation/'}${path}`;
  const month = index => months[roll.photos[index].month - 1];
  const date = index => `${month(index)} · Year ${roll.photos[index].year}`;
  const num = index => String(index + 1).padStart(2, '0');
  const prompt = (label, text, note = '') => `<details class="prompt-detail"><summary>${esc(label)}</summary>${note ? `<p class="small-note">${esc(note)}</p>` : ''}<pre>${esc(text)}</pre></details>`;
  const template = name => prompt('View the source prompt template', data.prompt_templates[name], 'Current repository template, not a retained transcript of the historical API request.');
  const closeButton = '<button class="icon-button" data-action="close" aria-label="Close and return to camera roll">×</button>';
  const readyPhoto = photo => photo.final_available ?? photo.passes.every(p => p.completed);
  const thumbnail = (path, alt, ready = true, css = '') => ready ? `<img class="${css}" src="${image(path)}" alt="${esc(alt)}" width="1024" height="1024">` : `<div class="empty-photo ${css}" role="img" aria-label="${esc(alt)} not generated yet"><span aria-hidden="true">◇</span><strong>Still to come</strong><span>${esc(alt)}</span></div>`;

  function renderAlbum() {
    const count = photos.length;
    const words = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five'];
    document.title = `A life in frames · ${firstName}'s camera roll`;
    document.getElementById('hero-title').innerHTML = `A year.<br>A life.<br><em>${words[count] || count} moment${count === 1 ? '' : 's'}.</em>`;
    document.querySelector('.hero-intro').textContent = albumId === 'maya' ? 'Meet Maya. Architecture student, weekend cyclist, keeper of a chipped blue mug. A year in Philadelphia, through the photos she might keep.' : data.story.seed.length > 220 ? data.story.seed.slice(0, 217) + '…' : data.story.seed;
    document.querySelector('.hero-actions .primary').innerHTML = `Open ${esc(firstName)}’s camera roll <span aria-hidden="true">↓</span>`;
    document.querySelector('.hero-actions .text-button').innerHTML = 'Read the story <span aria-hidden="true">↗</span>';
    document.querySelector('.hero-caption').innerHTML = `<span class="tiny-rule"></span>${albumId === 'maya' ? 'The original saved example from our visual consistency project.' : data.run?.status === 'completed' ? 'A new life, saved to your collection.' : 'The story is saved. Open the creation panel to check photo progress.'}`;
    document.getElementById('album-title').innerHTML = `${esc(firstName)}'s camera roll<span class="accent">.</span>`;
    document.querySelector('.section-heading .eyebrow').textContent = albumId === 'maya' ? 'Philadelphia · Year 1' : 'A fictional life · Year 1';
    document.querySelector('.section-heading .muted').textContent = 'Open a photo for its story and generation passes. Use the tabs above to explore the pipeline or rate the photos.';
    document.querySelector('.album-count').textContent = `${data.story.years.length} year / ${count} photograph${count === 1 ? '' : 's'}`;
    document.querySelector('.album-end p').textContent = `${words[count] || count} photograph${count === 1 ? '' : 's'}. One unfolding story.`;
    const stack = albumId === 'maya' ? [0, 3, 2] : photos.map((_, i) => i).filter(i => readyPhoto(photos[i])).slice(0, 3);
    document.getElementById('photo-stack').setAttribute('aria-label', `A preview of ${firstName}'s year`);
    document.getElementById('photo-stack').innerHTML = stack.length ? stack.map(i => `<button class="stack-photo" data-action="photo" data-scene="${i}" aria-label="Open ${month(i)} photo">${thumbnail(photos[i].final_path, titles[i])}<span>${month(i)}. ${esc(titles[i])}</span></button>`).join('') + '<span class="stack-note">the little things, remembered.</span>' : `<div class="album-placeholder"><p class="eyebrow">A new story begins</p><h3>${esc(firstName)}'s year<br>is taking shape.</h3><p>The story and scene plan are saved. Open the creation panel when you are ready to make the photographs.</p><button class="button primary" data-create-action="run" data-run="${esc(albumId)}">Continue this story →</button></div>`;
    document.getElementById('month-index').innerHTML = photos.map((_, i) => `<a href="#frame-${i}">${month(i)}</a>`).join('');
    document.getElementById('album-grid').innerHTML = photos.map((photo, i) => `<article class="photo-card" id="frame-${i}"><button data-action="photo" data-scene="${i}" aria-label="Open ${month(i)}: ${esc(titles[i])}"><div class="photo-wrap">${thumbnail(photo.final_path, titles[i], readyPhoto(photo))}<span class="photo-open" aria-hidden="true">↗</span></div><div class="photo-meta"><span>${date(i)}</span><span class="frame-number">Frame ${num(i)}</span></div><h3>${esc(titles[i])}</h3></button></article>`).join('');
  }
  renderAlbum();

  function setRoute(mode, scene = state.scene, step = 0) {
    const hash = mode === 'process' ? `#process/${scene}/${step}` : mode === 'photo' ? `#photo-${scene}` : `#${mode}`;
    if (location.hash !== hash) history.pushState(null, '', hash);
    readRoute();
  }

  function closeView(updateUrl = true) {
    if (dialog.open) dialog.close();
    document.body.classList.remove('dialog-open');
    state.mode = '';
    if (updateUrl) history.replaceState(null, '', '#album');
    if (opener?.isConnected) opener.focus({ preventScroll: true });
  }

  function readRoute() {
    const processMatch = location.hash.match(/^#process\/(\d+)\/(\d+)$/);
    const photoMatch = location.hash.match(/^#photo-(\d+)$/);
    let mode = '', scene = state.scene, step = state.step;
    if (processMatch) { mode = 'process'; scene = Number(processMatch[1]); step = Number(processMatch[2]); }
    else if (photoMatch) { mode = 'photo'; scene = Number(photoMatch[1]); }
    else if (location.hash === '#story') mode = 'story';
    else if (location.hash === '#about') mode = 'about';
    if (!mode || scene >= photos.length || step >= steps.length) { closeView(false); return; }
    if (!dialog.open) opener = document.activeElement;
    state = { mode, scene, step, pass: photos[scene].passes.length - 1, compare: false };
    renderDialog();
  }

  function renderDialog() {
    dialog.className = `${state.mode === 'about' ? 'story' : state.mode}-dialog`;
    content.innerHTML = state.mode === 'photo' ? photoView() : state.mode === 'story' ? storyView() : state.mode === 'about' ? aboutView() : processView();
    dialog.scrollTop = 0;
    document.body.classList.add('dialog-open');
    if (!dialog.open) dialog.showModal();
    (content.querySelector('.stage-title') || content.querySelector('#dialog-title')).focus({ preventScroll: true });
    content.querySelector('[aria-current="step"]')?.scrollIntoView({ block: 'nearest', inline: 'nearest', behavior: 'instant' });
  }

  function photoView() {
    const i = state.scene, photo = roll.photos[i];
    return `<div class="dialog-top"><p class="eyebrow">${esc(firstName)}'s camera roll · ${esc(locations[i])}</p>${closeButton}</div><div class="photo-layout"><div class="moment-image">${thumbnail(photos[i].final_path, titles[i], readyPhoto(photos[i]))}<span class="image-note">${date(i)}</span></div><div class="moment-info"><p class="eyebrow">Frame ${num(i)} / ${num(photos.length - 1)}</p><h2 id="dialog-title" tabindex="-1">${esc(titles[i])}</h2><p class="story-caption">${esc(photo.story_context)}</p><div class="people-line"><div class="avatar-stack">${photo.people.map(p => `${refs[p.person_id].image.completed ? `<img src="${image(refs[p.person_id].image.path)}" alt="${esc(entities[p.person_id].name)} reference" width="34" height="34">` : `<span class="pending-reference">${esc(entities[p.person_id].name[0])}</span>`}`).join('')}</div><span>From the story: ${photo.people.map(p => esc(entities[p.person_id].name.split(' ')[0])).join(', ')}</span></div><details><summary>The scene imagined in the story</summary><p class="small-note">${esc(photo.description)}</p></details><button class="button primary" data-action="inspect" data-scene="${i}">See how this photo was made <span aria-hidden="true">↗</span></button><div class="moment-controls"><span>${date(i)}</span><div><button class="icon-button" data-action="previous-photo" aria-label="Previous photograph" ${i === 0 ? 'disabled' : ''}>←</button><button class="icon-button" data-action="next-photo" aria-label="Next photograph" ${i === photos.length - 1 ? 'disabled' : ''}>→</button></div></div></div></div>`;
  }

  function storyView() {
    return `<div class="dialog-top"><p class="eyebrow">Meet the person behind the photographs</p>${closeButton}</div><div class="story-content"><p class="eyebrow">A fictional life · Year 1</p><h2 id="dialog-title" tabindex="-1">This is ${esc(firstName)}.</h2><p class="lead">${esc(data.story.seed)}</p><div class="rule"></div><h3>Their world</h3><p class="narrative">${esc(data.story.background)}</p>${data.story.years.map(y => `<h3>Year ${y.year}</h3><p class="narrative">${esc(y.narrative)}</p>`).join('')}<div class="button-row"><button class="button primary" data-action="album">Return to the camera roll ↓</button><button class="button secondary" data-action="process">Open pipeline →</button></div></div>`;
  }

  function aboutView() {
    return `<div class="dialog-top"><p class="eyebrow">CIS 5810 · Local project</p>${closeButton}</div><div class="story-content about-content"><h2 id="dialog-title" tabindex="-1">About this album.</h2><p>This fictional camera roll is part of <strong>Personalized Photo Album Generation with Visual Consistency</strong>.</p><p>${photos.length} planned photographs, ${data.manifest.references.length} reference assets and ${photos.reduce((n,p)=>n+p.passes.length,0)} scene passes. Copies and resized inputs are not additional generations.</p><p>${esc(data.provenance.attribution)}</p><p>Your albums and credentials stay on this computer. When you choose to create a story or photographs, the server sends the needed prompts and reference images to the configured model providers. Opening a saved album makes no generation calls.</p><p>Face scores, where available, are exploratory similarities, not identity probabilities. Assistant annotations are not independent team ratings. The proposed score-triggered corrective retry is not implemented.</p><p><a href="${esc(data.provenance.source_repository)}" target="_blank" rel="noopener">Original project ↗</a> · source commit <code>${esc(data.provenance.source_commit.slice(0,7))}</code></p><p>Story-first presentation inspired by <a href="https://1in8b.life/en/" target="_blank" rel="noopener">1 in 8 Billion ↗</a>.</p>${prompt('Saved generation settings',JSON.stringify(data.manifest.settings,null,2))}</div>`;
  }

  function requestLog() {
    if (!data.trace?.length) return '';
    return `<details class="trace-summary"><summary>Recorded requests for this album · ${data.trace.length}</summary><p class="small-note">Actual requests captured by this local run, including the schema sent to the text model. Provider credentials are not included.</p>${data.trace.map((entry,i)=>`<details><summary>${i+1}. ${esc(entry.stage)} · ${esc(entry.status)}</summary><pre>${esc(JSON.stringify(entry,null,2))}</pre></details>`).join('')}</details>`;
  }

  function sceneSelector() {
    return `<label class="scene-select">Selected moment<select id="scene-select" aria-label="Choose a moment to inspect">${photos.map((_, i) => `<option value="${i}" ${i === state.scene ? 'selected' : ''}>${month(i)} — ${esc(titles[i])}</option>`).join('')}</select></label>`;
  }

  function processView() {
    const stage = [briefStage, storyStage, sceneStage, referenceStage, passStage, reviewStage][state.step]();
    return `<div class="process-shell"><header class="process-top"><div class="process-title-group"><span class="workspace-album">${esc(firstName)}’s album</span><h2 id="dialog-title" tabindex="-1">Pipeline</h2></div>${window.workspaceNav.render('pipeline')}<div class="process-top-right">${albumId !== 'maya' ? '<button class="text-button" data-create-action="refresh-album">Refresh ↻</button>' : ''}<button class="back-to-photos" data-workspace-view="photos">← Photos</button></div></header><div class="process-body"><aside class="process-sidebar"><p class="eyebrow">Follow the process</p><nav class="step-navigation" aria-label="Pipeline steps">${steps.map((name, i) => `<button class="step-button" data-action="step" data-step="${i}" ${i === state.step ? 'aria-current="step"' : ''}><span class="step-number">${num(i)}</span>${name}</button>`).join('')}</nav><p class="sidebar-note">One biography<br>${photos.filter(readyPhoto).length} / ${photos.length} photos complete<br>All artifacts saved locally</p></aside><main class="process-main"><header class="stage-header"><div><p class="eyebrow">${num(state.step)} / ${num(steps.length - 1)} · ${steps[state.step]}</p><h2 class="stage-title" tabindex="-1">${headings[state.step]}</h2><p class="stage-description">${state.step === 5 && !data.scores ? 'Review this album’s saved photographs and passes. Face scores have not been added. Use Rate photos for independent human reviews.' : descriptions[state.step]}</p></div>${state.step >= 2 ? sceneSelector() : ''}</header>${stage}${requestLog()}<footer class="stage-footer"><button data-action="step" data-step="${state.step - 1}" ${state.step === 0 ? 'disabled' : ''}>← Previous step</button>${state.step < steps.length - 1 ? `<button class="next-step" data-action="step" data-step="${state.step + 1}">Next: ${steps[state.step + 1]} →</button>` : '<button class="next-step" data-action="close">Return to camera roll ↗</button>'}</footer></main></div></div>`;
  }

  function briefStage() {
    return `<div class="brief-grid"><div><section class="panel"><p class="eyebrow">Original user brief · saved input</p><p class="brief-quote">“${esc(data.story.seed)}”</p><div class="facts"><div class="fact"><strong>${data.story.years.length}</strong>year requested</div><div class="fact"><strong>${photos.length}</strong>photos planned</div><div class="fact"><strong>${esc(firstName)}</strong>the protagonist</div></div></section><div class="method-note">The brief supplies the person and a few anchors. The language model fills in a plausible fictional life, including relationships and recurring possessions.</div></div><figure class="brief-image">${thumbnail(refs[roll.protagonist_id].image.path, `${firstName} reference portrait`, refs[roll.protagonist_id].image.completed)}<figcaption>${esc(firstName)}, ${refs[roll.protagonist_id].image.completed ? "as imagined by the image model" : "waiting for a reference portrait"}.</figcaption></figure></div>`;
  }

  function storyStage() {
    return `<section class="panel stage-story"><p class="eyebrow">Saved output · story generation</p><h3>A person, before a photograph.</h3><p>${esc(data.story.background)}</p>${data.story.years.map(y => `<h3>Year ${y.year}</h3><p>${esc(y.narrative)}</p>`).join('')}</section><div class="method-note">This stage creates the narrative. A separate text stage turns it into dated scenes and shared catalogs; image-prompt compilation follows afterward.</div>${template('story')}`;
  }

  function castItem(appearance, kind) {
    const id = appearance[kind === 'people' ? 'person_id' : 'object_id'];
    const item = entities[id];
    return `<article class="cast-item">${refs[id].image.completed ? `<img src="${image(refs[id].image.path)}" alt="${esc(item.name)} reference" width="65" height="65">` : `<span class="pending-reference">${esc(item.name[0])}</span>`}<div><strong>${esc(item.name)}</strong><p>${esc(appearance.appearance)}</p><details><summary>Shared identity record</summary><p>${esc(item.visual_description)}</p><code>${esc(id)}</code></details></div></article>`;
  }

  function sceneStage() {
    const photo = roll.photos[state.scene];
    return `<section class="panel"><p class="eyebrow">${date(state.scene)} · Requested scene</p><p class="scene-request">${esc(photo.description)}</p><p class="context-line"><strong>Story context:</strong> ${esc(photo.story_context)}</p></section><h3 class="cast-heading">PEOPLE IN THIS SCENE · ${photo.people.length}</h3><div class="cast-list">${photo.people.map(p => castItem(p, 'people')).join('')}</div><h3 class="cast-heading">RECURRING OBJECTS · ${photo.objects.length}</h3><div class="cast-list">${photo.objects.map(o => castItem(o, 'objects')).join('')}</div><div class="method-note">Stable IDs connect each moment to a shared character or object. These records check structure and continuity in the plan; they do not prove that the generated image obeyed it.</div>${template('photos')}${prompt('Inspect the saved scene record', JSON.stringify(photo, null, 2))}`;
  }

  function referenceStage() {
    const photo = roll.photos[state.scene];
    const ids = [...photo.people.map(p => p.person_id), ...photo.objects.map(o => o.object_id)];
    return `<p class="small-note">Showing the ${ids.length} reusable references needed for ${month(state.scene)}. The full album has ${roll.people.length} character references and ${roll.objects.length} object references. Originals are 1024 × 1024; saved conditioning copies are 448 × 448.</p><div class="references-grid">${ids.map(id => { const ref = refs[id]; return `<article class="reference-card">${thumbnail(ref.image.path, `${entities[id].name} canonical reference`, ref.image.completed)}<div class="reference-info"><h3>${esc(entities[id].name)}</h3><p class="small-note">${ref.kind === 'people' ? 'Character reference' : 'Object reference'}</p>${ref.image.prompt ? prompt('Saved reference prompt', ref.image.prompt) : '<p class="small-note">The prompt is written when this reference begins.</p>'}<div class="seed">Seed ${ref.image.seed}</div></div></article>`; }).join('')}</div><div class="method-note">These references are themselves generated images. They provide a consistent target, not verified ground truth. The next step shows the saved source file for each input slot. The image adapter resizes every input, including previous scenes, onto a 448 × 448 canvas before submission.</div><details><summary>How reference prompts are compiled</summary><p class="small-note">One language-model prompt-compilation call per reference asset. The resulting image prompt above is saved in the manifest. The source templates below are not full historical request logs.</p>${prompt('Character compiler template', data.prompt_templates.image_character)}${prompt('Object compiler template', data.prompt_templates.image_object)}</details>`;
  }

  function carriedIds(scene, passIndex) {
    const pass = scene.passes[passIndex];
    const inherited = pass.inputs.filter(input => !input.entity_id).flatMap(input => {
      const previous = scene.passes.findIndex(p => p.path === input.path);
      if (previous < 0 || previous >= passIndex) return [];
      return [...scene.passes[previous].inputs.filter(i => i.entity_id).map(i => i.entity_id), ...carriedIds(scene, previous)];
    });
    return [...new Set(inherited)].filter(id => !pass.inputs.some(input => input.entity_id === id));
  }

  function passStage() {
    const scene = photos[state.scene], pass = scene.passes[state.pass], last = scene.passes.length - 1;
    const carried = carriedIds(scene, state.pass);
    const label = index => `${scene.passes.length === 1 ? 'Single pass' : `Pass ${index + 1}`}${index === last ? ' · Final' : ''}`;
    const comparison = state.compare ? `<div class="comparison-grid">${scene.passes.map((p, i) => `<figure>${thumbnail(p.path, `${month(state.scene)} ${label(i)}`, p.completed)}<figcaption>${label(i)}</figcaption></figure>`).join('')}</div>` : '';
    return `<div class="pass-toolbar"><div class="segmented" aria-label="Saved generation passes">${scene.passes.map((_, i) => `<button data-action="pass" data-pass="${i}" aria-pressed="${i === state.pass}">${label(i)}</button>`).join('')}</div>${scene.passes.length > 1 ? `<button class="compare-button" data-action="compare" aria-pressed="${state.compare}">${state.compare ? 'Hide' : 'Compare'} ${scene.passes.length === 2 ? 'both' : 'all'} passes</button>` : '<span class="small-note">All references fit in the first pass.</span>'}</div>${comparison}<div class="pass-layout"><figure class="pass-output">${thumbnail(pass.path, `${date(state.scene)}, ${label(state.pass)}`, pass.completed)}<figcaption><span>${label(state.pass)} · 1024 × 1024</span>${pass.completed ? `<a href="${image(pass.path)}" target="_blank" rel="noopener">Open full size ↗</a>` : '<span>Not generated yet</span>'}</figcaption></figure><section class="inputs-panel"><h3>Direct inputs · ${pass.inputs.length} image slots</h3><p class="small-note">${state.pass === 0 ? 'Start with up to four character and object references.' : 'The previous scene plus up to three new references. Every input is resized to 448 × 448 before submission.'}</p><div class="input-grid">${[...pass.inputs].sort((a,b) => a.slot-b.slot).map(input => `<figure class="input-card">${thumbnail(input.path, `Input ${input.slot}: ${input.entity_id ? entities[input.entity_id].name : 'Previous scene'}`, input.available ?? (input.entity_id ? refs[input.entity_id].image.completed : scene.passes.some(p => p.path === input.path && p.completed)))}<figcaption><span class="slot">INPUT ${input.slot} · ${input.entity_id ? '448 × 448 REFERENCE' : 'PREVIOUS SCENE'}</span>${esc(input.entity_id ? entities[input.entity_id].name : `Scene from pass ${scene.passes.findIndex(p => p.path === input.path) + 1}`)}</figcaption></figure>`).join('')}</div>${carried.length ? `<div class="carried"><strong>Carried through the previous scene</strong>${carried.map(id => esc(entities[id].name)).join(' · ')}<br>These reference portraits/objects are not directly supplied again.</div>` : ''}<p class="seed">Seed ${pass.seed}<br>${esc(data.manifest.settings.image_model)}</p></section></div><details class="pass-prompt"><summary>Read the exact prompt for ${label(state.pass).toLowerCase()}</summary><pre>${esc(pass.prompt || 'This prompt has not been written yet. It will appear when this stage starts.')}</pre></details><div class="method-note">${scene.passes.length > 1 ? 'These passes accommodate the reference limit. The later pass edits the existing scene to add remaining references; it is not a score-triggered retry.' : 'This scene has one saved pass. There is no second pass or retry to compare.'}</div>${prompt('Scene prompt compiler template', data.prompt_templates.image_scene, 'One compiler call per photo writes its pass prompts. This current source template is not a historical API request transcript.')}`;
  }

  const scoreText = value => Number.isFinite(value) ? value.toFixed(4) : 'Not scored';
  const candidateScore = (pass, id) => pass.candidates.find(c => c.person_id === id)?.similarity ?? null;

  function candidateCell(pass, id) {
    const candidate = pass.candidates.find(c => c.person_id === id);
    const face = candidate?.face_index == null ? null : pass.faces.find(f => f.face_index === candidate.face_index);
    if (!face) return '<span class="score-unassigned">No face assigned</span><span class="small-note">This does not prove the person is absent.</span>';
    return `<img src="${evaluation(face.aligned_crop)}" alt="Assigned ${esc(entities[id].name)} candidate, face ${candidate.face_index}" width="112" height="112"><strong class="score-value">${scoreText(candidate.similarity)}</strong><span class="small-note">Assigned candidate · face ${candidate.face_index}</span>`;
  }

  function reviewStage() {
    const scene = data.scores?.scenes.find(s => s.photo_index === state.scene);
    if (!scene) return `<section class="panel"><p class="eyebrow">Model results</p><h3>No face scores saved for this photo.</h3><p>The original Maya album has saved SCRFD + ArcFace results. Automatic scoring for new albums is not connected yet.</p><p class="small-note">CLIP scene scoring is not implemented. Human evaluation is available separately in Rate photos.</p><button class="button secondary" data-workspace-view="ratings">Open human ratings →</button></section>`;
    const protagonist = roll.protagonist_id, reference = data.scores.references.find(r => r.id === protagonist);
    const final = scene.passes.at(-1), first = scene.passes[0], multi = scene.passes.length > 1;
    const firstScore = candidateScore(first, protagonist), finalScore = candidateScore(final, protagonist);
    const delta = multi && firstScore !== null && finalScore !== null ? finalScore - firstScore : null;
    const signed = value => `${value > 0 ? '+' : value < 0 ? '−' : ''}${Math.abs(value).toFixed(4)}`;
    const scenes = data.scores.scenes;
    const detected = scenes.reduce((sum, s) => sum + s.passes.at(-1).faces.length, 0);
    const requested = scenes.reduce((sum, s) => sum + s.expected_people.length, 0);
    const comparable = scenes.filter(s => s.passes.length > 1 && candidateScore(s.passes[0], protagonist) !== null && candidateScore(s.passes.at(-1), protagonist) !== null);
    const lower = comparable.filter(s => candidateScore(s.passes.at(-1), protagonist) < candidateScore(s.passes[0], protagonist)).length;
    const others = scene.expected_people.filter(id => id !== protagonist);
    const review = data.review?.scenes.find(s => s.photo_index === state.scene) || data.review?.scenes[state.scene];
    const rows = scenes.map(s => {
      const a = candidateScore(s.passes[0], protagonist), b = candidateScore(s.passes.at(-1), protagonist), multiple = s.passes.length > 1;
      return `<tr ${s.photo_index === state.scene ? 'class="score-selected"' : ''}><th scope="row"><button data-action="score-scene" data-scene="${s.photo_index}" ${s.photo_index === state.scene ? 'aria-current="true"' : ''}>${esc(months[s.month - 1])} ${s.photo_index === state.scene ? '<span class="small-note">· selected</span>' : '↗'}</button></th><td>${multiple ? scoreText(a) : '—'}</td><td>${scoreText(b)}</td><td>${!multiple ? 'Single pass' : a === null || b === null ? 'Not comparable' : signed(b - a)}</td><td>${s.passes.at(-1).faces.length} / ${s.expected_people.length}</td></tr>`;
    }).join('');
    return `<div class="score-summary"><div><strong>ArcFace</strong><span>Saved face similarity</span></div><div><strong>${scenes.length} photos · ${scenes.reduce((n,s)=>n+s.passes.length,0)} passes</strong><span>Computed locally on saved images</span></div><div><strong>${detected} faces detected / ${requested} people requested</strong><span>Final-photo detection counts, not identities</span></div></div>
      <section class="score-comparison" aria-label="Protagonist face comparison"><header><div><p class="eyebrow">${date(state.scene)} · Protagonist</p><h3>${esc(firstName)}’s face, across passes</h3></div><span class="score-status">No validated pass/fail cutoff</span></header><p class="score-explanation">Higher ArcFace cosine similarity means a closer model match to the reference face. Values range from −1 to 1. <strong>This is not a percentage or a probability.</strong></p>
      <div class="score-faces ${multi ? '' : 'single-pass'}"><figure><figcaption>Reference portrait</figcaption><img src="${evaluation(reference.aligned_crop)}" alt="${esc(firstName)} reference face" width="112" height="112"><strong>${esc(entities[protagonist].name)}</strong><span class="small-note">The comparison target</span></figure>${multi ? `<figure><figcaption>First pass</figcaption>${candidateCell(first,protagonist)}</figure>` : ''}<figure><figcaption>${multi ? 'Final pass' : 'Final · only pass'}</figcaption>${candidateCell(final,protagonist)}</figure></div>
      <div class="score-reading"><p>${!multi ? '<strong>Single-pass photo.</strong> There is no before/after edit to compare.' : delta === null ? '<strong>No comparable pair.</strong> A missing assignment stays unscored.' : `<strong>Change: ${signed(delta)}</strong> · ${delta < 0 ? 'Lower' : delta > 0 ? 'Higher' : 'Unchanged'} assigned similarity after the edit. Pose, visibility and face assignment can affect this result; it is not a human identity verdict.`}</p><button class="text-button" data-action="inspect" data-scene="${state.scene}">View full photos & passes →</button></div></section>
      <details class="score-details"><summary>How the face score is produced</summary><ol class="score-method"><li><strong>Find the faces · SCRFD</strong><p>Locate faces and facial landmarks in the photo and reference portraits.</p></li><li><strong>Compare the faces · ArcFace</strong><p>Align the crops, encode their facial features, then compare each scene face with each character reference using cosine similarity.</p></li><li><strong>Assign candidates</strong><p>Choose a one-to-one pairing. Even weak matches can be assigned, so a pairing alone does not confirm identity.</p></li></ol><p class="small-note">The saved detector confidence threshold is ${data.scores.settings.detection_threshold}. That controls face detection; it is not an ArcFace identity cutoff. Negative similarity values are retained. No score-triggered retry has run.</p></details>
      <section class="score-album"><h3>${esc(firstName)} across the album</h3><p class="small-note">Same reference portrait, all saved photos. Select a month to inspect its face crops. Change = final minus first, calculated before rounding.</p><div class="score-table"><table><caption>Protagonist ArcFace similarity by photo</caption><thead><tr><th>Photo</th><th>First pass</th><th>Final pass</th><th>Change</th><th>Faces found / requested</th></tr></thead><tbody>${rows}</tbody></table></div>${comparable.length ? `<p class="score-observation"><strong>${lower} of ${comparable.length} comparable multi-pass photos have a lower final score.</strong> This is a pattern to investigate with independent human ratings, not proof that the edits caused identity loss.</p>` : ''}</section>
      <details class="score-details"><summary>Other characters & face detections · ${final.faces.length} faces found / ${scene.expected_people.length} people requested</summary><p class="small-note">Assigned candidates are tentative. An unassigned character is not proof of who is absent. The independent identity rubric currently rates the protagonist only.</p>${others.length ? `<div class="review-table-wrap"><table><thead><tr><th>Reference</th>${multi ? '<th>First-pass candidate</th>' : ''}<th>Final candidate</th></tr></thead><tbody>${others.map(id => {const ref=data.scores.references.find(r=>r.id===id);return `<tr><td><img src="${evaluation(ref.aligned_crop)}" alt="${esc(entities[id].name)} reference" width="112" height="112"><strong>${esc(entities[id].name)}</strong></td>${multi?`<td>${candidateCell(first,id)}</td>`:''}<td>${candidateCell(final,id)}</td></tr>`;}).join('')}</tbody></table></div>` : ''}<img class="score-detections" src="${evaluation(final.detections_image)}" alt="${month(state.scene)} final photo with numbered face detections" width="1024" height="1024"></details>
      <section class="score-evidence"><div><h3>Human results are separate</h3><p>Independent majority judgments are the primary evaluation. ArcFace supports the analysis; face counts do not measure identity accuracy.</p><button class="text-button" data-workspace-view="ratings">Open human ratings →</button></div><div><h3>CLIP scene alignment</h3><p><strong>Not implemented.</strong> There is no saved CLIP score for this album. Human setting and action ratings are available through Rate photos.</p></div></section>
      ${review ? prompt('Assistant observations · not independent human ratings', JSON.stringify(review,null,2)) : ''}${prompt('Technical evidence · similarity matrices and assignments', JSON.stringify(scene.passes.map(p=>({pass:p.pass_index+1,rows:data.scores.matrix_row_order,cosine_matrix:p.cosine_matrix,candidates:p.candidates})),null,2))}`;
  }

  function changeScene(index) {
    if (index < 0 || index >= photos.length) return;
    state.scene = index;
    state.pass = photos[index].passes.length - 1;
    state.compare = false;
    history.replaceState(null, '', state.mode === 'process' ? `#process/${index}/${state.step}` : `#photo-${index}`);
    renderDialog();
  }

  document.addEventListener('click', event => {
    const button = event.target.closest('[data-action]');
    if (!button || button.disabled) return;
    const action = button.dataset.action;
    if (action === 'photo') setRoute('photo', Number(button.dataset.scene));
    if (action === 'story' || action === 'about') setRoute(action);
    if (action === 'process') setRoute('process', 0, 0);
    if (action === 'inspect') setRoute('process', Number(button.dataset.scene), 4);
    if (action === 'close' || action === 'album') {
      closeView();
      if (action === 'album') document.getElementById('album').scrollIntoView();
    }
    if (action === 'previous-photo') changeScene(state.scene - 1);
    if (action === 'next-photo') changeScene(state.scene + 1);
    if (action === 'score-scene') changeScene(Number(button.dataset.scene));
    if (action === 'step') {
      const step = Number(button.dataset.step);
      if (step < 0 || step >= steps.length) return;
      state.step = step;
      state.compare = false;
      history.replaceState(null, '', `#process/${state.scene}/${step}`);
      renderDialog();
    }
    if (action === 'pass') { state.pass = Number(button.dataset.pass); renderDialog(); }
    if (action === 'compare') { state.compare = !state.compare; renderDialog(); }
  });
  content.addEventListener('change', event => {
    if (event.target.id === 'scene-select') changeScene(Number(event.target.value));
  });
  dialog.addEventListener('cancel', event => { event.preventDefault(); closeView(); });
  dialog.addEventListener('click', event => { if (event.target === dialog) closeView(); });
  document.addEventListener('keydown', event => {
    if (!dialog.open) return;
    if (event.key === 'Tab') {
      const controls = [...dialog.querySelectorAll('button:not(:disabled), a[href], select, summary')].filter(el => el.getClientRects().length);
      if (event.shiftKey && document.activeElement === controls[0]) {
        event.preventDefault(); controls.at(-1).focus();
      } else if (!event.shiftKey && document.activeElement === controls.at(-1)) {
        event.preventDefault(); controls[0].focus();
      }
    }
    if (state.mode !== 'photo' || /INPUT|TEXTAREA|SELECT/.test(event.target.tagName)) return;
    if (event.key === 'ArrowLeft') { event.preventDefault(); changeScene(state.scene - 1); }
    if (event.key === 'ArrowRight') { event.preventDefault(); changeScene(state.scene + 1); }
  });
  window.addEventListener('hashchange', readRoute);
  window.albumApp = {
    load(next, id = 'maya', preserveRoute = false) {
      const savedHash = location.hash;
      closeView(false);
      data = next;
      albumId = id;
      roll = data.camera_roll;
      photos = data.manifest.photos;
      entities = Object.fromEntries([...roll.people, ...roll.objects].map(item => [item.id, item]));
      refs = Object.fromEntries(data.manifest.references.map(item => [item.entity_id, item]));
      firstName = entities[roll.protagonist_id].name.split(' ')[0];
      titles = id === 'maya' ? ['After hours', 'Around the table', 'A slow afternoon', 'Make a wish', 'Back in the kitchen'] : roll.photos.map(p => `${p.description.split(/[.!?]/)[0].split(/\s+/).slice(0,8).join(' ')}${p.description.split(/[.!?]/)[0].split(/\s+/).length > 8 ? '…' : ''}`);
      locations = id === 'maya' ? ['Architecture studio', 'Fishtown apartment', 'Along the Schuylkill', 'Penn Treaty Park', 'The apartment kitchen'] : roll.photos.map(p => `Year ${p.year}`);
      state = {mode:'',scene:0,step:0,pass:photos[0].passes.length-1,compare:false};
      const url = new URL(location.href);
      if (id === 'maya') url.searchParams.delete('album'); else url.searchParams.set('album',id);
      url.hash = preserveRoute ? savedHash : 'album';
      history.replaceState(null,'',url);
      renderAlbum();
      if (preserveRoute) readRoute();
      window.dispatchEvent(new CustomEvent('album-loaded',{detail:{id}}));
    },
    close() { closeView(false); },
    inspect(step = 0) { setRoute('process',0,step); },
    currentId() { return albumId; }
  };
  if (!new URL(location.href).searchParams.has('album')) readRoute();
})();
