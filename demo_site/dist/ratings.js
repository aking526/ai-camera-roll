'use strict';

(() => {
  const dialog = document.getElementById('ratings-dialog');
  const content = document.getElementById('ratings-content');
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[c]));
  const labels = {same:'Same person',different:'Different',unclear:'Unclear',unresolved:'No majority',not_visible:'No identifiable candidate',yes:'Yes',partial:'Partly',no:'No',present:'Present',absent:'Absent',matches:'Matches reference',correct:'Correct',wrong:'Wrong',not_applicable:'Not applicable'};
  let token = '', overview = {rounds:[]}, bundle = null, round = null, results = null;
  let view = 'home', index = 0, section = 0, busy = false, error = '', opener = null, lastSession = null;
  let timer = null, saving = null, revision = 0;
  const dirty = new Map();
  const sectionNames = ['People', 'Objects', 'Scene'];
  const label = value => value === null ? 'Cannot confidently count' : (value === undefined || value === '' ? 'Not answered' : labels[value] ?? String(value));

  async function api(path, body, refreshed = false) {
    let response;
    try { response = await fetch(path, body === undefined ? {} : {method:'POST',headers:{'Content-Type':'application/json','X-Demo-Token':token},body:JSON.stringify(body)}); }
    catch { throw new Error('The local server is unavailable. Keep your answers open and restart Open_Demo.command, then try saving again.'); }
    const value = await response.json().catch(()=>({}));
    if (response.status === 403 && value.code === 'token' && body !== undefined && !refreshed) {
      token = (await api('/api/config')).csrf_token;
      return api(path,body,true);
    }
    if (!response.ok) throw new Error(value.error || 'The review could not be saved. Your current answers remain in this panel.');
    return value;
  }

  function remember(id) { try { localStorage.setItem('camera-roll-review-session', id); } catch {} }
  async function refresh() {
    token = (await api('/api/config')).csrf_token;
    overview = await api('/api/ratings');
    lastSession = null;
    try {
      const id = localStorage.getItem('camera-roll-review-session');
      if (id) lastSession = (await api(`/api/ratings/sessions/${id}`)).session;
    } catch {}
  }
  async function open() {
    if (!dialog.open) opener = document.activeElement;
    error = '';
    try { await refresh(); } catch (failure) { error = failure.message; }
    view = bundle?.session.status === 'draft' && dirty.size ? 'rate' : 'home';
    render();
    if (!dialog.open) dialog.showModal();
    document.body.classList.add('dialog-open');
    if (location.hash !== '#ratings') history.pushState(null,'','#ratings');
    content.querySelector('#rating-title').focus({preventScroll:true});
  }
  async function close() {
    try { await saveDirty(); } catch { return false; }
    dialog.close();
    document.body.classList.toggle('dialog-open',Boolean(document.querySelector('dialog[open]')));
    if (location.hash === '#ratings') history.replaceState(null,'','#album');
    if (!document.querySelector('dialog[open]') && opener?.isConnected) opener.focus({preventScroll:true});
    return true;
  }

  function guide() {
    return `<details class="rating-guide"><summary>How to rate</summary><ol><li><strong>Judge only what is visible.</strong> The requested scene tells you what should be there. A plausible story is not evidence that the image followed it.</li><li><strong>Identity:</strong> compare facial features with the portrait. Pose, expression, clothing and a hat can change without changing identity. Choose “Different” only when a visible likely protagonist has clearly different facial features. Choose “Unclear” for a small, obscured or ambiguous face; choose “No identifiable candidate” when you cannot locate a plausible protagonist.</li><li><strong>People:</strong> count visible people independently of who they are. If overlapping or ambiguous figures prevent a count, mark that explicitly.</li><li><strong>Objects:</strong> first identify whether a recognizable candidate is present. Count visible instances of that required object type; extras count as duplicates. Then compare its appearance with the reference, and its placement, ownership or state with the request. A visible object can have the wrong appearance or be in the wrong place. Use “Partly” when some requested state details are met and some are not; use “Unclear” when you cannot tell.</li><li><strong>Setting and action:</strong> judge the location and main activity separately. “Partly” means some requested elements are satisfied. “Unclear” means the image does not provide enough evidence.</li><li><strong>Work independently.</strong> Do not open the existing evaluation, model scores, or another person's ratings. Use one name per actual reviewer; submitting under several names does not create independent reviews.</li></ol><p>There are no default answers. Drafts save locally as you answer. You can pause and resume; final submission locks your responses. Scores, previous annotations and generation-method labels are hidden here, but Maya is familiar development material, not a fully blinded or held-out test.</p></details>`;
  }

  function roadmap() {
    return `<details class="rating-guide rating-evaluation-plan"><summary>Evaluation plan · three iterations</summary><p class="eyebrow">Three evidence-driven iterations · beyond the proposal</p><div class="rating-roadmap"><article><span class="status-label">1 · Measurement starting now</span><h3>Find a specific failure.</h3><p>Rate the frozen Maya baseline. Choose one frequent, well-supported defect, change one relevant part of the pipeline, then rate its new outputs with the same rubric.</p></article><article><span class="status-label">2 · Planned</span><h3>Test one bounded retry.</h3><p>Calibrate a face-score trigger on development labels. Compare the same initial outputs with and without one permitted retry. Measure human judgments and all added calls.</p></article><article><span class="status-label">3 · Evidence will choose it</span><h3>Address what remains.</h3><p>Choose one remaining issue, such as face preservation during edits or object-state instructions. Make one focused change and run another paired evaluation.</p></article></div><p class="rating-note"><strong>Started does not mean completed.</strong> Building this workflow starts the baseline measurement for iteration 1. Each completed iteration needs a hypothesis, a change, comparable before/after evidence, and a decision. Three reviewers or three metrics do not count as three iterations. The saved course guidance calls for three iterations beyond the proposal by November 20, then three further iterations beyond the midterm report.</p></details>`;
  }

  function roundCard(r) {
    const mine = lastSession?.round_id === r.id ? lastSession : null;
    const continuing = mine?.status === 'draft';
    return `<article class="rating-round rating-round-card"><div><p class="eyebrow">${r.item_count} photos · saved evaluation set</p><h3>${esc(r.label)}</h3><p class="rating-team-progress"><strong>${r.submitted_count} of ${r.min_reviewers}</strong> reviewers finished${r.submitted_count > r.min_reviewers ? ' (minimum reached)' : ''}</p><progress max="${Math.max(r.min_reviewers,r.submitted_count)}" value="${r.submitted_count}" aria-label="Submitted reviews"></progress><p class="small-note">${r.results_available ? 'Team results are ready. Finish your own review before opening them.' : `Results appear after ${r.min_reviewers} different people submit. Each person rates all ${r.item_count} photos independently.`}${r.draft_count ? ` ${r.draft_count} review${r.draft_count === 1 ? '' : 's'} in progress.` : ''}</p></div><div class="rating-round-actions"><button class="button primary" data-rating-action="${continuing?'resume':'join'}" ${continuing?`data-session="${mine.id}"`:`data-round="${r.id}"`}>${continuing?'Continue my review':'Start my review'} →</button>${continuing ? `<span class="small-note">Saved draft for ${esc(mine.reviewer)}</span><button class="text-button" data-rating-action="join" data-round="${r.id}">Review as someone else</button>` : mine ? `<button class="text-button" data-rating-action="resume" data-session="${mine.id}">View ${esc(mine.reviewer)}’s submitted review</button>` : ''}${r.results_available ? `<button class="button secondary" data-rating-action="results" data-round="${r.id}">View team results</button>` : ''}</div></article>`;
  }

  function homeView() {
    const selected = window.albumApp.currentId();
    const current = overview.rounds.find(r=>r.album_id===selected);
    const other = overview.rounds.filter(r=>r.id!==current?.id);
    return `<div class="rating-home"><p class="rating-intro">Compare each photo with its references. Your independent answers help us decide what to improve.</p>${current ? roundCard(current) : `<article class="rating-round"><h3>Rate this album</h3><p>Create a fixed set of this album’s final photos. No new images will be generated.</p><button class="button primary" data-rating-action="create" data-album="${esc(selected)}">Set up a rating round →</button></article>`}<div class="rating-quick-guide"><span><strong>1.</strong> Enter your name</span><span><strong>2.</strong> Rate each photo</span><span><strong>3.</strong> Check & submit</span></div><p class="small-note">Drafts save automatically. Use <strong>Unclear</strong> when you cannot judge confidently. Your answers stay editable until you submit.</p>${guide()}${other.length ? `<details class="rating-guide"><summary>Other saved rounds (${other.length})</summary>${other.map(roundCard).join('')}</details>` : ''}${roadmap()}</div>`;
  }

  function joinView() {
    return `<div class="rating-join"><button class="text-button" data-rating-action="home">← Rating overview</button><p class="rating-intro">${esc(round.label)} · ${round.item_count} photos</p><form id="rating-join-form" class="rating-form"><div class="rating-field"><label for="rating-reviewer">Your name or initials</label><input id="rating-reviewer" name="reviewer" maxlength="60" required autocomplete="off" placeholder="For example, Ethan"><p class="field-help">Use the same name to resume a saved review.</p></div><fieldset><legend>Have you seen these photos before?</legend><p class="small-note">Include seeing their review notes or scores. Either answer is fine.</p><div class="rating-options"><label><input type="radio" name="exposure" value="yes" required> Yes</label><label><input type="radio" name="exposure" value="no" required> No</label></div></fieldset><label class="rating-independence"><input type="checkbox" name="independent" required> I will rate independently, without consulting previous evaluations or other reviewers.</label><div class="rating-actions"><button class="button primary" type="submit">Begin my review →</button></div></form>${guide()}</div>`;
  }

  const choices = (name, title, values, selected, help='') => `<fieldset><legend>${title}</legend>${help?`<p class="small-note">${help}</p>`:''}<div class="rating-options">${values.map(value=>`<label><input type="radio" name="${name}" value="${value}" ${selected===value?'checked':''} required> ${esc(label(value))}</label>`).join('')}</div></fieldset>`;
  const select = (id, title, values, selected, attrs='') => `<div class="rating-field"><label for="${id}">${title}</label><select id="${id}" ${attrs}><option value="">Choose…</option>${values.map(value=>`<option value="${value}" ${selected===value?'selected':''}>${esc(label(value))}</option>`).join('')}</select></div>`;
  function complete(answer, item) {
    return answer && (item.protagonist_required === false || answer.identity) && Object.hasOwn(answer,'people_count') && answer.setting && answer.action && item.objects.every(o=>{
      const a=answer.objects?.[o.id];return a?.presence && (a.presence!=='present'||(a.count>=1&&a.appearance&&a.state));
    });
  }
  function objectView(object, i, answer) {
    const a=answer||{}, enabled=a.presence==='present';
    return `<fieldset class="rating-object" data-object="${esc(object.id)}"><legend>${esc(object.name)}</legend><img src="${esc(object.image_url)}" alt="${esc(object.name)} reference"><div><p>${esc(object.description)}</p><p><strong>Requested appearance or state:</strong> ${esc(object.expected_state)}</p></div><div class="rating-object-fields">${select(`object-${i}-presence`,'Can you see this object?',['present','absent','unclear'],a.presence,'data-field="presence" required')}<div class="rating-field"><label for="object-${i}-count">Visible instances</label><input id="object-${i}-count" type="number" min="1" max="20" step="1" data-field="count" value="${enabled?esc(a.count):''}" ${enabled?'required':'disabled'}></div>${select(`object-${i}-appearance`,'Appearance vs reference',['matches','different','unclear'],a.appearance,`data-field="appearance" ${enabled?'required':'disabled'}`)}${select(`object-${i}-state`,'Placement, ownership & state',['correct','partial','wrong','unclear'],a.state,`data-field="state" ${enabled?'required':'disabled'}`)}</div><p class="small-note rating-object-help">${a.presence==='absent'?'Absent means zero visible instances; appearance and state are not applicable.':a.presence==='unclear'?'When presence is unclear, count, appearance and state remain unassessed.':'Judge presence first, then count, appearance and state separately.'}</p></fieldset>`;
  }

  function firstIncompleteSection(answer, item) {
    const a=answer||{};
    if ((item.protagonist_required !== false && !a.identity) || !Object.hasOwn(a,'people_count')) return 0;
    if (!item.objects.every(o=>{const v=a.objects?.[o.id];return v?.presence && (v.presence!=='present'||(v.count>=1&&v.appearance&&v.state));})) return 1;
    return 2;
  }

  function rateView() {
    const item=bundle.items[index], answer=bundle.answers[item.id]||{};
    return `<div class="rating-progress" aria-label="Photograph progress"><span>${esc(bundle.session.reviewer)} · <strong>Photo ${index+1} of ${bundle.items.length}</strong></span><div>${bundle.items.map((p,i)=>`<button data-rating-action="item" data-index="${i}" ${i===index?'aria-current="step"':''} aria-label="Review photograph ${i+1}${complete(bundle.answers[p.id],p)?', complete':''}">${i+1}${complete(bundle.answers[p.id],p)?' ✓':''}</button>`).join('')}</div></div><div class="rating-layout"><aside class="rating-image-panel"><figure><img src="${esc(item.image_url)}" alt="Photograph being rated"><figcaption><a href="${esc(item.image_url)}" target="_blank" rel="noopener">Open photo at full size ↗</a></figcaption></figure><details class="rating-scene-request"><summary>What this photo should show</summary><p>${esc(item.scene_request)}</p></details></aside><form id="rating-form" class="rating-form" novalidate><nav class="rating-sections" aria-label="Rating sections">${sectionNames.map((name,i)=>`<button type="button" data-rating-action="section" data-section="${i}" ${i===section?'aria-current="step"':''}><span>${i+1}</span>${name}</button>`).join('')}</nav><p class="rating-section-hint">${section===0?'Compare the face, then count visible people.':section===1?'Check each required object against its reference and requested state.':'Check the requested setting and main action.'}</p><section data-rating-section="0" ${section!==0?'hidden':''}><div class="rating-target"><img src="${esc(item.protagonist.image_url)}" alt="${esc(item.protagonist.name)} canonical portrait"><div><strong>${esc(item.protagonist.name)} · reference portrait</strong><details><summary>Character description</summary><p>${esc(item.protagonist.description)}</p></details></div></div>${item.protagonist_required === false ? '<p class="rating-note">The protagonist is not requested in this scene. Identity is not applicable.</p>' : choices('identity','Does the person match the portrait?',['same','different','unclear','not_visible'],answer.identity,'Judge facial features, not clothing. Use Unclear for small or obscured faces.')}<fieldset><legend>How many people can you see?</legend><p class="small-note">The request calls for ${item.expected_people_count}. Count what is visible, regardless of identity.</p><div class="rating-field"><label for="rating-people-count">Visible people</label><input id="rating-people-count" type="number" min="0" max="20" step="1" value="${esc(answer.people_count)}" ${answer.people_count===null?'disabled':'required'}></div><label><input id="rating-count-unclear" type="checkbox" ${answer.people_count===null?'checked':''}> I cannot confidently count them</label></fieldset></section><section data-rating-section="1" ${section!==1?'hidden':''}>${item.objects.length?item.objects.map((o,i)=>objectView(o,i,answer.objects?.[o.id])).join(''):'<p class="rating-note">No recurring objects are required for this photo. Continue to Scene.</p>'}</section><section data-rating-section="2" ${section!==2?'hidden':''}><div class="rating-request"><p class="eyebrow">Requested scene</p><p>${esc(item.scene_request)}</p></div>${choices('setting','Does the setting match?',['yes','partial','no','unclear'],answer.setting)}${choices('action','Does the main action match?',['yes','partial','no','unclear'],answer.action)}<div class="rating-field"><label for="rating-notes">Optional observations</label><textarea id="rating-notes" maxlength="2000" rows="3" placeholder="What looks wrong, or remains uncertain?">${esc(answer.notes)}</textarea></div></section>${guide()}</form></div><div class="rating-actions"><button class="button secondary" data-rating-action="previous" ${index===0&&section===0?'disabled':''}>← ${section===0?'Previous photo':sectionNames[section-1]}</button><span id="rating-save-status" class="rating-save-status" role="status">${dirty.size?'Unsaved changes':'Saved locally'}</span><button class="button primary" data-rating-action="next">${section<2?`Next: ${sectionNames[section+1]}`:index===bundle.items.length-1||bundle.items.every(i=>complete(bundle.answers[i.id],i))?'Check my answers':'Next photo'} →</button></div>`;
  }

  function validateSection(number) {
    const panel=content.querySelector(`[data-rating-section="${number}"]`);
    const invalid=[...panel.querySelectorAll('input,select,textarea')].find(input=>!input.disabled&&!input.checkValidity());
    if (!invalid) return true;
    section=number;
    render();
    content.querySelector(`[data-rating-section="${number}"] :invalid`)?.reportValidity();
    return false;
  }

  function reviewView() {
    const submitted=bundle.session.status==='submitted';
    return `<p class="eyebrow">${esc(bundle.session.reviewer)} · ${esc(bundle.round.label)}</p><h2>${submitted?'Your independent review is saved.':'Check your answers before submitting.'}</h2><p>${submitted?'Your submitted answers are locked to preserve the original observations. Other reviewers should work independently before opening team results.':'Open any photograph to change its answers. Submission locks this review; drafts remain editable until then.'}</p><p class="rating-note">Prior exposure recorded: ${bundle.session.prior_exposure?'Yes':'No'}. One submitted review is one person’s evidence, not a team majority.</p>${bundle.items.map((item,i)=>{const a=bundle.answers[item.id]||{};return `<article class="rating-round"><h3>Photograph ${i+1}</h3><p>Identity: <strong>${esc(label(a.identity))}</strong> · Visible people: ${Object.hasOwn(a,'people_count')?esc(label(a.people_count)):'Not answered'} · Setting: ${esc(label(a.setting))} · Action: ${esc(label(a.action))}</p><ul>${item.objects.map(o=>{const v=a.objects?.[o.id]||{};return `<li>${esc(o.name)}: ${esc(label(v.presence))}; count ${Object.hasOwn(v,'count')?esc(label(v.count)):'not answered'}; appearance ${esc(label(v.appearance))}; state ${esc(label(v.state))}.</li>`;}).join('')}</ul>${a.notes?`<p><strong>Notes:</strong> ${esc(a.notes)}</p>`:''}${!submitted?`<button class="text-button" data-rating-action="item" data-index="${i}">Edit photograph ${i+1}</button>`:''}</article>`;}).join('')}<div class="rating-actions">${!submitted?`<button class="button primary" data-rating-action="submit" ${!bundle.items.every(i=>complete(bundle.answers[i.id],i))?'disabled':''}>Submit my ${bundle.items.length} ratings →</button>`:`<button class="button secondary" data-rating-action="export-session">Download my ratings</button>`}<button class="text-button" data-rating-action="home">Back to rating overview</button></div>`;
  }

  function resultsView() {
    const metrics=results.metrics||{};
    const metricNames={identity_match:'Protagonist identity matches',identity_unclear:'Identity unclear',identity_unresolved:'No identity majority',identity_not_visible:'No protagonist candidate',object_visibility:'Required objects visible',object_duplication:'Required objects duplicated',object_state_wrong:'Required object states wrong',scene_alignment:'Setting and action both satisfied'};
    const metricCaptions={identity_match:'of scenes requiring the protagonist',identity_unclear:'of scenes requiring the protagonist',identity_unresolved:'of scenes requiring the protagonist',identity_not_visible:'of scenes requiring the protagonist',object_visibility:'of required objects across photos',object_duplication:'of required objects across photos · lower is better',object_state_wrong:'of required objects across photos · lower is better',scene_alignment:'of all photos in this round'};
    const yesNo=v=>v===true?'Yes':v===false?'No':label(v);
    return `<p class="eyebrow">Development evidence · ${esc(round.label)}</p><h2>Human evaluation results</h2><p>${results.reviewer_count} submitted reviewers · ${results.prior_exposure_count} reported prior exposure.</p><p>Each result is a count based on independent human judgments. A majority means more than half of all submitted reviewers agree.</p><p class="rating-note"><strong>Human ratings and model scores answer different questions.</strong> These counts are human judgments, not ArcFace confidence scores. Saved face-similarity scores are shown separately in Pipeline → Scores &amp; results.</p><details class="rating-guide"><summary>How these results are counted</summary><ul><li><strong>Identity:</strong> only photos that request the protagonist are counted. Unclear faces, missing candidates and unresolved votes remain in that denominator.</li><li><strong>Scene alignment:</strong> all photos are counted. A majority must mark both the setting and main action Yes.</li><li><strong>Objects:</strong> each required object in each photo counts once. Visibility, duplication and wrong state are separate judgments; they are not added into one quality score.</li><li><strong>Lower error counts are better, but zero is not proof of perfection.</strong> Absent or unclear objects can leave appearance and state unassessed. Review the detailed judgments before interpreting a low duplication or wrong-state count.</li></ul><p>These are observations for this frozen round. A claim of improvement needs comparable results before and after a pipeline change.</p></details><div class="rating-metrics">${Object.entries(metrics).map(([key,m])=>`<article><p>${esc(metricNames[key]||key.replaceAll('_',' '))}</p><strong>${esc(m.numerator)} / ${esc(m.denominator)}</strong><p class="small-note">${esc(metricCaptions[key]||'')}</p>${m.note?`<p class="small-note">${esc(m.note)}</p>`:''}</article>`).join('')}</div><div class="rating-table"><table><caption>Per-photo team judgments</caption><thead><tr><th>Photo</th><th>Protagonist</th><th>Visible people</th><th>Setting</th><th>Main action</th></tr></thead><tbody>${results.items.map((item,i)=>`<tr><td><img src="${esc(item.image_url)}" alt="Rated photograph ${i+1}">Photo ${i+1}</td><td>${esc(label(item.identity))}<br><span class="small-note">${Object.entries(item.identity_votes).map(([v,n])=>`${esc(label(v))}: ${n}`).join(' · ')}</span></td><td>${esc(label(item.people_count))}<br>Requested: ${item.expected_people_count}</td><td>${esc(label(item.setting))}</td><td>${esc(label(item.action))}</td></tr>`).join('')}</tbody></table></div><details class="rating-guide"><summary>Required-object results and unresolved judgments</summary>${results.items.map((item,i)=>`<h3>Photograph ${i+1}</h3><ul>${item.objects.map(o=>`<li><strong>${esc(o.name)}</strong> — ${esc(label(o.presence))}; duplicates: ${esc(yesNo(o.duplicate))}; appearance: ${esc(label(o.appearance))}; state: ${esc(label(o.state))}.</li>`).join('')}</ul>`).join('')}</details><p class="rating-note">${esc(results.limitations||'Review disagreements before choosing a change. Prior exposure, small sample size and familiar characters limit what this development round establishes.')}</p><div class="rating-actions"><button class="button secondary" data-rating-action="export-round">Download round data</button><button class="text-button" data-rating-action="home">Back to rating overview</button></div><section class="rating-round"><h3>Use this round to finish iteration 1.</h3><ol><li>Choose one repeated defect that the independent ratings support.</li><li>Write one hypothesis and change one relevant pipeline component.</li><li>Generate paired outputs with the same scene requirements and record all calls.</li><li>Freeze a new rating round and use the same rubric. Compare identity, object and scene outcomes, including regressions.</li><li>Record the result and whether you kept or rejected the change. A failed hypothesis is still useful evidence.</li></ol></section>${roadmap()}`;
  }

  function render() {
    const titles={home:'Rate photos',join:'Start your review',rate:`Photo ${index+1} of ${bundle?.items.length||0}`,review:bundle?.session.status==='submitted'?'Review submitted':'Check & submit',results:'Team results'};
    content.innerHTML=`<div class="rating-shell"><header class="rating-top"><div class="rating-heading"><h2 id="rating-title" tabindex="-1">${titles[view]}</h2></div>${window.workspaceNav?.render('ratings')||''}<button class="text-button rating-exit" data-workspace-view="photos">${view==='rate'?'Save & exit':'← Photos'}</button></header><div class="rating-body">${error?`<p class="rating-error" role="alert">${esc(error)}</p>`:''}${({home:homeView,join:joinView,rate:rateView,review:reviewView,results:resultsView}[view])()}</div></div>`;
  }

  function capture() {
    const form=content.querySelector('#rating-form');
    if(!form)return;
    const answer={objects:{},notes:form.querySelector('#rating-notes').value};
    if(bundle.items[index].protagonist_required === false) answer.identity='not_applicable';
    for(const field of ['identity','setting','action']){const checked=form.querySelector(`input[name="${field}"]:checked`);if(checked)answer[field]=checked.value;}
    const count=form.querySelector('#rating-people-count');
    if(form.querySelector('#rating-count-unclear').checked)answer.people_count=null;
    else if(count.value!==''&&count.validity.valid)answer.people_count=Number(count.value);
    for(const group of form.querySelectorAll('[data-object]')){
      const presence=group.querySelector('[data-field="presence"]').value;
      if(!presence)continue;
      const value={presence};
      if(presence!=='present')Object.assign(value,{count:presence==='absent'?0:null,appearance:'not_applicable',state:'not_applicable'});
      else for(const field of ['count','appearance','state']){const input=group.querySelector(`[data-field="${field}"]`);if(input.value!==''&&input.validity.valid)value[field]=field==='count'?Number(input.value):input.value;}
      answer.objects[group.dataset.object]=value;
    }
    const id=bundle.items[index].id;
    bundle.answers[id]=answer;dirty.set(id,{answer,revision:++revision});
    saveStatus('Saving draft…');clearTimeout(timer);timer=setTimeout(()=>saveDirty().catch(()=>{}),550);
  }
  function saveStatus(message, failed=false) {
    const status=document.getElementById('rating-save-status');
    if(status){status.textContent=message;status.classList.toggle('rating-error',failed);}
  }
  async function saveDirty() {
    clearTimeout(timer);
    if(saving)return saving;
    if(!dirty.size)return;
    saving=(async()=>{
      try{
        while(dirty.size){const [itemId,pending]=dirty.entries().next().value;const response=await api(`/api/ratings/sessions/${bundle.session.id}/answer`,{item_id:itemId,answer:pending.answer});bundle.session=response.session;if(dirty.get(itemId)?.revision===pending.revision)dirty.delete(itemId);}
        saveStatus('Saved locally');
      }catch(failure){saveStatus(failure.message,true);throw failure;}
      finally{saving=null;}
    })();
    return saving;
  }
  async function navigate(action, button) {
    if(busy)return;
    busy=true;error='';
    try{
      await saveDirty();
      if(action==='home'){await refresh();view='home';}
      if(action==='create'){round=(await api('/api/ratings/rounds',{album_id:button.dataset.album})).round;view='join';}
      if(action==='join'){round=(await api(`/api/ratings/rounds/${button.dataset.round}`)).round;view='join';}
      if(action==='resume'){bundle=await api(`/api/ratings/sessions/${button.dataset.session}`);remember(bundle.session.id);index=Math.max(0,bundle.items.findIndex(i=>!complete(bundle.answers[i.id],i)));section=firstIncompleteSection(bundle.answers[bundle.items[index].id],bundle.items[index]);view=bundle.session.status==='submitted'||bundle.items.every(i=>complete(bundle.answers[i.id],i))?'review':'rate';}
      if(action==='item'){index=Number(button.dataset.index);section=0;view='rate';}
      if(action==='previous'){if(section>0)section--;else{index--;section=2;}view='rate';}
      if(action==='section'){const target=Number(button.dataset.section);for(let i=section;i<target;i++)if(!validateSection(i))return;section=target;}
      if(action==='next'){
        if(!validateSection(section))return;
        if(section<2)section++;
        else {
          for(let i=0;i<3;i++)if(!validateSection(i))return;
          if(bundle.items.every(i=>complete(bundle.answers[i.id],i)))view='review';
          else if(index<bundle.items.length-1){index++;section=0;}else view='review';
        }
      }
      if(action==='submit'){bundle=await api(`/api/ratings/sessions/${bundle.session.id}/submit`,{});view='review';await refresh();}
      if(action==='results'){round=(await api(`/api/ratings/rounds/${button.dataset.round}`)).round;results=await api(`/api/ratings/rounds/${round.id}/results`);view='results';}
      if(action.startsWith('export-')){
        const path=action==='export-session'?`/api/ratings/sessions/${bundle.session.id}/export`:`/api/ratings/rounds/${round.id}/export`;
        const value=await api(path);const url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:'application/json'}));const link=document.createElement('a');link.href=url;link.download=`camera-roll-${action==='export-session'?'my-review':'round'}-${action==='export-session'?bundle.session.id:round.id}.json`;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);return;
      }
      render();content.querySelector('#rating-title').focus({preventScroll:true});
    }catch(failure){error=failure.message;render();}
    finally{busy=false;}
  }

  document.addEventListener('click',event=>{
    const button=event.target.closest('[data-rating-action]');if(!button||button.disabled)return;
    const action=button.dataset.ratingAction;
    if(action==='open')open();else if(action==='close')close();else navigate(action,button);
  });
  content.addEventListener('input',event=>{if(event.target.closest('#rating-form'))capture();});
  content.addEventListener('change',event=>{
    if(!event.target.closest('#rating-form'))return;
    if(event.target.id==='rating-count-unclear'){
      const count=content.querySelector('#rating-people-count');count.disabled=event.target.checked;count.required=!event.target.checked;if(event.target.checked)count.value='';
    }
    if(event.target.dataset.field==='presence'){
      const group=event.target.closest('[data-object]'),present=event.target.value==='present';
      for(const input of group.querySelectorAll('[data-field]:not([data-field="presence"])')){input.disabled=!present;input.required=present;if(!present)input.value='';}
      group.querySelector('.rating-object-help').textContent=present?'Compare appearance with the reference; compare state and ownership with the request.':event.target.value==='absent'?'Absent means zero instances; appearance and state are not applicable.':'Count, appearance and state are unassessed until presence is clear.';
    }
    capture();
  });
  content.addEventListener('submit',async event=>{
    event.preventDefault();if(event.target.id!=='rating-join-form'||busy)return;
    busy=true;error='';
    try{
      const form=new FormData(event.target);
      bundle=await api(`/api/ratings/rounds/${round.id}/sessions`,{reviewer:String(form.get('reviewer')).trim(),prior_exposure:form.get('exposure')==='yes'});
      remember(bundle.session.id);index=Math.max(0,bundle.items.findIndex(i=>!complete(bundle.answers[i.id],i)));section=firstIncompleteSection(bundle.answers[bundle.items[index].id],bundle.items[index]);view=bundle.session.status==='submitted'||bundle.items.every(i=>complete(bundle.answers[i.id],i))?'review':'rate';render();content.querySelector('#rating-title').focus({preventScroll:true});
    }catch(failure){error=failure.message;render();}
    finally{busy=false;}
  });
  dialog.addEventListener('cancel',event=>{event.preventDefault();close();});
  dialog.addEventListener('keydown',event=>{
    if(event.key!=='Tab')return;
    const controls=[...dialog.querySelectorAll('button:not(:disabled),input:not(:disabled),select:not(:disabled),textarea,summary,a[href]')].filter(e=>e.getClientRects().length);
    if(event.shiftKey&&document.activeElement===controls[0]){event.preventDefault();controls.at(-1)?.focus();}
    else if(!event.shiftKey&&document.activeElement===controls.at(-1)){event.preventDefault();controls[0]?.focus();}
  });
  window.addEventListener('beforeunload',event=>{if(dirty.size){event.preventDefault();event.returnValue='';}});
  window.addEventListener('hashchange',()=>{if(location.hash==='#ratings'&&!dialog.open)open();else if(location.hash!=='#ratings'&&dialog.open)close();});
  window.ratingReview={open,close};
  if(location.hash==='#ratings')open();
})();
