'use strict';

(() => {
  const dialog = document.getElementById('creation-dialog');
  const content = document.getElementById('creation-content');
  const picker = document.getElementById('album-picker');
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[c]));
  const months = ['January','February','March','April','May','June','July','August','September','October','November','December'];
  const statusNames = {planning:'Writing story',ready:'Story ready',generating:'Making photos',completed:'Complete',failed:'Needs attention',interrupted:'Paused'};
  let config = null, runs = [], currentRun = null, currentData = null;
  let pollTimer = null, busy = false, error = '', opener = null, selectionVersion = 0;
  let draft = {brief:'',count:3};

  async function api(path, body) {
    const options = body === undefined ? {} : {method:'POST',headers:{'Content-Type':'application/json','X-Demo-Token':config?.csrf_token || ''},body:JSON.stringify(body)};
    let response;
    try { response = await fetch(path, options); }
    catch { throw new Error('The local server is not reachable. Open_Demo.command starts it again.'); }
    const value = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(value.error || 'The local server could not complete this request. Reload the page and try again.');
    return value;
  }

  async function refreshLibrary() {
    const result = await api('/api/runs');
    runs = result.runs;
    const selected = window.albumApp.currentId();
    picker.innerHTML = '<option value="maya">Maya Chen · Original example</option>' + runs.map(run => `<option value="${run.id}">${esc(run.name)} · ${statusNames[run.status] || run.status}</option>`).join('');
    picker.value = selected;
    const active = runs.find(run => ['planning','generating'].includes(run.status));
    document.getElementById('library-status').innerHTML = active ? `<button class="text-button" data-create-action="run" data-run="${active.id}">${esc(active.name)} is in progress ↗</button>` : '';
  }

  function closeCreator() {
    clearTimeout(pollTimer);
    picker.value = window.albumApp.currentId();
    if (dialog.open) dialog.close();
    if (!document.getElementById('detail-dialog').open) document.body.classList.remove('dialog-open');
    if (opener?.isConnected) opener.focus({preventScroll:true});
  }

  async function openCreator(identifier) {
    if (!dialog.open) opener = document.activeElement;
    clearTimeout(pollTimer);
    error = ''; currentData = null; currentRun = null;
    try {
      config = await api('/api/config');
      const active = runs.find(run => ['planning','generating'].includes(run.status));
      identifier ||= active?.id;
      if (identifier) {
        currentRun = (await api(`/api/runs/${identifier}`)).run;
        if (currentRun.data_available) currentData = await api(`/api/runs/${identifier}/data`);
      }
    } catch (failure) { error = failure.message; }
    render();
    document.body.classList.add('dialog-open');
    if (!dialog.open) dialog.showModal();
    (content.querySelector('#new-brief') || content.querySelector('#creation-title')).focus({preventScroll:true});
    schedulePoll();
  }

  function schedulePoll() {
    clearTimeout(pollTimer);
    if (dialog.open && currentRun && ['planning','generating'].includes(currentRun.status)) pollTimer = setTimeout(poll, 1800);
  }

  async function poll() {
    const id = currentRun?.id;
    if (!id || !dialog.open) return;
    try {
      const run = (await api(`/api/runs/${id}`)).run;
      if (currentRun?.id !== id) return;
      const changed = JSON.stringify(run) !== JSON.stringify(currentRun);
      currentRun = run;
      if (run.data_available && (!currentData || !['planning','generating'].includes(run.status))) currentData = await api(`/api/runs/${id}/data`);
      if (changed) render(true);
      if (!['planning','generating'].includes(run.status)) await refreshLibrary();
      error = '';
    } catch (failure) { error = failure.message; render(true); }
    schedulePoll();
  }

  function render(preserveFocus = false) {
    const previousAction = preserveFocus ? document.activeElement?.dataset?.createAction : null;
    const scroll = dialog.scrollTop;
    const step = !currentRun ? 0 : currentRun.phase === 'planning' || currentRun.status === 'ready' ? 1 : 2;
    content.innerHTML = `<div class="dialog-top"><p class="eyebrow">New story</p><button class="back-to-photos" data-workspace-view="photos">← Back to photos</button></div><div class="creator-content"><div class="creation-steps" aria-label="Creation stages">${['01 · Describe a person','02 · Review the story','03 · Make the photographs'].map((name,i)=>`<span class="creation-step ${i === step ? 'is-current' : i < step ? 'is-done' : ''}" ${i === step ? 'aria-current="step"' : ''}>${name}</span>`).join('')}</div>${error ? `<div class="error-note" role="alert">${esc(error)}</div>` : ''}${currentRun ? runView() : formView()}</div>`;
    dialog.scrollTop = preserveFocus ? scroll : 0;
    if (previousAction) content.querySelector(`[data-create-action="${previousAction}"]`)?.focus({preventScroll:true});
  }

  function formView() {
    const ready = config?.credentials.text_ready;
    return `<h2 id="creation-title" tabindex="-1">Create a new story.</h2><p class="muted">A person, a place, a few things they love. We’ll turn your brief into a fictional year and a camera roll.</p><form id="creation-form" class="creation-form"><label for="new-brief">Who is this story about?</label><textarea id="new-brief" name="brief" maxlength="2000" required placeholder="Nora is a ceramic artist in Portland. She loves early mornings in her studio and keeps a speckled yellow mug by the window…">${esc(draft.brief)}</textarea><p class="field-help">Use a fictional person. Include a location, everyday interests, and a possession or two that should recur.</p><div class="creation-options"><div><label for="new-photo-count">Photographs in the album</label><select id="new-photo-count">${[1,2,3,4,5].map(n=>`<option value="${n}" ${n===Number(draft.count)?'selected':''}>${n} photograph${n===1?'':'s'}${n===1?' · smallest run':''}</option>`).join('')}</select></div><div><span class="form-label">One year of their life</span><p class="field-help">A focused first chapter, with room to grow.</p></div></div>${!ready ? '<div class="service-note">The text service is not configured. Add the provider key to the private local credentials file, then reopen this panel.</div>' : ''}<div class="form-actions"><button class="button primary" type="submit" ${!ready||busy?'disabled':''}>${busy?'Starting…':'Create the story'} <span aria-hidden="true">→</span></button><p class="field-help">This first step uses the text model. You’ll review the image-request count before any photographs are generated.</p></div></form>`;
  }

  function runView() {
    const run = currentRun, working = ['planning','generating'].includes(run.status);
    const complete = run.status === 'completed';
    const title = complete ? 'A new camera roll, ready to keep.' : run.status === 'ready' ? `${run.name}’s story is ready.` : working ? run.phase === 'planning' ? 'A year is taking shape.' : 'Bringing the story into focus.' : 'Your progress is saved.';
    const heading = `<h2 id="creation-title" tabindex="-1">${esc(title)}</h2>`;
    if (working) {
      const fraction = run.plan ? Math.min(100,100*run.usage.image_completed/run.plan.image_calls) : 0;
      return `${heading}<p class="muted">${run.phase==='planning'?'Writing the story, then selecting its moments and recurring cast.':'Creating reusable references first, then each photo and its saved passes.'}</p><section class="generation-status" aria-live="polite"><div class="activity-line"><span class="working-dot" aria-hidden="true"></span><span>${esc(run.progress)}</span></div>${run.plan ? `<div class="progress-track" role="progressbar" aria-label="Images saved" aria-valuemin="0" aria-valuemax="${run.plan.image_calls}" aria-valuenow="${run.usage.image_completed}"><span style="width:${fraction}%"></span></div><p class="small-note">${run.usage.image_completed} of ${run.plan.image_calls} reference and scene images saved.</p>` : ''}<p class="request-count">Text requests: ${run.usage.llm_attempts} · Image requests: ${run.usage.image_attempts}${run.max_image_calls ? ` / ${run.max_image_calls} allowed` : ''}</p></section><p class="creation-note">You can close this panel while the server continues. Keep the local server running. Each completed step is saved for inspection and recovery.</p><div class="plan-actions">${run.data_available?'<button class="button secondary" data-create-action="inspect">Open saved pipeline →</button>':''}<button class="text-button" data-create-action="stop" ${run.stop_requested||busy?'disabled':''}>${run.stop_requested?'Stopping after the current request…':'Stop after the current request'}</button></div>`;
    }
    const failure = run.error ? `<div class="error-note" role="alert">${esc(run.error)}</div>` : '';
    if (!run.data_available || !currentData) return `${heading}${failure}${run.error ? '' : `<p class="muted">${esc(run.progress)}</p>`}<p class="request-count">Text attempts: ${run.usage.llm_attempts}. No image generation has started.</p><button class="button primary" data-create-action="resume-plan" ${busy?'disabled':''}>Resume the story →</button>`;
    const remaining = Math.max(0,run.plan.image_calls-run.usage.image_completed);
    const ceiling = run.usage.image_attempts+remaining;
    const overLimit = ceiling>(config?.limits.max_image_calls || 40);
    const body = `<div class="plan-grid"><section class="plan-story"><p class="eyebrow">Saved story · Year 1</p><p>${esc(currentData.story.background)}</p><details><summary>Read the complete year</summary>${currentData.story.years.map(y=>`<p class="narrative">${esc(y.narrative)}</p>`).join('')}</details><ol class="plan-scenes">${currentData.camera_roll.photos.map(p=>`<li><strong>${months[p.month-1]} · Year ${p.year}</strong><p>${esc(p.description)}</p></li>`).join('')}</ol></section><aside class="plan-metrics"><p class="eyebrow">${complete?'Saved album':'Before making the photos'}</p><h3>${run.plan.photo_count} photograph${run.plan.photo_count===1?'':'s'}</h3><dl><div><dt>Reference images</dt><dd>${run.plan.reference_count}</dd></div><div><dt>Scene-generation passes</dt><dd>${run.plan.scene_passes}</dd></div><div><dt>Total images to create</dt><dd>${run.plan.image_calls}</dd></div><div><dt>Images saved so far</dt><dd>${run.usage.image_completed}</dd></div></dl><p class="field-help">References and extra passes also use image requests. No automatic retries are made.</p>${complete ? '' : `<p><strong>${remaining} image requests remain.</strong> Starting this step uses your configured image provider and may incur charges. Text calls also compile the image prompts.</p>`}<p class="request-count">Used: ${run.usage.llm_attempts} text attempts · ${run.usage.image_attempts} image attempts</p></aside></div>`;
    return `${heading}${failure}${body}<div class="plan-actions">${complete ? '<button class="button primary" data-create-action="open-album">Open the camera roll ↗</button>' : `<button class="button primary" data-create-action="generate" ${busy||!config?.credentials.images_ready||overLimit?'disabled':''}>${busy?'Starting…':run.usage.image_attempts?'Resume photographs':'Create the photographs'} →</button>`}<button class="button secondary" data-create-action="inspect">Open pipeline →</button>${complete?'<button class="text-button" data-create-action="new">Create another story ＋</button>':''}</div>${!complete&&!config?.credentials.images_ready?'<p class="service-note">Image credentials are not configured locally. The story remains saved.</p>':''}${overLimit?'<p class="error-note">This run would exceed the local 40-request ceiling. The saved work remains available to inspect.</p>':''}`;
  }

  async function perform(action) {
    if (busy || !currentRun) return;
    busy = true; error = ''; render(true);
    try {
      const id = currentRun.id;
      let endpoint, body = {};
      if (action === 'generate') {
        endpoint = 'generate';
        body.max_image_calls = currentRun.usage.image_attempts + Math.max(0,currentRun.plan.image_calls-currentRun.usage.image_completed);
      } else endpoint = action === 'resume-plan' ? 'resume' : 'stop';
      currentRun = (await api(`/api/runs/${id}/${endpoint}`,body)).run;
      await refreshLibrary();
    } catch (failure) { error = failure.message; }
    finally { busy = false; render(true); schedulePoll(); }
  }

  async function loadAlbum(identifier, inspect = false, preserveRoute = false) {
    const version = ++selectionVersion;
    try {
      const album = identifier === 'maya' ? window.CAMERA_ROLL_DATA : await api(`/api/runs/${identifier}/data`);
      if (version !== selectionVersion) return;
      closeCreator();
      window.albumApp.load(album,identifier,preserveRoute);
      picker.value = identifier;
      if (inspect) window.albumApp.inspect();
      else if (!preserveRoute) document.getElementById('album').scrollIntoView({behavior:'smooth'});
    } catch (failure) { error = failure.message; if(dialog.open)render(true); else await openCreator(identifier); }
  }

  document.addEventListener('click',event=>{
    const button=event.target.closest('[data-create-action]');
    if(!button||button.disabled)return;
    const action=button.dataset.createAction;
    if(action==='close')closeCreator();
    if(action==='new')openCreator();
    if(action==='refresh-album')loadAlbum(window.albumApp.currentId(),false,true).then(refreshLibrary).catch(()=>{});
    if(action==='run')openCreator(button.dataset.run);
    if(['generate','resume-plan','stop'].includes(action))perform(action);
    if(action==='inspect'||action==='open-album')loadAlbum(currentRun.id,action==='inspect');
  });
  content.addEventListener('input',event=>{if(event.target.id==='new-brief')draft.brief=event.target.value;});
  content.addEventListener('change',event=>{if(event.target.id==='new-photo-count')draft.count=Number(event.target.value);});
  content.addEventListener('submit',async event=>{
    event.preventDefault(); if(busy)return;
    draft={brief:content.querySelector('#new-brief').value.trim(),count:Number(content.querySelector('#new-photo-count').value)};
    if(!draft.brief)return;
    busy=true;error='';render(true);
    try {
      currentRun=(await api('/api/runs',{brief:draft.brief,years:1,photos_per_year:draft.count})).run;
      currentData=null;await refreshLibrary();
    }catch(failure){error=failure.message;}
    finally{busy=false;render();schedulePoll();}
  });
  picker.addEventListener('change',()=>{
    const id=picker.value,run=runs.find(r=>r.id===id);
    if(id==='maya'||run?.status==='completed')loadAlbum(id);
    else openCreator(id);
  });
  dialog.addEventListener('cancel',event=>{event.preventDefault();closeCreator();});
  dialog.addEventListener('click',event=>{if(event.target===dialog)closeCreator();});
  dialog.addEventListener('keydown',event=>{
    if(event.key!=='Tab')return;
    const controls=[...dialog.querySelectorAll('button:not(:disabled),textarea,select,summary,a[href]')].filter(el=>el.getClientRects().length);
    if(event.shiftKey&&document.activeElement===controls[0]){event.preventDefault();controls.at(-1).focus();}
    else if(!event.shiftKey&&document.activeElement===controls.at(-1)){event.preventDefault();controls[0].focus();}
  });
  window.storyCreator = {close: closeCreator};
  window.addEventListener('album-loaded',event=>{picker.value=event.detail.id;});
  (async()=>{
    try{
      config=await api('/api/config');await refreshLibrary();
      const saved=new URL(location.href).searchParams.get('album');
      if(saved&&runs.some(r=>r.id===saved))await loadAlbum(saved,false,true);
    }catch{document.getElementById('library-status').textContent='Open the local server to create new stories.';}
  })();
})();
