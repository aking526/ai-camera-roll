'use strict';

(() => {
  const views = {photos: 'Photos', pipeline: 'Pipeline', ratings: 'Rate photos'};
  let changing = false;

  function render(active) {
    return `<nav class="workspace-tabs" aria-label="Album sections">${Object.entries(views).map(([id, name]) => `<button data-workspace-view="${id}" ${id === active ? 'aria-current="page"' : ''}>${name}</button>`).join('')}</nav>`;
  }

  async function open(view) {
    if (changing) return;
    if (view === 'ratings' && document.getElementById('ratings-dialog').open) return;
    if (view === 'pipeline' && document.getElementById('detail-dialog').open && document.getElementById('detail-dialog').classList.contains('process-dialog')) return;
    changing = true;
    try {
      if (document.getElementById('ratings-dialog').open && await window.ratingReview.close() === false) return;
      if (document.getElementById('creation-dialog').open) window.storyCreator.close();
      window.albumApp.close();
      if (view === 'pipeline') window.albumApp.inspect();
      else if (view === 'ratings') await window.ratingReview.open();
      else {
        history.pushState(null, '', '#album');
        document.getElementById('album').scrollIntoView({behavior: 'instant'});
      }
    } finally { changing = false; }
  }

  window.workspaceNav = {render, open};
  document.getElementById('workspace-navigation').innerHTML = render('photos');
  document.addEventListener('click', event => {
    const button = event.target.closest('[data-workspace-view]');
    if (button) open(button.dataset.workspaceView);
  });
})();
