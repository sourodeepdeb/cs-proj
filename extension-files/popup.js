document.addEventListener('DOMContentLoaded', function() {

  // load saved url and show it in the input
  chrome.storage.local.get('chatUrl', function(data) {
    if (data.chatUrl && data.chatUrl !== 'http://localhost:5000') {
      document.getElementById('urlInput').value = data.chatUrl;
    }
  });

  // open chatbot tab when button clicked
  document.getElementById('btnChat').addEventListener('click', function() {
    chrome.runtime.sendMessage({ type: 'OPEN_CHAT' });
    window.close();
  });

  // save the url to chrome storage
  document.getElementById('btnSave').addEventListener('click', function() {
    var val = document.getElementById('urlInput').value.trim();
    // fall back to localhost if nothing entered
    var url = val || 'http://localhost:5000';
    chrome.storage.local.set({ chatUrl: url }, function() {
      // briefly show saved confirmation
      var msg = document.getElementById('savedMsg');
      msg.textContent = '✓ Saved!';
      msg.style.display = 'block';
      setTimeout(function() { msg.style.display = 'none'; }, 2500);
    });
  });

  // toggle the crisis resources dropdown
  document.getElementById('btnResources').addEventListener('click', function() {
    var panel = document.getElementById('resPanel');
    var btn = document.getElementById('btnResources');
    var isOpen = panel.classList.contains('open');
    panel.classList.toggle('open');
    // swap arrow direction
    btn.textContent = isOpen ? '▸ Crisis resources' : '▾ Crisis resources';
  });

});
