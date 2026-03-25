document.addEventListener('DOMContentLoaded', function() {

  // open chatbot and unlock tabs
  document.getElementById('btnTalk').addEventListener('click', function() {
    chrome.runtime.sendMessage({ type: 'OPEN_CHAT' });
    window.close();
  });

  // user confirmed ok — unlock tabs
  document.getElementById('btnOk').addEventListener('click', function() {
    chrome.runtime.sendMessage({ type: 'USER_CONFIRMED_OK' });
    window.close();
  });

  // if user tries to switch tabs, snap back
  document.addEventListener('visibilitychange', function() {
    if (document.hidden) chrome.runtime.sendMessage({ type: 'FORCE_FOCUS' });
  });

  // also catch when window loses focus
  window.addEventListener('blur', function() {
    chrome.runtime.sendMessage({ type: 'FORCE_FOCUS' });
  });

});
