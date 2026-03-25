// track which tab is the alert page
var alertTabId = null;
var lockedTabId = null;
var alertActive = false;

// interval that keeps forcing user back
var lockInterval = null;

// every 300ms check if user switched away
function startLocking() {
  if (lockInterval) clearInterval(lockInterval);
  lockInterval = setInterval(function() {
    if (!alertActive || alertTabId === null) {
      clearInterval(lockInterval);
      lockInterval = null;
      return;
    }
    // if they switched tabs, snap them back
    chrome.tabs.query({ active: true, currentWindow: true }, function(tabs) {
      if (tabs.length > 0 && tabs[0].id !== alertTabId) {
        chrome.tabs.update(alertTabId, { active: true });
      }
    });
  }, 300);
}

// cleanup when user confirms they're ok
function stopLocking() {
  alertActive = false;
  alertTabId = null;
  if (lockInterval) { clearInterval(lockInterval); lockInterval = null; }
  // tell the original tab its unlocked
  if (lockedTabId !== null) {
    chrome.tabs.sendMessage(lockedTabId, { type: "USER_CONFIRMED_OK" }, function() {});
    lockedTabId = null;
  }
}

// read saved ngrok url from storage
function getChatUrl(callback) {
  chrome.storage.local.get("chatUrl", function(data) {
    callback(data.chatUrl || "http://localhost:5000");
  });
}

// listen for messages from content.js and alert.js
chrome.runtime.onMessage.addListener(function(msg, sender) {

  // distress detected — open alert page
  if (msg.type === "DISTRESS_DETECTED" && !alertActive) {
    alertActive = true;
    lockedTabId = sender.tab ? sender.tab.id : null;
    chrome.tabs.create({ url: chrome.runtime.getURL("alert.html"), active: true }, function(tab) {
      alertTabId = tab.id;
      startLocking();
    });
  }

  // user clicked im ok — unlock everything
  if (msg.type === "USER_CONFIRMED_OK") {
    stopLocking();
  }

  // user clicked talk to happyly — open chatbot
  if (msg.type === "OPEN_CHAT") {
    getChatUrl(function(url) {
      chrome.tabs.create({ url: url, active: true });
    });
    stopLocking();
  }

  // alert page lost focus — force it back
  if (msg.type === "FORCE_FOCUS" && alertTabId !== null) {
    chrome.tabs.update(alertTabId, { active: true });
  }
});

// catch tab switches and redirect back to alert
chrome.tabs.onActivated.addListener(function(activeInfo) {
  if (!alertActive || alertTabId === null) return;
  if (activeInfo.tabId !== alertTabId) {
    chrome.tabs.update(alertTabId, { active: true });
  }
});

// if alert tab gets closed just unlock
chrome.tabs.onRemoved.addListener(function(tabId) {
  if (tabId === alertTabId) { stopLocking(); }
});
