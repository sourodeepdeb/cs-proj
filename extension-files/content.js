// words that mean someone is struggling, which we got from analysis of embedding/csv files we scraped
const NEGATIVE_KEYWORDS = [
  "depressed","depression","suicidal","suicide","kill myself","end it all",
  "want to die","worthless","hopeless","can't go on","give up","no point",
  "hate myself","hate my life","no reason to live","tired of everything",
  "nothing matters","empty inside","i'm done","can't take it","overwhelmed",
  "lonely","nobody cares","miserable","broken","don't want to be here",
  "not worth it","can't do this anymore","feeling lost","can't cope",
  "no hope","falling apart","numb","feel numb","deeply sad","breaking down",
  "mental breakdown","crisis","desperate","despair","darkness","void",
  "self harm","cutting myself","hurt myself","end my life","disappear forever",
  "give up on life","life is pointless","what's the point","no future",
  "dark thoughts","hate everything","nobody loves me","i give up",
  "want it to end","feel like dying","wish i was dead","life sucks",
  "rock bottom","can't keep going","too much pain","don't want to exist",
  "wish i wasn't here","tired of living","no will to live","lost all hope",
  "nothing to live for","everyone would be better without me"
];

// words that mean someone is doing well, which we got from analysis of embedding/csv files we scraped
const POSITIVE_KEYWORDS = [
  "happy","happiness","joyful","joy","excited","grateful","thankful","blessed",
  "amazing","awesome","wonderful","fantastic","incredible","great","love","loved",
  "smile","laughing","fun","thrilled","elated","ecstatic","overjoyed","delighted",
  "content","satisfied","peaceful","calm","hopeful","confident","motivated",
  "inspired","proud","accomplished","success","cheerful","vibrant","thriving",
  "healthy","strong","positive","upbeat","enthusiastic","relaxed","comfortable",
  "doing well","doing great","feeling good","feeling great","feeling amazing",
  "life is good","loving life","having fun","never better","so happy","i'm happy",
  "i'm great","i'm good","feel good","feel great","pretty good","so excited"
];

// track if alert is already showing
let alertActive = false;
let lastAlertTime = 0;

// 30 second cooldown so it doesnt spam and completely fry computers
const COOLDOWN_MS = 30000;

// count how many keywords match
function countHits(text, keywords) {
  const lower = text.toLowerCase();
  return keywords.filter(function(kw) { return lower.includes(kw); }).length;
}

// decide whether to fire the alert
function analyzeText(text) {
  if (!text || !text.trim()) return;
  if (alertActive) return;
  var now = Date.now();
  if (now - lastAlertTime < COOLDOWN_MS) return;

  var negHits = countHits(text, NEGATIVE_KEYWORDS);
  var posHits = countHits(text, POSITIVE_KEYWORDS);

  // only trigger if negative words outweigh positive
  if (negHits > 0 && posHits <= negHits) {
    var lower = text.toLowerCase();
    var word = "distress";
    // find the first keyword that triggered it
    for (var i = 0; i < NEGATIVE_KEYWORDS.length; i++) {
      if (lower.includes(NEGATIVE_KEYWORDS[i])) { word = NEGATIVE_KEYWORDS[i]; break; }
    }
    alertActive = true;
    lastAlertTime = now;
    // tell background.js to open the alert tab
    chrome.runtime.sendMessage({ type: "DISTRESS_DETECTED", word: word, url: window.location.href });
  }
}

// get text from whatever element is active
function getText(el) {
  if (!el) return "";
  if (typeof el.value === "string") return el.value;
  return el.innerText || el.textContent || "";
}

// listen for any keypress on the page
document.addEventListener("keyup", function() {
  analyzeText(getText(document.activeElement));
}, true);

// listen for any input event
document.addEventListener("input", function(e) {
  analyzeText(getText(e.target));
}, true);

// also catch copy paste
document.addEventListener("paste", function() {
  setTimeout(function() { analyzeText(getText(document.activeElement)); }, 150);
}, true);

// attach listeners to all input fields
function attach() {
  var els = document.querySelectorAll('input,textarea,[contenteditable],[contenteditable="true"],[contenteditable="plaintext-only"]');
  els.forEach(function(el) {
    if (el._hl) return;
    el._hl = true;
    el.addEventListener("keyup", function() { analyzeText(getText(el)); }, true);
    el.addEventListener("input", function() { analyzeText(getText(el)); }, true);
    el.addEventListener("paste", function() { setTimeout(function(){ analyzeText(getText(el)); }, 150); }, true);
  });
}

// run on load and whenever new elements appear
attach();
new MutationObserver(attach).observe(document.documentElement, { childList: true, subtree: true });

// reset alert when user says they're ok
chrome.runtime.onMessage.addListener(function(msg) {
  if (msg.type === "USER_CONFIRMED_OK") { alertActive = false; }
});
