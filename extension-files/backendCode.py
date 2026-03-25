#install colab requirements
!pip install flask flask-cors openai numpy pandas pyngrok -q

#for purpose of our project, we had our data in a gdrive
from google.colab import drive
drive.mount('/content/drive')

import os, ast, re
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from pyngrok import ngrok

# our api keys and file path. unlisted here since openAI policy is you can't leak or else ban from chatGPT/any openAI products
OPENAI_API_KEY = "---"
NGROK_TOKEN    = "---"
EMBEDDINGS_CSV = "---"

# connect ngrok and openai
ngrok.set_auth_token(NGROK_TOKEN)
client = OpenAI(api_key=OPENAI_API_KEY)

# start the flask server
app = Flask(__name__)
CORS(app)

# store embeddings loaded from csv
stored_embeddings = []
stored_texts = []
csv_keywords = set()

# words we want to ignore
STOP_WORDS = {
    "the","a","an","and","or","but","in","on","at","to","for","of","with",
    "is","it","i","my","me","you","he","she","they","we","this","that",
    "was","are","be","been","have","has","had","do","did","will","would",
    "could","should","not","no","so","just","like","get","got","its","if",
    "as","up","out","about","what","when","how","all","from","by","there",
    "their","them","then","than","more","can","one","your","who","which",
    "his","her","our","were","been","being","into","through","during",
    "before","after","above","below","between","each","here","where","why",
    "because","while","although","though","since","until","any","some",
    "most","other","also","only","even","well","back","still","way","take",
    "go","come","make","know","think","see","look","want","give","use",
    "find","tell","ask","seem","feel","try","leave","call","keep","let",
    "put","turn","mean","become","show","hear","play","run","move","live",
    "believe","hold","bring","happen","write","provide","sit","stand",
    "lose","pay","meet","include","continue","set","learn","change","lead",
    "understand","watch","follow","stop","create","speak","read","spend",
    "grow","open","walk","win","offer","remember","love","consider","appear",
    "buy","wait","serve","die","send","expect","build","stay","fall","cut",
    "reach","kill","remain","suggest","raise","pass","sell","require","report"
}

# words that mean someone is struggling
NEGATIVE_KEYWORDS = [
    "depressed","depression","suicidal","suicide","kill myself","end it all",
    "want to die","worthless","hopeless","can't go on","give up","no point",
    "hate myself","hate my life","no reason to live","tired of everything",
    "nothing matters","empty inside","i'm done","can't take it","overwhelmed",
    "lonely","nobody cares","miserable","broken","don't want to be here",
    "not worth it","can't do this anymore","feeling lost","can't cope",
    "no hope","falling apart","numb","exhausted","anxious","panic","trapped",
    "stuck","pointless","sad","crying","hurting","suffering","alone",
    "isolated","neglected","abandoned","rejected","failure","useless",
    "self harm","cutting","hurt myself","end my life","not here anymore",
    "disappear","run away","give up on life","life is pointless",
    "what's the point","no future","dark thoughts","intrusive thoughts",
    "hate everything","nobody loves me","i give up","cant go on",
    "want it to end","feel like dying","wish i was dead","life sucks",
    "terrible","awful","horrible","dreadful","unbearable","devastating",
    "heartbroken","grief","grieving","mourning","falling apart","breaking down",
    "mental breakdown","nervous breakdown","crisis","desperate","despair",
    "anguish","agony","torment","nightmare","darkness","void","rock bottom"
]

# words that mean someone is happy
POSITIVE_KEYWORDS = [
    "happy","happiness","joyful","joy","excited","excitement","grateful",
    "gratitude","thankful","blessed","blissful","bliss","amazing","awesome",
    "wonderful","fantastic","incredible","brilliant","great","magnificent",
    "superb","excellent","love","loved","loving","adore","cherish","appreciate",
    "appreciated","smile","smiling","laughing","laugh","fun","funny","hilarious",
    "thrilled","elated","ecstatic","euphoric","overjoyed","delighted","content",
    "satisfied","fulfilled","peaceful","calm","serene","tranquil","optimistic",
    "hopeful","confident","motivated","inspired","energized","proud","accomplished",
    "achieved","success","successful","winning","won","cheerful","bubbly","radiant",
    "glowing","beaming","alive","vibrant","thriving","flourishing","blooming",
    "healthy","strong","powerful","capable","unstoppable","determined","positive",
    "upbeat","enthusiastic","passionate","driven","focused","relaxed","comfortable",
    "cozy","safe","secure","warm","friendship","friends","family","together",
    "connection","community","adventure","exploring","discovering","progress",
    "forward","better","best","perfect","celebrating","celebration","refreshed",
    "renewed","revived","rejuvenated","restored","healed","looking forward",
    "pumped","stoked","psyched","hyped","ready","prepared","beautiful","gorgeous",
    "stunning","lovely","charming","lucky","fortunate","giggling","chuckling",
    "grinning","sparkling","shining","sunshine","bright","light","dream","aspire",
    "doing well","doing great","feeling amazing","feeling wonderful","feeling good",
    "feeling great","life is good","loving life","enjoying life","having fun",
    "having a blast","on top of the world","over the moon","never better",
    "best day","great day","good day","wonderful day","i'm good","i'm great",
    "i'm happy","so happy","really happy","super happy","very happy","feel good",
    "feel great","feel amazing","feel wonderful","feel fantastic","feel awesome",
    "pretty good","pretty great","pretty happy","so excited","really excited",
    "very excited","cant wait","can't wait","looking forward","hyped up",
    "pumped up","fired up","stoked about","thrilled about","love this","love it",
    "loving it","loving this","this is great","this is amazing","so good",
    "so great","so awesome","killing it","crushing it","nailed it","smashed it",
    "absolutely love","really enjoying","genuinely happy","truly happy",
    "beyond happy","beyond excited","beyond grateful","so blessed","so thankful",
    "incredibly grateful","deeply grateful","very grateful","much better",
    "way better","so much better","feeling much better","feeling way better",
    "incredible day","amazing day","perfect day","great news","amazing news",
    "good news","positive vibes","good vibes","great vibes","high spirits",
    "in a great mood","in a good mood","spirits are high","on cloud nine"
]

# pull unique words from reddit comments
def extract_csv_keywords(df):
    words = set()
    for text in df["parent_comment"].dropna().astype(str):
        tokens = re.findall(r"[a-z']{4,}", text.lower())
        for t in tokens:
            # skip boring common words
            if t not in STOP_WORDS and len(t) >= 4:
                words.add(t)
    return words

# load saved embeddings from google drive
def load_embeddings():
    global stored_embeddings, stored_texts, csv_keywords
    if not os.path.exists(EMBEDDINGS_CSV):
        print("CSV not found — check your path")
        return
    df = pd.read_csv(EMBEDDINGS_CSV)
    for _, row in df.iterrows():
        try:
            # convert string back to float array
            vec = np.array(ast.literal_eval(row["parent_embedding"]), dtype=np.float32)
            stored_embeddings.append(vec)
            stored_texts.append(str(row.get("parent_comment", "")))
        except:
            continue
    csv_keywords = extract_csv_keywords(df)
    print(f"Loaded {len(stored_embeddings)} embeddings")
    print(f"Extracted {len(csv_keywords)} keywords from CSV")

# turn text into a vector using openai
def embed_text(text):
    text = str(text).replace("\n", " ")
    return np.array(client.embeddings.create(input=[text], model="text-embedding-3-small").data[0].embedding, dtype=np.float32)

# measure how similar two vectors are
def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

# check if message matches distressed reddit posts
def is_distressed(emb):
    if not stored_embeddings: return False, 0.0
    best = max(cosine_sim(emb, e) for e in stored_embeddings)
    return best >= 0.78, round(best, 4)

# count keyword hits, return mood change amounts
def score_message(text):
    lower = text.lower()
    neg_hits = sum(1 for kw in NEGATIVE_KEYWORDS if kw in lower)
    pos_hits = sum(1 for kw in POSITIVE_KEYWORDS if kw in lower)
    # check against words from our reddit data
    words = set(re.findall(r"[a-z']{4,}", lower))
    csv_hits = len(words & csv_keywords)
    is_negative = neg_hits > 0 or csv_hits >= 3
    is_positive = pos_hits > 0 and neg_hits == 0
    # cap how much mood can change
    boost = min(pos_hits * 0.05, 0.25)
    drop = min(neg_hits * 0.08 + (0.05 if csv_hits >= 3 else 0), 0.3)
    return is_negative, is_positive, round(boost, 3), round(drop, 3)

# tells gpt how to behave
SYSTEM_PROMPT = "You are HappyLy, an empathetic AI companion who listens carefully and responds with warmth and care. Keep responses to 2-3 sentences. Acknowledge feelings first if someone seems distressed. You are NOT a replacement for professional mental health support."

# main chat endpoint the frontend calls
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "")
    history = data.get("history", [])

    # embed the message and run ml check
    emb = embed_text(user_message)
    ml_flag, score = is_distressed(emb)

    # also run keyword check
    is_neg, is_pos, boost, drop = score_message(user_message)
    flagged = ml_flag or is_neg

    # build message history for gpt
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": user_message})

    # call gpt-4o and get response
    completion = client.chat.completions.create(model="gpt-4o", messages=messages, max_tokens=350, temperature=0.8)
    return jsonify({
        "reply": completion.choices[0].message.content,
        "flagged": flagged,
        "positive": is_pos,
        "mood_boost": boost,
        "mood_drop": drop,
        "similarity_score": score
    })

# the entire frontend lives in this string
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>HappyLy</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Syne:wght@700;800&display=swap" rel="stylesheet"/>
<style>
/* reset all default browser spacing */
*{box-sizing:border-box;margin:0;padding:0}
/* dark background, centered layout */
body{min-height:100vh;display:flex;align-items:center;justify-content:center;font-family:'DM Sans',sans-serif;overflow:hidden;transition:background 2s ease;}
/* floating orb animation */
@keyframes hlfloat{0%,100%{transform:translateY(0)}50%{transform:translateY(-16px)}}
/* message slide up animation */
@keyframes hlup{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}
/* typing dots bounce */
@keyframes hldot{0%,80%,100%{transform:scale(0.55);opacity:0.25}40%{transform:scale(1);opacity:1}}
/* wellness popup pop in */
@keyframes hlpopIn{from{opacity:0;transform:translate(-50%,-50%) scale(0.88)}to{opacity:1;transform:translate(-50%,-50%) scale(1)}}
/* overlay fade in */
@keyframes hlfadein{from{opacity:0}to{opacity:1}}
/* background glow orbs */
.orb1{position:fixed;width:520px;height:520px;border-radius:50%;top:-130px;right:-130px;animation:hlfloat 9s ease-in-out infinite;pointer-events:none;transition:background 2s ease;}
.orb2{position:fixed;width:380px;height:380px;border-radius:50%;bottom:-100px;left:-100px;animation:hlfloat 12s ease-in-out infinite reverse;pointer-events:none;transition:background 2s ease;}
.orb3{position:fixed;width:260px;height:260px;border-radius:50%;top:50%;left:50%;transform:translate(-50%,-50%);pointer-events:none;transition:background 2s ease;opacity:0.35;}
/* main chat window glass effect */
.window{width:min(680px,96vw);height:min(820px,96vh);display:flex;flex-direction:column;background:rgba(255,255,255,0.025);backdrop-filter:blur(48px) saturate(180%);border:1px solid rgba(255,255,255,0.07);border-radius:28px;box-shadow:0 0 0 1px rgba(255,255,255,0.03) inset,0 32px 80px rgba(0,0,0,0.6);overflow:hidden;}
/* top bar with logo and mood */
.header{padding:18px 24px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid rgba(255,255,255,0.05);background:rgba(255,255,255,0.02);flex-shrink:0;}
/* app logo square */
.logo-ring{width:38px;height:38px;border-radius:11px;display:flex;align-items:center;justify-content:center;font-size:20px;transition:background 2s ease,box-shadow 2s ease;}
.appname{font-family:'Syne',sans-serif;font-weight:800;color:#fff;font-size:17px;letter-spacing:-0.4px;}
.tagline{font-size:10px;color:rgba(255,255,255,0.28);letter-spacing:0.8px;text-transform:uppercase;margin-top:1px;}
.mood-label{font-size:10px;color:rgba(255,255,255,0.25);letter-spacing:0.7px;text-transform:uppercase;}
/* mood bar container */
.mood-track{width:80px;height:6px;border-radius:3px;background:rgba(255,255,255,0.07);overflow:hidden;}
/* colored fill inside mood bar */
.mood-fill{height:100%;border-radius:3px;transition:all 1.4s cubic-bezier(0.4,0,0.2,1);}
.mood-tag{font-size:10px;color:rgba(255,255,255,0.3);letter-spacing:0.5px;transition:color 2s ease;min-width:44px;text-align:right;}
/* scrollable message area */
.messages{flex:1;overflow-y:auto;padding:28px 24px;display:flex;flex-direction:column;gap:18px;}
.messages::-webkit-scrollbar{width:4px;}
.messages::-webkit-scrollbar-thumb{background:rgba(255,255,255,0.08);border-radius:2px;}
/* each message row */
.msg{display:flex;align-items:flex-end;gap:10px;animation:hlup 0.28s ease forwards;}
.msg.user{flex-direction:row-reverse;}
/* bot avatar circle */
.av{width:30px;height:30px;border-radius:9px;display:flex;align-items:center;justify-content:center;font-size:15px;flex-shrink:0;transition:background 2s ease;}
/* chat bubble base styles */
.bubble{max-width:74%;padding:12px 17px;font-size:14px;line-height:1.65;color:rgba(255,255,255,0.88);}
/* bot bubble rounded left side */
.bubble.bot{border-radius:20px 20px 20px 5px;background:rgba(255,255,255,0.055);border:1px solid rgba(255,255,255,0.055);}
/* user bubble rounded right side */
.bubble.user{border-radius:20px 20px 5px 20px;transition:background 2s ease,border-color 2s ease;}
.bubble.err{color:#ff6b81;}
/* input area at the bottom */
.inputbar{padding:14px 20px 22px;border-top:1px solid rgba(255,255,255,0.05);background:rgba(255,255,255,0.015);flex-shrink:0;}
/* rounded input row with send button */
.inputrow{display:flex;align-items:center;gap:10px;background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.09);border-radius:18px;padding:5px 5px 5px 18px;}
.inputrow input{flex:1;background:none;border:none;color:rgba(255,255,255,0.9);font-size:14px;outline:none;font-family:'DM Sans',sans-serif;padding:10px 0;}
.inputrow input::placeholder{color:rgba(255,255,255,0.22);}
/* send button changes color when active */
.sendbtn{width:42px;height:42px;border-radius:13px;border:none;cursor:pointer;font-size:20px;display:flex;align-items:center;justify-content:center;transition:all 0.2s;}
.sendbtn.on{color:#fff;}
.sendbtn.off{background:rgba(255,255,255,0.04);color:rgba(255,255,255,0.18);cursor:default;}
.footer-note{text-align:center;margin-top:10px;font-size:10px;color:rgba(255,255,255,0.15);letter-spacing:0.6px;text-transform:uppercase;}
/* typing indicator dots */
.dot{width:7px;height:7px;border-radius:50%;background:rgba(255,255,255,0.45);}
/* dark overlay behind wellness popup */
.overlay{position:fixed;inset:0;background:rgba(0,0,0,0.65);backdrop-filter:blur(10px);z-index:200;display:none;animation:hlfadein 0.3s ease;}
.overlay.show{display:block;}
/* wellness check popup card */
.popup{position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);width:min(420px,90vw);background:linear-gradient(145deg,#0d0018,#150030);border:1px solid rgba(168,85,247,0.2);border-radius:28px;padding:36px 32px;box-shadow:0 0 100px rgba(168,85,247,0.15),0 0 0 1px rgba(255,255,255,0.03) inset;animation:hlpopIn 0.4s cubic-bezier(0.34,1.56,0.64,1) forwards;}
.pop-icon{width:60px;height:60px;border-radius:18px;background:linear-gradient(135deg,rgba(168,85,247,0.2),rgba(109,40,217,0.2));border:1px solid rgba(168,85,247,0.3);display:flex;align-items:center;justify-content:center;font-size:26px;margin-bottom:22px;}
.pop-title{font-family:'Syne',sans-serif;font-weight:800;font-size:24px;color:#fff;margin-bottom:10px;letter-spacing:-0.4px;}
.pop-sub{font-size:14px;color:rgba(255,255,255,0.42);line-height:1.65;margin-bottom:26px;}
/* crisis resource cards */
.res{display:flex;align-items:center;gap:14px;padding:13px 16px;background:rgba(255,255,255,0.035);border:1px solid rgba(255,255,255,0.06);border-radius:14px;margin-bottom:10px;cursor:pointer;transition:all 0.2s;}
.res:hover{background:rgba(168,85,247,0.1);transform:translateX(3px);}
.res-icon{font-size:22px;}
.res-title{font-size:13px;font-weight:500;color:rgba(255,255,255,0.85);}
.res-sub{font-size:11px;color:rgba(255,255,255,0.3);margin-top:2px;}
.pop-btns{display:flex;gap:10px;margin-top:4px;}
.pbtn{flex:1;padding:13px;border-radius:14px;cursor:pointer;font-size:13px;font-family:'DM Sans',sans-serif;border:none;transition:all 0.2s;}
.pbtn:hover{opacity:0.85;transform:scale(1.03);}
.pbtn.ghost{background:rgba(255,255,255,0.05);color:rgba(255,255,255,0.6);}
.pbtn.primary{color:#fff;font-weight:600;}
.closebtn{position:absolute;top:16px;right:18px;background:none;border:none;cursor:pointer;color:rgba(255,255,255,0.25);font-size:24px;line-height:1;}
</style>
</head>
<body>
<!-- ambient background orbs -->
<div class="orb1" id="orb1"></div>
<div class="orb2" id="orb2"></div>
<div class="orb3" id="orb3"></div>

<!-- main chat window -->
<div class="window">
  <div class="header">
    <div style="display:flex;align-items:center;gap:13px;">
      <!-- logo changes emoji based on mood -->
      <div class="logo-ring" id="logoRing">😊</div>
      <div>
        <div class="appname">HappyLy</div>
        <div class="tagline">listening between the lines</div>
      </div>
    </div>
    <div style="display:flex;align-items:center;gap:12px;">
      <!-- mood bar fills based on conversation -->
      <div style="display:flex;align-items:center;gap:8px;">
        <span class="mood-label">mood</span>
        <div class="mood-track"><div class="mood-fill" id="moodFill"></div></div>
        <span class="mood-tag" id="moodTag">Good</span>
      </div>
      <div id="statusDot" style="width:8px;height:8px;border-radius:50%;transition:all 1.4s;"></div>
    </div>
  </div>

  <!-- messages render here dynamically -->
  <div class="messages" id="msgs">
    <div class="msg">
      <div class="av" style="background:linear-gradient(135deg,#a855f7,#7c3aed);">😊</div>
      <div class="bubble bot">Hey, I'm HappyLy 😊  I'm here to chat — about anything. What's on your mind?</div>
    </div>
  </div>

  <!-- text input and send button -->
  <div class="inputbar">
    <div class="inputrow">
      <input id="inp" placeholder="What's on your mind..." onkeydown="onKey(event)" oninput="syncBtn()"/>
      <button class="sendbtn off" id="sendBtn" onclick="send()">↑</button>
    </div>
    <div class="footer-note">Not a substitute for professional mental health support</div>
  </div>
</div>

<!-- wellness popup shown when distress detected -->
<div class="overlay" id="overlay">
  <div class="popup">
    <button class="closebtn" onclick="closeWell()">×</button>
    <div class="pop-icon">🫂</div>
    <div class="pop-title">Hey — are you okay?</div>
    <div class="pop-sub">I picked up on something in what you shared. You don't have to carry this alone — people are ready to help right now.</div>
    <!-- crisis hotline links -->
    <div class="res"><span class="res-icon">📞</span><div><div class="res-title">988 Suicide & Crisis Lifeline</div><div class="res-sub">Call or text 988 — free, 24/7</div></div></div>
    <div class="res"><span class="res-icon">💬</span><div><div class="res-title">Crisis Text Line</div><div class="res-sub">Text HOME to 741741</div></div></div>
    <div class="res"><span class="res-icon">🌐</span><div><div class="res-title">NAMI Helpline</div><div class="res-sub">1-800-950-6264 — Mon–Fri, 10am–10pm ET</div></div></div>
    <div class="pop-btns">
      <button class="pbtn ghost" onclick="closeWell()">I'm okay, thanks</button>
      <button class="pbtn primary" id="talkBtn" onclick="talkToMe()">Let's talk 💙</button>
    </div>
  </div>
</div>

<script>
// track mood score between 0 and 1
let mood = 0.65, dismissed = false, history = [], busy = false;

// each stage defines colors for that mood level
const STAGES = [
  { min:0,    max:0.2,  body:"linear-gradient(135deg,#030008 0%,#0a0018 60%,#060010 100%)", orb1:"radial-gradient(circle,rgba(88,28,135,0.2) 0%,transparent 70%)",   orb2:"radial-gradient(circle,rgba(68,8,115,0.16) 0%,transparent 70%)",   orb3:"radial-gradient(circle,rgba(50,0,80,0.14) 0%,transparent 70%)",    logo:"linear-gradient(135deg,#581c87,#3b0764)", fill:"linear-gradient(90deg,#3b0764,#581c87)",           btn:"linear-gradient(135deg,#581c87,#3b0764)", av:"linear-gradient(135deg,#581c87,#3b0764)", ub:"linear-gradient(135deg,rgba(88,28,135,0.3),rgba(59,7,100,0.3))",   ubb:"rgba(88,28,135,0.4)",  dot:"#581c87", glow:"rgba(88,28,135,0.55)",  emoji:"😔", label:"Very Low" },
  { min:0.2,  max:0.4,  body:"linear-gradient(135deg,#050012 0%,#0f0028 60%,#08001a 100%)", orb1:"radial-gradient(circle,rgba(109,40,217,0.18) 0%,transparent 70%)",  orb2:"radial-gradient(circle,rgba(91,33,182,0.14) 0%,transparent 70%)",  orb3:"radial-gradient(circle,rgba(76,29,149,0.11) 0%,transparent 70%)",  logo:"linear-gradient(135deg,#6d28d9,#4c1d95)", fill:"linear-gradient(90deg,#4c1d95,#6d28d9)",           btn:"linear-gradient(135deg,#6d28d9,#4c1d95)", av:"linear-gradient(135deg,#6d28d9,#4c1d95)", ub:"linear-gradient(135deg,rgba(109,40,217,0.26),rgba(76,29,149,0.26))",ubb:"rgba(109,40,217,0.36)", dot:"#6d28d9", glow:"rgba(109,40,217,0.5)", emoji:"😕", label:"Low"      },
  { min:0.4,  max:0.58, body:"linear-gradient(135deg,#060015 0%,#130030 60%,#0a001e 100%)", orb1:"radial-gradient(circle,rgba(147,51,234,0.15) 0%,transparent 70%)",  orb2:"radial-gradient(circle,rgba(124,58,237,0.12) 0%,transparent 70%)", orb3:"radial-gradient(circle,rgba(109,40,217,0.1) 0%,transparent 70%)",  logo:"linear-gradient(135deg,#9333ea,#7c3aed)", fill:"linear-gradient(90deg,#7c3aed,#9333ea)",           btn:"linear-gradient(135deg,#9333ea,#7c3aed)", av:"linear-gradient(135deg,#9333ea,#7c3aed)", ub:"linear-gradient(135deg,rgba(147,51,234,0.22),rgba(124,58,237,0.22))",ubb:"rgba(147,51,234,0.3)", dot:"#9333ea", glow:"rgba(147,51,234,0.5)", emoji:"😐", label:"Okay"     },
  { min:0.58, max:0.72, body:"linear-gradient(135deg,#060015 0%,#0d0025 40%,#0a1200 100%)", orb1:"radial-gradient(circle,rgba(168,85,247,0.12) 0%,transparent 70%)",  orb2:"radial-gradient(circle,rgba(132,204,22,0.09) 0%,transparent 70%)",  orb3:"radial-gradient(circle,rgba(101,163,13,0.07) 0%,transparent 70%)",  logo:"linear-gradient(135deg,#a855f7,#84cc16)",  fill:"linear-gradient(90deg,#7c3aed,#a855f7,#84cc16)",  btn:"linear-gradient(135deg,#a855f7,#84cc16)",  av:"linear-gradient(135deg,#a855f7,#84cc16)",  ub:"linear-gradient(135deg,rgba(168,85,247,0.18),rgba(132,204,22,0.14))",ubb:"rgba(168,85,247,0.28)", dot:"#a855f7", glow:"rgba(168,85,247,0.45)", emoji:"🙂", label:"Good"     },
  { min:0.72, max:0.85, body:"linear-gradient(135deg,#001a08 0%,#002d10 60%,#001408 100%)", orb1:"radial-gradient(circle,rgba(34,197,94,0.15) 0%,transparent 70%)",   orb2:"radial-gradient(circle,rgba(22,163,74,0.12) 0%,transparent 70%)",  orb3:"radial-gradient(circle,rgba(21,128,61,0.1) 0%,transparent 70%)",   logo:"linear-gradient(135deg,#22c55e,#16a34a)",  fill:"linear-gradient(90deg,#16a34a,#22c55e,#4ade80)",  btn:"linear-gradient(135deg,#22c55e,#16a34a)",  av:"linear-gradient(135deg,#22c55e,#16a34a)",  ub:"linear-gradient(135deg,rgba(34,197,94,0.22),rgba(22,163,74,0.22))", ubb:"rgba(34,197,94,0.3)",  dot:"#22c55e", glow:"rgba(34,197,94,0.5)",   emoji:"😊", label:"Happy"    },
  { min:0.85, max:1.01, body:"linear-gradient(135deg,#001206 0%,#003d15 50%,#001f08 100%)", orb1:"radial-gradient(circle,rgba(74,222,128,0.2) 0%,transparent 70%)",   orb2:"radial-gradient(circle,rgba(34,197,94,0.16) 0%,transparent 70%)",  orb3:"radial-gradient(circle,rgba(22,163,74,0.13) 0%,transparent 70%)",  logo:"linear-gradient(135deg,#4ade80,#22c55e)",  fill:"linear-gradient(90deg,#22c55e,#4ade80,#86efac)",  btn:"linear-gradient(135deg,#4ade80,#22c55e)",  av:"linear-gradient(135deg,#4ade80,#22c55e)",  ub:"linear-gradient(135deg,rgba(74,222,128,0.26),rgba(34,197,94,0.26))",ubb:"rgba(74,222,128,0.36)", dot:"#4ade80", glow:"rgba(74,222,128,0.6)",  emoji:"🤩", label:"Amazing"  }
];

// find which stage the current mood is in
function getStage(m){ return STAGES.find(s => m >= s.min && m < s.max) || STAGES[STAGES.length-1]; }

// update all colors on the page
function applyTheme(s) {
  document.body.style.background = s.body;
  document.getElementById('orb1').style.background = s.orb1;
  document.getElementById('orb2').style.background = s.orb2;
  document.getElementById('orb3').style.background = s.orb3;
  const logo = document.getElementById('logoRing');
  logo.style.background = s.logo;
  logo.style.boxShadow = '0 4px 20px ' + s.glow;
  // emoji changes with mood
  logo.textContent = s.emoji;
  document.getElementById('moodFill').style.background = s.fill;
  document.getElementById('moodTag').textContent = s.label;
  const sd = document.getElementById('statusDot');
  sd.style.background = s.dot;
  sd.style.boxShadow = '0 0 10px ' + s.glow;
  const sb = document.getElementById('sendBtn');
  if (sb.classList.contains('on')) { sb.style.background = s.btn; sb.style.boxShadow = '0 4px 14px ' + s.glow; }
  document.getElementById('talkBtn').style.background = s.btn;
  // update all existing bubbles too
  document.querySelectorAll('.av').forEach(a => { a.style.background = s.av; a.textContent = s.emoji; });
  document.querySelectorAll('.bubble.user').forEach(b => { b.style.background = s.ub; b.style.borderColor = s.ubb; });
}

// clamp mood 0-1 and refresh theme
function setMood(v) {
  mood = Math.max(0, Math.min(1, v));
  document.getElementById('moodFill').style.width = (mood * 100) + '%';
  applyTheme(getStage(mood));
}

// add a message bubble to the chat
function addMsg(text, role) {
  const msgs = document.getElementById('msgs');
  const row = document.createElement('div');
  const s = getStage(mood);
  row.className = 'msg' + (role === 'user' ? ' user' : '');
  if (role === 'bot') {
    row.innerHTML = '<div class="av" style="background:' + s.av + '">' + s.emoji + '</div><div class="bubble bot">' + text + '</div>';
  } else {
    row.innerHTML = '<div class="bubble user" style="background:' + s.ub + ';border:1px solid ' + s.ubb + '">' + text + '</div>';
  }
  msgs.appendChild(row);
  msgs.scrollTop = msgs.scrollHeight;
}

// show animated dots while waiting
function addTyping() {
  const msgs = document.getElementById('msgs');
  const row = document.createElement('div');
  const s = getStage(mood);
  row.className = 'msg'; row.id = 'typing';
  row.innerHTML = '<div class="av" style="background:' + s.av + '">' + s.emoji + '</div><div class="bubble bot" style="display:flex;gap:5px;align-items:center;padding:14px 17px;"><div class="dot" style="animation:hldot 1.2s 0s infinite ease-in-out"></div><div class="dot" style="animation:hldot 1.2s 0.18s infinite ease-in-out"></div><div class="dot" style="animation:hldot 1.2s 0.36s infinite ease-in-out"></div></div>';
  msgs.appendChild(row);
  msgs.scrollTop = msgs.scrollHeight;
}

// remove typing indicator when reply arrives
function removeTyping() { const t = document.getElementById('typing'); if (t) t.remove(); }

// toggle send button color on/off
function syncBtn() {
  const v = document.getElementById('inp').value.trim();
  const b = document.getElementById('sendBtn');
  const s = getStage(mood);
  if (v && !busy) {
    b.className = 'sendbtn on';
    b.style.background = s.btn;
    b.style.boxShadow = '0 4px 14px ' + s.glow;
  } else {
    b.className = 'sendbtn off';
    b.style.background = '';
    b.style.boxShadow = '';
  }
}

// send on enter key press
function onKey(e) { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); } }

// send message to flask backend
async function send() {
  const inp = document.getElementById('inp');
  const text = inp.value.trim();
  if (!text || busy) return;
  busy = true; inp.value = ''; syncBtn();
  addMsg(text, 'user');
  history.push({ role: 'user', content: text });
  addTyping();
  try {
    // POST to our /chat route
    const res = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, history: history.slice(-12) })
    });
    const data = await res.json();
    removeTyping();
    if (data.error) {
      addMsg('⚠️ ' + data.error, 'bot');
    } else {
      addMsg(data.reply, 'bot');
      history.push({ role: 'assistant', content: data.reply });
      // update mood based on backend response
      if (data.flagged) {
        setMood(mood - data.mood_drop);
        if (!dismissed) setTimeout(() => document.getElementById('overlay').classList.add('show'), 700);
      } else if (data.positive) {
        setMood(mood + data.mood_boost);
      } else {
        setMood(mood + 0.02);
      }
    }
  } catch(e) { removeTyping(); addMsg('⚠️ Could not reach backend.', 'bot'); }
  busy = false; syncBtn();
}

// hide the wellness popup
function closeWell() { document.getElementById('overlay').classList.remove('show'); dismissed = true; }

// switch to supportive chat mode
function talkToMe() {
  closeWell();
  const r = "I'm really glad you're still here with me. Take your time — what's going on? I'm listening. 💙";
  addMsg(r, 'bot');
  history.push({ role: 'assistant', content: r });
}

// start at neutral mood
setMood(0.65);
</script>
</body>
</html>"""

# serve the frontend
@app.route("/")
def index():
    return HTML

# start everything up
load_embeddings()
public_url = ngrok.connect(5000)
print(f"\n>>> OPEN THIS LINK: {public_url}\n")
app.run(port=5000)
