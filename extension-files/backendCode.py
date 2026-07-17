#install colab requirements
!pip install flask flask-cors openai numpy pandas pyngrok cryptography -q

#for purpose of our project, we had our data in a gdrive
from google.colab import drive
drive.mount('/content/drive')

import os, ast, re, base64
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from pyngrok import ngrok
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

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

    if "parent_comment" not in df.columns:
        return words

    for text in df["parent_comment"].dropna().astype(str):
        tokens = re.findall(r"[a-z']{4,}", text.lower())

        for token in tokens:
            # skip boring common words
            if token not in STOP_WORDS and len(token) >= 4:
                words.add(token)

    return words

# load saved embeddings from google drive
def load_embeddings():
    global stored_embeddings, stored_texts, csv_keywords

    stored_embeddings = []
    stored_texts = []
    csv_keywords = set()

    if not os.path.exists(EMBEDDINGS_CSV):
        print("CSV not found - check your path")
        return

    try:
        df = pd.read_csv(EMBEDDINGS_CSV)
    except Exception as error:
        print("Could not read CSV:", error)
        return

    if "parent_embedding" not in df.columns:
        print("parent_embedding column not found")
        return

    for _, row in df.iterrows():
        try:
            # convert string back to float array
            vector = np.array(
                ast.literal_eval(row["parent_embedding"]),
                dtype=np.float32
            )

            stored_embeddings.append(vector)
            stored_texts.append(str(row.get("parent_comment", "")))

        except Exception:
            continue

    csv_keywords = extract_csv_keywords(df)

    print(f"Loaded {len(stored_embeddings)} embeddings")
    print(f"Extracted {len(csv_keywords)} keywords from CSV")

# turn text into a vector using openai
def embed_text(text):
    text = str(text).replace("\n", " ")

    response = client.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    )

    return np.array(
        response.data[0].embedding,
        dtype=np.float32
    )

# measure how similar two vectors are
def cosine_sim(a, b):
    top = np.dot(a, b)
    bottom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-9

    return float(top / bottom)

# check if message matches distressed reddit posts
def is_distressed(embedding):
    if not stored_embeddings:
        return False, 0.0

    similarities = []

    for stored_embedding in stored_embeddings:
        if len(embedding) != len(stored_embedding):
            continue

        similarities.append(
            cosine_sim(embedding, stored_embedding)
        )

    if not similarities:
        return False, 0.0

    best = max(similarities)

    return best >= 0.78, round(best, 4)

# count keyword hits, return mood change amounts
def score_message(text):
    lower = text.lower()

    neg_hits = sum(
        1 for keyword in NEGATIVE_KEYWORDS
        if keyword in lower
    )

    pos_hits = sum(
        1 for keyword in POSITIVE_KEYWORDS
        if keyword in lower
    )

    # check against words from our reddit data
    words = set(
        re.findall(r"[a-z']{4,}", lower)
    )

    csv_hits = len(words & csv_keywords)

    is_negative = neg_hits > 0 or csv_hits >= 3
    is_positive = pos_hits > 0 and neg_hits == 0

    # cap how much mood can change
    boost = min(pos_hits * 0.05, 0.25)

    drop = min(
        neg_hits * 0.08 + (0.05 if csv_hits >= 3 else 0),
        0.3
    )

    return (
        is_negative,
        is_positive,
        round(boost, 3),
        round(drop, 3),
        neg_hits,
        pos_hits,
        csv_hits
    )

# turn a password into an encryption key
def derive_key(password, salt):
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=390000
    )

    key = kdf.derive(
        password.encode("utf-8")
    )

    return base64.urlsafe_b64encode(key)

# check a message for distress
@app.route("/check", methods=["POST"])
def check_message():
    data = request.get_json(silent=True) or {}

    user_message = str(
        data.get("message", "")
    ).strip()

    if not user_message:
        return jsonify({
            "error": "Enter a message first."
        }), 400

    (
        is_negative,
        is_positive,
        boost,
        drop,
        neg_hits,
        pos_hits,
        csv_hits
    ) = score_message(user_message)

    ml_flag = False
    similarity_score = 0.0

    try:
        # embed the message and run ml check
        embedding = embed_text(user_message)

        ml_flag, similarity_score = is_distressed(
            embedding
        )

    except Exception as error:
        # keyword detection still works if openai cannot be reached
        print("Embedding error:", error)

    flagged = ml_flag or is_negative

    return jsonify({
        "flagged": flagged,
        "positive": is_positive,
        "mood_boost": boost,
        "mood_drop": drop,
        "similarity_score": similarity_score,
        "negative_hits": neg_hits,
        "positive_hits": pos_hits,
        "csv_hits": csv_hits
    })

# encrypt text using a password
@app.route("/encrypt", methods=["POST"])
def encrypt_text():
    data = request.get_json(silent=True) or {}

    text = str(
        data.get("text", "")
    )

    password = str(
        data.get("password", "")
    )

    if not text:
        return jsonify({
            "error": "Enter text to encrypt."
        }), 400

    if not password:
        return jsonify({
            "error": "Enter an encryption password."
        }), 400

    try:
        # make a new salt so the same text does not encrypt the same way twice
        salt = os.urandom(16)

        key = derive_key(
            password,
            salt
        )

        encrypted = Fernet(key).encrypt(
            text.encode("utf-8")
        )

        combined = salt + encrypted

        encoded = base64.urlsafe_b64encode(
            combined
        ).decode("utf-8")

        return jsonify({
            "result": encoded
        })

    except Exception as error:
        print("Encryption error:", error)

        return jsonify({
            "error": "Could not encrypt the text."
        }), 500

# decrypt text using the same password
@app.route("/decrypt", methods=["POST"])
def decrypt_text():
    data = request.get_json(silent=True) or {}

    encoded = str(
        data.get("text", "")
    ).strip()

    password = str(
        data.get("password", "")
    )

    if not encoded:
        return jsonify({
            "error": "Enter encrypted text."
        }), 400

    if not password:
        return jsonify({
            "error": "Enter the decryption password."
        }), 400

    try:
        # pull the salt back out before decoding the message
        raw = base64.urlsafe_b64decode(
            encoded.encode("utf-8")
        )

        if len(raw) <= 16:
            raise ValueError

        salt = raw[:16]
        token = raw[16:]

        key = derive_key(
            password,
            salt
        )

        decrypted = Fernet(key).decrypt(
            token
        ).decode("utf-8")

        return jsonify({
            "result": decrypted
        })

    except (InvalidToken, ValueError, TypeError):
        return jsonify({
            "error": "Wrong password or damaged encrypted text."
        }), 400

    except Exception as error:
        print("Decryption error:", error)

        return jsonify({
            "error": "Could not decrypt the text."
        }), 500

# the main page html is on a separate file
return render_template("displayPage.html")

# serve the main page
@app.route("/")
def index():
    return HOME_HTML

# serve the separate calm page
@app.route("/calm")
def calm():
    return CALM_HTML

# start everything up
load_embeddings()

public_url = ngrok.connect(5000)

print(f"\n>>> OPEN THIS LINK: {public_url}\n")

app.run(
    port=5000,
    debug=False
)
