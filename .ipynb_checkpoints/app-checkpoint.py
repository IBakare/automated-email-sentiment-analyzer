import os
import pandas as pd
from textblob import TextBlob
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import streamlit as st
from collections import Counter
import re

# -----------------------------------
# STREAMLIT PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="Email Sentiment Dashboard",
    layout="wide",
    menu_items=None
)




# -----------------------------------
# GLOBAL CSS + JS (cyberpunk theme + matrix background)
# -----------------------------------
st.markdown("""
<canvas id="matrixCanvas"></canvas>

<style>

/* CYBERPUNK BACKGROUND */
body {
    background-color: #000;
}

/* App container */
[data-testid="stAppViewContainer"] {
    background-color: #000;
}

/* Matrix canvas */
#matrixCanvas {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    z-index: -1;
    pointer-events: none;
}

/* Sidebar neon (collapsible) */
[data-testid="stSidebar"] {
    background-color: #000;
    border-right: 2px solid #00ffea;
    box-shadow: 0 0 20px #00ffea;
    padding: 25px 20px !important;
    overflow-x: hidden !important;
}

/* Sidebar text: labels only, so icon fonts aren't overridden */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label {
    color: #00ffea !important;
    font-family: 'Courier New', monospace !important;
}

/* Collapse/expand arrow: keep it as an icon, not text */
[data-testid="stIconMaterial"] {
    font-family: "Material Symbols Rounded" !important;
    color: #00ffea !important;
}

/* Subtle neon glow on sidebar edge */
[data-testid="stSidebar"]::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 2px;
    height: 100%;
    background: #00ffea;
    box-shadow: 0 0 15px #00ffea;
}

/* Neon glow: text inputs and text areas */
[data-baseweb="input"],
[data-baseweb="base-input"],
[data-baseweb="textarea"] {
    background-color: #0a0a0a !important;
    border: 1px solid #00ffea !important;
    box-shadow: 0 0 10px #00ffea !important;
    border-radius: 6px !important;
}
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    color: #00ffea !important;
    font-family: 'Courier New', monospace !important;
}

/* Neon glow: buttons */
[data-testid="stButton"] button,
[data-testid="stDownloadButton"] button {
    background-color: #000 !important;
    color: #00ffea !important;
    border: 1px solid #00ffea !important;
    box-shadow: 0 0 12px #00ffea !important;
    border-radius: 6px !important;
}
[data-testid="stButton"] button:hover,
[data-testid="stDownloadButton"] button:hover {
    box-shadow: 0 0 22px #00ffea !important;
    background-color: #00ffea !important;
    color: #000 !important;
}

</style>



<script>
function toggleSidebar() {
    const sidebar = parent.document.querySelector('[data-testid="stSidebar"]');
    if (!sidebar) return;

    const isHidden = sidebar.style.display === "none";
    sidebar.style.display = isHidden ? "block" : "none";
}
</script>



<script>
// Matrix background
var canvas = document.getElementById('matrixCanvas');
var ctx = canvas.getContext('2d');
function resizeCanvas() {
    canvas.height = window.innerHeight;
    canvas.width = window.innerWidth;
}
resizeCanvas();
window.onresize = resizeCanvas;

var letters = '01';
letters = letters.split('');
var fontSize = 14;
var columns = canvas.width / fontSize;
var drops = [];
for (var i = 0; i < columns; i++) drops[i] = 1;

function drawMatrix() {
    ctx.fillStyle = 'rgba(0, 0, 0, 0.05)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#00ffea';
    ctx.font = fontSize + 'px monospace';
    for (var i = 0; i < drops.length; i++) {
        var text = letters[Math.floor(Math.random() * letters.length)];
        ctx.fillText(text, i * fontSize, drops[i] * fontSize);
        if (drops[i] * fontSize > canvas.height && Math.random() > 0.975) drops[i] = 0;
        drops[i]++;
    }
}
setInterval(drawMatrix, 33);
</script>
""", unsafe_allow_html=True)

# Title
st.title("📧 Email Sentiment Dashboard")

# -----------------------------------
# SIDEBAR MODE SWITCH
# -----------------------------------
mode = st.sidebar.radio(
    "Choose Mode",
    ["Dynamic Mode", "IMAP Mode", "Dataset Mode"]
)

# -----------------------------------
# SENTIMENT + EMOTION FUNCTIONS
# -----------------------------------
def categorize(p):
    if p > 0:
        return "Positive"
    elif p < 0:
        return "Negative"
    return "Neutral"

def detect_emotion(text):
    text = text.lower()
    emotions = {
        "joy": ["happy", "joy", "delighted", "pleased", "excited"],
        "anger": ["angry", "furious", "irritated", "annoyed"],
        "fear": ["afraid", "scared", "fearful", "terrified"],
        "sadness": ["sad", "unhappy", "depressed", "down"],
        "surprise": ["surprised", "shocked", "amazed"]
    }
    detected = []
    for emotion, words in emotions.items():
        if any(w in text for w in words):
            detected.append(emotion)
    return detected if detected else ["neutral"]

def extract_keywords(text, top_n=10):
    words = re.findall(r"\b\w+\b", text.lower())
    stopwords = set(["the", "and", "is", "to", "in", "of", "for", "on", "with", "a", "an"])
    filtered = [w for w in words if w not in stopwords]
    freq = Counter(filtered)
    return freq.most_common(top_n)

# -----------------------------------
# COLOUR + TOPIC FUNCTIONS
# -----------------------------------
def color_sentiment(val):
    if val == "Positive":
        return "color: green; font-weight: bold;"
    elif val == "Negative":
        return "color: red; font-weight: bold;"
    return "color: grey; font-weight: bold;"

palette = {
    "Positive": "green",
    "Negative": "red",
    "Neutral": "grey"
}

def wc_background(category):
    if category == "Positive":
        return "#e6ffe6"
    elif category == "Negative":
        return "#ffe6e6"
    return "#f2f2f2"

def classify_topic(text):
    text = text.lower()
    support_words = ["help", "issue", "problem", "support", "fix", "error", "unable", "not working"]
    complaint_words = ["angry", "disappointed", "frustrated", "complaint", "bad", "terrible", "refund"]
    hr_words = ["leave", "holiday", "vacation", "salary", "payroll", "hr", "manager", "team"]
    marketing_words = ["offer", "promotion", "discount", "newsletter", "campaign", "marketing"]

    if any(w in text for w in support_words):
        return "Support"
    if any(w in text for w in complaint_words):
        return "Complaint"
    if any(w in text for w in hr_words):
        return "HR"
    if any(w in text for w in marketing_words):
        return "Marketing"
    return "General"

def color_topic(val):
    colors = {
        "Support": "blue",
        "Complaint": "red",
        "HR": "purple",
        "Marketing": "orange",
        "General": "grey"
    }
    return f"color: {colors.get(val, 'black')}; font-weight: bold;"

def threat_level(df):
    avg = df["polarity"].mean()
    if avg > 0.2:
        return "LOW", "#00ff00"
    elif avg > -0.2:
        return "MEDIUM", "#ffff00"
    return "HIGH", "#ff0000"

# -----------------------------------
# MODE 1: DYNAMIC MODE
# -----------------------------------
if mode == "Dynamic Mode":
    st.header("🔍 Live Sentiment Analyzer")

    user_text = st.text_area("Type or paste any text here:")

    if st.button("Analyze Text"):
        if user_text.strip() == "":
            st.warning("Please enter some text.")
        else:
            blob = TextBlob(user_text)
            polarity = blob.sentiment.polarity
            category = categorize(polarity)
            emotions = detect_emotion(user_text)
            keywords = extract_keywords(user_text)

            st.subheader("Sentiment Result")
            st.write(f"**Polarity Score:** {polarity}")
            st.write(f"**Category:** {category}")
            st.write(f"**Detected Emotions:** {', '.join(emotions)}")

            st.subheader("Top Keywords")
            st.write(keywords)

            wc = WordCloud(width=800, height=400, background_color=wc_background(category)).generate(user_text)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

# -----------------------------------
# MODE 2: IMAP MODE
# -----------------------------------
elif mode == "IMAP Mode":
    st.header("📥 IMAP Email Sentiment Analyzer")

    st.info("Enter your IMAP credentials below")

    host = st.text_input("IMAP Host", "imap.gmail.com")
    username = st.text_input("Email Address")
    password = st.text_input("App Password", type="password")
    folder = st.text_input("Folder Name", "INBOX")

    if st.button("Fetch Emails"):
        try:
            from imapclient import IMAPClient
            import pyzmail

            email_texts = []

            with IMAPClient(host) as client:
                client.login(username, password)
                client.select_folder(folder)
                messages = client.search(["ALL"])

                for msgid, data in client.fetch(messages, ["RFC822"]).items():
                    message = pyzmail.PyzMessage.factory(data[b"RFC822"])

                    if message.text_part:
                        body = message.text_part.get_payload().decode(message.text_part.charset)
                    elif message.html_part:
                        body = message.html_part.get_payload().decode(message.html_part.charset)
                    else:
                        body = "(No readable text content)"

                    email_texts.append((f"email_{msgid}", body))

            st.success(f"Fetched {len(email_texts)} emails!")

            rows = []
            for filename, text in email_texts:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                category = categorize(polarity)
                emotions = detect_emotion(text)
                topic = classify_topic(text)
                rows.append([filename, text, polarity, category, emotions, topic])

            df = pd.DataFrame(rows, columns=["filename", "text", "polarity", "category", "emotions", "topic"])

            # Threat level
            level, color = threat_level(df)
            st.markdown(f"""
            <div class="threat-panel" style="border:2px solid {color}; box-shadow:0 0 20px {color}; color:{color};">
                THREAT LEVEL: <b>{level}</b>
            </div>
            """, unsafe_allow_html=True)

            # Styled Table
            st.subheader("📄 Email Table")
            st.dataframe(
                df.style.apply(lambda col: col.map(color_sentiment), subset=["category"])
                        .apply(lambda col: col.map(color_topic), subset=["topic"])
            )

            # Download CSV
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", csv, "imap_sentiment.csv", "text/csv")

            # Dashboard
            st.subheader("📊 Sentiment Dashboard")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(data=df, x="filename", y="polarity", hue="category", palette=palette, ax=ax)
            ax.axhline(0, color="black")
            plt.xticks(rotation=45)
            st.pyplot(fig)

            # Wordclouds
            st.subheader("☁ Wordclouds")
            for idx, row in df.iterrows():
                bg = wc_background(row["category"])
                wc = WordCloud(width=800, height=400, background_color=bg).generate(row["text"])
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                st.write(f"### {row['filename']}")
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Error: {e}")

# -----------------------------------
# MODE 3: DATASET MODE
# -----------------------------------
elif mode == "Dataset Mode":
    st.header("📁 Folder-Based Email Dashboard")

    folder = st.text_input("Folder Path", "emails")

    if st.button("Analyze Folder"):
        email_texts = []

        for filename in os.listdir(folder):
            if filename.endswith(".txt"):
                with open(os.path.join(folder, filename), "r", encoding="utf-8") as f:
                    email_texts.append((filename, f.read()))

        rows = []
        for filename, text in email_texts:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            category = categorize(polarity)
            emotions = detect_emotion(text)
            topic = classify_topic(text)
            rows.append([filename, text, polarity, category, emotions, topic])

        df = pd.DataFrame(rows, columns=["filename", "text", "polarity", "category", "emotions", "topic"])

        # Threat level
        level, color = threat_level(df)
        st.markdown(f"""
        <div class="threat-panel" style="border:2px solid {color}; box-shadow:0 0 20px {color}; color:{color};">
            THREAT LEVEL: <b>{level}</b>
        </div>
        """, unsafe_allow_html=True)

        # Dashboard
        st.subheader("📊 Sentiment Dashboard")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=df, x="filename", y="polarity", hue="category", palette=palette, ax=ax)
        ax.axhline(0, color="black")
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Styled Table
        st.subheader("📄 Table")
        st.dataframe(
            df.style.apply(lambda col: col.map(color_sentiment), subset=["category"])
                    .apply(lambda col: col.map(color_topic), subset=["topic"])
        )

        # Download CSV
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "dataset_sentiment.csv", "text/csv")

        # Wordclouds
        st.subheader("☁ Wordclouds")
        for idx, row in df.iterrows():
            bg = wc_background(row["category"])
            wc = WordCloud(width=800, height=400, background_color=bg).generate(row["text"])
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.write(f"### {row['filename']}")
            st.pyplot(fig)
