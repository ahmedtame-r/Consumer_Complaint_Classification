"""
Gradio app for the Consumer Complaint Classification model.

Loads whichever model won the comparison in the notebook (RNN variant or
DistilBERT) from the `best_model/` artifacts folder and serves predictions.

Run with:
    python app.py
"""

import os
import re
import json
import pickle
import time
from datetime import datetime

import numpy as np
import torch
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import gradio as gr

ARTIFACT_DIR = "best_model"

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Same preprocessing used to train the RNN models."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = [LEMMATIZER.lemmatize(w) for w in text.split() if w not in STOP_WORDS and len(w) > 1]
    return " ".join(tokens)


def load_artifacts(artifact_dir=ARTIFACT_DIR):
    with open(os.path.join(artifact_dir, "model_type.txt")) as f:
        model_type = f.read().strip()
    with open(os.path.join(artifact_dir, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)

    if model_type == "transformer":
        tok = AutoTokenizer.from_pretrained(artifact_dir)
        model = AutoModelForSequenceClassification.from_pretrained(artifact_dir)
        model.eval()
        cfg = None
    else:
        with open(os.path.join(artifact_dir, "tokenizer.pkl"), "rb") as f:
            tok = pickle.load(f)
        with open(os.path.join(artifact_dir, "config.json")) as f:
            cfg = json.load(f)
        model = tf.keras.models.load_model(os.path.join(artifact_dir, "rnn_model.keras"))

    return model_type, model, tok, le, cfg


MODEL_TYPE, MODEL, TOKENIZER, LABEL_ENCODER, CFG = load_artifacts()

NUM_CATEGORIES = len(LABEL_ENCODER.classes_)
MODEL_LABEL = "DistilBERT" if MODEL_TYPE == "transformer" else "RNN"
FRAMEWORK_LABEL = "PyTorch + Transformers" if MODEL_TYPE == "transformer" else "TensorFlow / Keras"


def get_best_accuracy(artifact_dir=ARTIFACT_DIR):
    """Attempts to retrieve saved test accuracy metrics from model artifacts."""
    possible_files = ["metrics.json", "config.json", "eval_results.json", "model_metrics.json"]
    for filename in possible_files:
        filepath = os.path.join(artifact_dir, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                    for key in ["test_accuracy", "accuracy", "test_acc", "acc", "val_accuracy"]:
                        if key in data and isinstance(data[key], (int, float)):
                            val = data[key]
                            return f"{val * 100:.2f}%" if val <= 1.0 else f"{val:.2f}%"
            except Exception:
                pass
    return None


ACCURACY_VALUE = get_best_accuracy()


def predict_complaint(narrative):
    """Unchanged prediction logic."""
    if not narrative or not narrative.strip():
        return "Please enter a complaint narrative.", {}

    if MODEL_TYPE == "transformer":
        inputs = TOKENIZER(narrative, max_length=128, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            probs = torch.softmax(MODEL(**inputs).logits, dim=1).numpy()[0]
    else:
        seq = TOKENIZER.texts_to_sequences([clean_text(narrative)])
        padded = pad_sequences(seq, maxlen=CFG["max_len"], padding="post", truncating="post")
        probs = MODEL.predict(padded, verbose=0)[0]

    class_probs = {LABEL_ENCODER.inverse_transform([i])[0]: float(p) for i, p in enumerate(probs)}
    top_pred = LABEL_ENCODER.inverse_transform([np.argmax(probs)])[0]
    return top_pred, class_probs


# ---------------------------------------------------------------------------
# Presentation-layer helpers
# ---------------------------------------------------------------------------

def humanize(label: str) -> str:
    return label.replace("_", " ").replace("-", " ").title()


CATEGORY_ICONS = {
    "credit_card": "💳",
    "mortgage": "🏠",
    "mortgages_and_loans": "🏠",
    "credit_reporting": "📈",
    "debt_collection": "💰",
    "retail_banking": "🏦",
    "checking_or_savings_account": "🏦",
    "money_transfers": "💸",
    "vehicle_loan_or_lease": "🚗",
    "student_loan": "🎓",
    "payday_loan": "💵",
}


def category_icon(label: str) -> str:
    return CATEGORY_ICONS.get(label, "📄")


FINANCIAL_KEYWORDS = [
    "credit card", "mortgage", "loan", "payment", "credit report",
    "checking account", "debt", "escrow", "bank", "interest",
    "fees", "fraud", "collections", "charge", "refund", "dispute",
    "balance", "overdraft", "unauthorized", "chargeback", "transaction"
]

NEGATIVE_WORDS = [
    "refuse", "refuses", "refused", "unauthorized", "wrong", "incorrect",
    "error", "fraud", "fraudulent", "never", "won't", "wont", "denied",
    "deny", "harass", "harassing", "threat", "threatened", "unfair",
    "illegal", "violation", "mistake", "fail", "failed", "ignore",
    "ignored", "overcharge", "overcharged", "scam"
]

POSITIVE_WORDS = [
    "thank", "thanks", "helpful", "resolved", "appreciation", "great",
    "excellent", "satisfied", "good", "prompt", "assisted"
]


def analyze_text(text, is_predicting=False, prediction_done=False):
    text = text or ""
    words = re.findall(r"[A-Za-z']+", text)
    word_count = len(words)
    char_count = len(text)
    sentence_count = len(re.findall(r"[.!?]+", text)) if text.strip() else 0
    est_tokens = max(0, round(len(text) / 4)) if text.strip() else 0
    lower = text.lower()
    found_keywords = sorted({kw for kw in FINANCIAL_KEYWORDS if kw in lower})

    neg_hits = sum(1 for w in NEGATIVE_WORDS if w in lower)
    pos_hits = sum(1 for w in POSITIVE_WORDS if w in lower)

    if not text.strip():
        sentiment = "—"
        status = "Idle"
    else:
        if is_predicting:
            status = "Analyzing..."
        elif prediction_done:
            status = "Prediction Complete"
        else:
            status = "Typing..."

        if pos_hits > neg_hits:
            sentiment = "Positive"
        elif neg_hits > 0:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

    return {
        "words": word_count,
        "chars": char_count,
        "sentences": sentence_count,
        "tokens": est_tokens,
        "keywords": found_keywords,
        "sentiment": sentiment,
        "status": status,
    }


def render_character_counter(text):
    a = analyze_text(text)
    return f"""
    <div class="mini-counters">
        <span><b>Words:</b> {a["words"]}</span>
        <span class="dot">&middot;</span>
        <span><b>Characters:</b> {a["chars"]}</span>
        <span class="dot">&middot;</span>
        <span><b>Estimated Tokens:</b> {a["tokens"]}</span>
    </div>
    """


def render_analytics(text, is_predicting=False, prediction_done=False):
    a = analyze_text(text, is_predicting=is_predicting, prediction_done=prediction_done)
    
    if a["keywords"]:
        chips = "".join(f'<span class="kw-chip">{kw}</span>' for kw in a["keywords"])
    else:
        chips = '<span class="kw-chip empty">No financial keywords detected</span>'

    sentiment_class = {
        "Positive": "sentiment-positive",
        "Neutral": "sentiment-neutral",
        "Negative": "sentiment-negative",
        "—": "sentiment-idle",
    }.get(a["sentiment"], "sentiment-idle")

    return f"""
    <div class="analytics-grid">
        <div class="analytics-item">
            <div class="analytics-label">Word Count</div>
            <div class="analytics-value">{a['words']}</div>
        </div>
        <div class="analytics-item">
            <div class="analytics-label">Character Count</div>
            <div class="analytics-value">{a['chars']}</div>
        </div>
        <div class="analytics-item">
            <div class="analytics-label">Sentence Count</div>
            <div class="analytics-value">{a['sentences']}</div>
        </div>
        <div class="analytics-item">
            <div class="analytics-label">Estimated Tokens</div>
            <div class="analytics-value">{a['tokens']}</div>
        </div>
        <div class="analytics-item">
            <div class="analytics-label">Sentiment</div>
            <div class="analytics-value {sentiment_class}">{a['sentiment']}</div>
        </div>
        <div class="analytics-item">
            <div class="analytics-label">Processing Status</div>
            <div class="analytics-value status-pill">{a['status']}</div>
        </div>
    </div>
    <div class="analytics-keywords">
        <div class="analytics-label">Detected Financial Keywords</div>
        <div class="kw-row">{chips}</div>
    </div>
    """


def confidence_status(pct):
    if pct >= 70:
        return "High Confidence", "status-high"
    if pct >= 40:
        return "Medium Confidence", "status-medium"
    return "Low Confidence", "status-low"


def render_result(narrative):
    start = time.time()
    top_pred, class_probs = predict_complaint(narrative)
    elapsed_ms = (time.time() - start) * 1000
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not class_probs:
        empty = f"""
        <div class="prediction-card empty">
            <div class="prediction-icon">💬</div>
            <div class="prediction-text">
                <div class="prediction-label">Waiting for input</div>
                <div class="prediction-value">{top_pred}</div>
            </div>
        </div>
        """
        return empty, "", render_analytics(narrative, prediction_done=False)

    top_prob = class_probs[top_pred]
    pct = top_prob * 100
    status_label, status_class = confidence_status(pct)

    pred_html = f"""
    <div class="prediction-card">
        <div class="prediction-top">
            <div class="prediction-icon">🏆</div>
            <div class="prediction-text">
                <div class="prediction-label">Predicted Category</div>
                <div class="prediction-value">{humanize(top_pred)}</div>
            </div>
            <div class="prediction-confidence">{pct:.1f}<span>%</span></div>
        </div>
        <div class="prediction-meta">
            <div class="meta-item">
                <div class="meta-label">Prediction Confidence</div>
                <div class="meta-value">{pct:.2f}%</div>
            </div>
            <div class="meta-item">
                <div class="meta-label">Model Used</div>
                <div class="meta-value">{MODEL_LABEL}</div>
            </div>
            <div class="meta-item">
                <div class="meta-label">Inference Time</div>
                <div class="meta-value">{elapsed_ms:.1f} ms</div>
            </div>
            <div class="meta-item">
                <div class="meta-label">Prediction Status</div>
                <div class="meta-value"><span class="status-badge {status_class}">{status_label}</span></div>
            </div>
            <div class="meta-item meta-item-wide">
                <div class="meta-label">Prediction Date & Time</div>
                <div class="meta-value">{timestamp}</div>
            </div>
        </div>
    </div>
    """

    ranked = sorted(class_probs.items(), key=lambda kv: kv[1], reverse=True)[:5]
    rows = ""
    for label, prob in ranked:
        row_pct = prob * 100
        is_top = label == top_pred
        rows += f"""
        <div class="bar-row {'top' if is_top else ''}">
            <div class="bar-row-label" title="{humanize(label)}">{category_icon(label)} {humanize(label)}</div>
            <div class="bar-track"><div class="bar-fill" style="width:{row_pct:.1f}%"></div></div>
            <div class="bar-pct">{row_pct:.1f}%</div>
        </div>
        """
    bars_html = f"""
    <div class="bars-card">
        <div class="bars-title">Confidence Breakdown &middot; Top 5</div>
        <div class="bars-wrap">{rows}</div>
    </div>
    """
    
    analytics_html = render_analytics(narrative, prediction_done=True)
    return pred_html, bars_html, analytics_html


EXAMPLES = [
    ("credit_card", "I noticed an unauthorized charge of $450 on my credit card statement and the bank refuses to refund it."),
    ("debt_collection", "The debt collection agency keeps calling me at work about a debt I already paid off in 2022."),
    ("mortgage", "My mortgage servicer misapplied my escrow payment and reported a false late payment to the credit bureaus."),
    ("credit_reporting", "There is an incorrect late payment mark on my credit report for a loan that was fully paid off."),
    ("retail_banking", "My checking account was charged overdraft fees despite having enough balance in my savings account."),
]


def render_example_cards():
    cards = ""
    for category, text in EXAMPLES:
        icon = category_icon(category)
        label = humanize(category)
        safe_text = text.replace("\\", "\\\\").replace("`", "\\`").replace("</", "<\\/")
        cards += f"""
        <div class="example-card" onclick='fillComplaintExample(`{safe_text}`)'>
            <div class="example-header">
                <div class="example-icon">{icon}</div>
                <div class="example-label">{label}</div>
            </div>
            <div class="example-text">{text}</div>
        </div>
        """
    return f'<div class="examples-grid">{cards}</div>'


# ---------------------------------------------------------------------------
# Theme & Styling
# ---------------------------------------------------------------------------

DARK_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.violet,
    secondary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="#08090D",
    body_background_fill_dark="#08090D",
    background_fill_primary="rgba(255,255,255,0.05)",
    background_fill_primary_dark="rgba(255,255,255,0.05)",
    background_fill_secondary="#111827",
    background_fill_secondary_dark="#111827",
    block_background_fill="rgba(255,255,255,0.05)",
    block_background_fill_dark="rgba(255,255,255,0.05)",
    block_border_color="rgba(255,255,255,0.08)",
    block_border_color_dark="rgba(255,255,255,0.08)",
    block_label_background_fill="rgba(255,255,255,0.06)",
    block_label_background_fill_dark="rgba(255,255,255,0.06)",
    block_label_text_color="#94A3B8",
    block_label_text_color_dark="#94A3B8",
    block_title_text_color="#F8FAFC",
    block_title_text_color_dark="#F8FAFC",
    body_text_color="#F8FAFC",
    body_text_color_dark="#F8FAFC",
    body_text_color_subdued="#94A3B8",
    body_text_color_subdued_dark="#94A3B8",
    border_color_accent="#7C5CFC",
    border_color_accent_dark="#7C5CFC",
    border_color_primary="rgba(255,255,255,0.08)",
    border_color_primary_dark="rgba(255,255,255,0.08)",
    button_primary_background_fill="linear-gradient(135deg, #7C5CFC 0%, #A855F7 100%)",
    button_primary_background_fill_dark="linear-gradient(135deg, #7C5CFC 0%, #A855F7 100%)",
    button_primary_background_fill_hover="linear-gradient(135deg, #6a4de8 0%, #9333e6 100%)",
    button_primary_background_fill_hover_dark="linear-gradient(135deg, #6a4de8 0%, #9333e6 100%)",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
    button_secondary_background_fill="rgba(255,255,255,0.06)",
    button_secondary_background_fill_dark="rgba(255,255,255,0.06)",
    button_secondary_text_color="#F8FAFC",
    button_secondary_text_color_dark="#F8FAFC",
    button_secondary_border_color="rgba(255,255,255,0.1)",
    button_secondary_border_color_dark="rgba(255,255,255,0.1)",
    input_background_fill="rgba(0,0,0,0.25)",
    input_background_fill_dark="rgba(0,0,0,0.25)",
    input_border_color="rgba(255,255,255,0.1)",
    input_border_color_dark="rgba(255,255,255,0.1)",
    input_border_color_focus="#7C5CFC",
    input_border_color_focus_dark="#7C5CFC",
    shadow_drop="0 8px 32px rgba(0, 0, 0, 0.45)",
)

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg-0: #08090D;
    --bg-1: #111827;
    --bg-2: #15162B;
    --card: rgba(255,255,255,0.05);
    --card-border: rgba(255,255,255,0.08);
    --primary: #7C5CFC;
    --accent: #A855F7;
    --success: #22C55E;
    --warning: #9CA3AF;
    --danger: #EF4444;
    --text: #F8FAFC;
    --text-sub: #94A3B8;
}

*, *::before, *::after {
    box-sizing: border-box;
}

html, body {
    margin: 0;
    padding: 0;
    overflow-x: hidden;
    width: 100%;
}

body, .gradio-container {
    background: linear-gradient(180deg, var(--bg-0) 0%, var(--bg-1) 55%, var(--bg-2) 100%) !important;
    font-family: 'Inter', ui-sans-serif, system-ui, sans-serif !important;
}

.gradio-container {
    max-width: 1440px !important;
    width: 100% !important;
    margin: 0 auto !important;
    padding: 0 32px 40px 32px !important;
    position: relative;
    overflow-x: hidden;
    box-sizing: border-box;
}

/* Ambient glowing blobs */
.gradio-container::before,
.gradio-container::after {
    content: "";
    position: fixed;
    width: 620px;
    height: 620px;
    border-radius: 50%;
    filter: blur(120px);
    z-index: 0;
    pointer-events: none;
    opacity: 0.35;
}
.gradio-container::before {
    top: -180px;
    left: -120px;
    background: radial-gradient(circle, var(--primary) 0%, transparent 70%);
}
.gradio-container::after {
    top: 400px;
    right: -160px;
    background: radial-gradient(circle, var(--accent) 0%, transparent 70%);
}

.gradio-container > * { position: relative; z-index: 1; }

footer { display: none !important; }

/* ---------- Navbar ---------- */
#navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 24px 0 0 0;
    width: 100%;
}
#navbar .logo {
    font-weight: 800;
    font-size: 1.25rem;
    color: var(--text);
    letter-spacing: -0.01em;
    line-height: 1;
}
#navbar .logo span {
    background: linear-gradient(90deg, var(--primary), var(--accent));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
#navbar .nav-badges { display: flex; gap: 12px; align-items: center; }
.nav-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--card);
    border: 1px solid var(--card-border);
    backdrop-filter: blur(15px);
    padding: 8px 16px;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--text-sub);
    white-space: nowrap;
}
.nav-badge .status-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--success);
    box-shadow: 0 0 8px var(--success);
}

/* ---------- Hero ---------- */
#hero { 
    text-align: center; 
    padding: 48px 16px 12px 16px; 
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}
#hero h1 {
    font-size: 3.2rem;
    font-weight: 800;
    line-height: 1.2;
    letter-spacing: -0.02em;
    color: var(--text);
    margin: 0 0 16px 0;
    text-align: center;
    width: 100%;
}
#hero h1 span {
    background: linear-gradient(90deg, var(--primary), var(--accent));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
#hero p {
    color: var(--text-sub);
    max-width: 680px;
    margin: 0 auto;
    font-size: 1.05rem;
    line-height: 1.65;
    text-align: center;
}

/* ---------- KPI cards ---------- */
#kpi-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 16px;
    margin: 36px auto 12px auto;
    width: 100%;
    align-items: stretch;
}
.kpi-card {
    background: var(--card);
    border: 1px solid var(--card-border);
    backdrop-filter: blur(15px);
    border-radius: 20px;
    padding: 20px 16px;
    text-align: center;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    min-height: 96px;
    box-sizing: border-box;
    transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
}
.kpi-card:hover { 
    transform: translateY(-4px); 
    border-color: rgba(124,92,252,0.5); 
    box-shadow: 0 8px 24px rgba(124,92,252,0.15);
}
.kpi-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-sub);
    margin-bottom: 6px;
    white-space: nowrap;
}
.kpi-value { 
    font-size: 1.25rem; 
    font-weight: 700; 
    color: var(--text); 
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 100%;
}

/* ---------- Section heading ---------- */
.section-heading { text-align: center; margin-top: 56px; margin-bottom: 20px; width: 100%; }
.section-heading h2 { font-size: 2.2rem; font-weight: 800; color: var(--text); letter-spacing: -0.01em; margin: 0 0 8px 0; text-align: center; }
.section-heading h2 span {
    background: linear-gradient(90deg, var(--primary), var(--accent));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.section-heading p { color: var(--text-sub); max-width: 560px; margin: 0 auto; font-size: 1rem; text-align: center; }

/* ---------- Glass cards (panels) ---------- */
#cards-row { 
    margin-top: 8px; 
    gap: 24px !important; 
    align-items: stretch !important;
}

#cards-row > .column {
    display: flex !important;
    flex-direction: column !important;
}

.glass-card {
    background: var(--card) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 24px !important;
    padding: 26px !important;
    backdrop-filter: blur(15px) !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
    min-width: 0 !important;
    box-sizing: border-box !important;
    display: flex !important;
    flex-direction: column !important;
    flex: 1 1 auto !important;
    height: 100% !important;
}

.glass-card-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 16px;
    line-height: 1.3;
}

/* ---------- Complaint textarea ---------- */
#complaint-textarea {
    flex: 1 1 auto;
    display: flex;
    flex-direction: column;
    margin-bottom: 12px;
}

#complaint-textarea > label {
    display: flex;
    flex-direction: column;
    flex: 1;
    margin: 0;
    padding: 0;
}

#complaint-textarea textarea {
    background: rgba(0,0,0,0.3) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: var(--text) !important;
    border-radius: 14px !important;
    padding: 16px !important;
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
    transition: all 0.25s ease !important;
    resize: none !important;
    box-sizing: border-box !important;
    width: 100% !important;
    flex: 1 1 auto !important;
    min-height: 220px !important;
}
#complaint-textarea textarea:focus { 
    border-color: var(--primary) !important; 
    box-shadow: 0 0 20px rgba(124, 92, 252, 0.35) !important;
    outline: none !important;
}
#complaint-textarea textarea::placeholder { 
    color: rgba(148,163,184,0.7) !important; 
}

.mini-counters {
    display: flex;
    gap: 12px;
    align-items: center;
    color: var(--text-sub);
    font-size: 0.85rem;
    margin-top: 4px;
    margin-bottom: 16px;
    white-space: nowrap;
    overflow-x: auto;
}
.mini-counters b { color: var(--text); }
.mini-counters .dot { opacity: 0.4; }

/* ---------- Buttons ---------- */
#action-row { 
    margin-top: auto; 
    gap: 12px !important; 
    display: flex !important;
    align-items: center !important;
    width: 100% !important;
}
#action-row button {
    border-radius: 14px !important;
    font-weight: 700 !important;
    height: 48px !important;
    min-height: 48px !important;
    max-height: 48px !important;
    padding: 0 20px !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    flex: 1 1 0% !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    box-sizing: border-box !important;
    margin: 0 !important;
}
#action-row button:hover { transform: translateY(-2px); scale: 1.01; }
#action-row button:active { transform: translateY(0); scale: 0.99; }
.gr-button-primary, button.primary {
    box-shadow: 0 6px 24px rgba(124,92,252,0.4) !important;
}
.gr-button-primary:hover, button.primary:hover {
    box-shadow: 0 10px 32px rgba(168,85,247,0.5) !important;
}

/* ---------- Right Panel & Predictions ---------- */
.right-panel-content {
    display: flex;
    flex-direction: column;
    justify-content: center;
    flex: 1 1 auto;
    width: 100%;
}

.prediction-card {
    background: linear-gradient(135deg, rgba(124,92,252,0.18) 0%, rgba(168,85,247,0.10) 100%);
    border: 1px solid rgba(124,92,252,0.35);
    border-radius: 20px;
    padding: 22px;
    animation: fadeIn 0.5s ease;
    width: 100%;
    box-sizing: border-box;
}
.prediction-card.empty {
    background: rgba(255,255,255,0.03);
    border: 1px dashed var(--card-border);
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 16px;
    padding: 32px 22px;
    min-height: 180px;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}
.prediction-top { 
    display: flex; 
    align-items: center; 
    gap: 16px; 
    width: 100%;
}
.prediction-icon {
    width: 48px; height: 48px; min-width: 48px;
    border-radius: 14px;
    background: rgba(255,255,255,0.12);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.5rem;
    flex-shrink: 0;
}
.prediction-text { flex: 1 1 auto; min-width: 0; }
.prediction-label {
    font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em;
    color: var(--text-sub); margin-bottom: 2px;
}
.prediction-value { 
    font-size: 1.35rem; 
    font-weight: 800; 
    color: var(--text);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    line-height: 1.25;
}
.prediction-confidence { 
    font-size: 1.8rem; 
    font-weight: 800; 
    color: var(--text);
    white-space: nowrap;
    flex-shrink: 0;
}
.prediction-confidence span { font-size: 1rem; font-weight: 600; opacity: 0.7; }

.prediction-meta {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 14px;
    margin-top: 20px;
    padding-top: 18px;
    border-top: 1px solid rgba(255,255,255,0.08);
}
.meta-item { display: flex; flex-direction: column; justify-content: center; min-width: 0; }
.meta-item-wide { grid-column: span 2; }
.meta-label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--text-sub); margin-bottom: 4px; white-space: nowrap; }
.meta-value { font-size: 0.95rem; font-weight: 700; color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.status-badge { display: inline-block; padding: 4px 10px; border-radius: 999px; font-size: 0.78rem; font-weight: 700; line-height: 1; }
.status-high { background: rgba(34,197,94,0.18); color: var(--success); }
.status-medium { background: rgba(245,158,11,0.18); color: #f59e0b; }
.status-low { background: rgba(239,68,68,0.18); color: var(--danger); }

/* ---------- Confidence bars ---------- */
.bars-card { margin-top: 20px; width: 100%; }
.bars-title {
    font-size: 0.78rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.06em; color: var(--text-sub); margin-bottom: 14px;
}
.bars-wrap { display: flex; flex-direction: column; gap: 12px; }
.bar-row { display: grid; grid-template-columns: 160px 1fr 50px; align-items: center; gap: 12px; min-height: 24px; }
.bar-row-label { 
    font-size: 0.85rem; 
    color: var(--text-sub); 
    white-space: nowrap; 
    overflow: hidden; 
    text-overflow: ellipsis; 
    display: flex;
    align-items: center;
    gap: 6px;
}
.bar-row.top .bar-row-label { color: var(--text); font-weight: 700; }
.bar-track { height: 9px; background: rgba(255,255,255,0.06); border: 1px solid var(--card-border); border-radius: 999px; overflow: hidden; }
.bar-fill {
    height: 100%; border-radius: 999px;
    background: linear-gradient(90deg, #5a3ccf, var(--accent));
    width: 0%;
    animation: growBar 0.9s cubic-bezier(0.22, 1, 0.36, 1) forwards;
}
@keyframes growBar { from { width: 0%; } }
.bar-row.top .bar-fill { 
    background: linear-gradient(90deg, var(--primary), #d7b8ff); 
    box-shadow: 0 0 12px rgba(168, 85, 247, 0.6);
}
.bar-pct { font-size: 0.82rem; font-weight: 600; color: var(--text-sub); text-align: right; white-space: nowrap; }
.bar-row.top .bar-pct { color: #d7b8ff; font-weight: 700; }

/* ---------- Analytics ---------- */
.analytics-grid {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 16px;
    width: 100%;
}
.analytics-item {
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--card-border);
    border-radius: 16px;
    padding: 16px 14px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    min-height: 80px;
    box-sizing: border-box;
}
.analytics-label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--text-sub); margin-bottom: 6px; white-space: nowrap; }
.analytics-value { font-size: 1.15rem; font-weight: 700; color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; line-height: 1.2; }
.sentiment-positive { color: var(--success); }
.sentiment-neutral { color: var(--warning); }
.sentiment-negative { color: var(--danger); }
.sentiment-idle { color: var(--text-sub); }
.status-pill { color: #d7b8ff; }

.analytics-keywords { margin-top: 20px; }
.kw-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; min-height: 32px; align-items: center; }
.kw-chip {
    background: rgba(124,92,252,0.15);
    border: 1px solid rgba(124,92,252,0.3);
    color: #d7b8ff;
    padding: 6px 14px;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    white-space: nowrap;
}
.kw-chip.empty { background: rgba(255,255,255,0.05); border-color: var(--card-border); color: var(--text-sub); }

/* ---------- Example cards ---------- */
.examples-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
    gap: 16px;
    margin-top: 8px;
    width: 100%;
}
.example-card {
    background: var(--card);
    border: 1px solid var(--card-border);
    backdrop-filter: blur(15px);
    border-radius: 20px;
    padding: 20px;
    cursor: pointer;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    transition: transform 0.25s ease, border-color 0.25s ease, background 0.25s ease, box-shadow 0.25s ease;
    box-sizing: border-box;
    height: 100%;
}
.example-card:hover {
    transform: translateY(-4px) scale(1.01);
    border-color: rgba(124,92,252,0.55);
    background: linear-gradient(160deg, rgba(124,92,252,0.14), rgba(168,85,247,0.06));
    box-shadow: 0 10px 30px rgba(124,92,252,0.3);
}
.example-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 12px;
}
.example-icon { font-size: 1.4rem; line-height: 1; flex-shrink: 0; }
.example-label { 
    font-weight: 700; 
    color: var(--text); 
    font-size: 0.95rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.example-text {
    font-size: 0.85rem;
    color: var(--text-sub);
    line-height: 1.5;
    display: -webkit-box;
    -webkit-line-clamp: 4;
    -webkit-box-orient: vertical;
    overflow: hidden;
    margin: 0;
}

/* ---------- Footer ---------- */
#footer {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    color: var(--text-sub);
    font-size: 0.88rem;
    padding: 50px 0 10px 0;
    width: 100%;
}
#footer .stack {
    display: flex; 
    justify-content: center; 
    align-items: center;
    gap: 10px; 
    margin-top: 14px; 
    flex-wrap: wrap;
}
#footer .stack span {
    background: var(--card);
    border: 1px solid var(--card-border);
    padding: 6px 16px;
    border-radius: 999px;
    font-weight: 600;
    color: var(--text);
    font-size: 0.8rem;
    white-space: nowrap;
}

/* ---------- Responsiveness ---------- */
@media (max-width: 1280px) {
    .analytics-grid { grid-template-columns: repeat(3, 1fr); }
}
@media (max-width: 1024px) {
    .gradio-container { padding: 0 24px 32px 24px !important; }
    .analytics-grid { grid-template-columns: repeat(3, 1fr); }
    .bar-row { grid-template-columns: 140px 1fr 48px; }
}
@media (max-width: 768px) {
    .gradio-container { padding: 0 16px 24px 16px !important; }
    #hero h1 { font-size: 2.2rem; }
    #hero p { font-size: 0.95rem; }
    #kpi-row { grid-template-columns: repeat(2, 1fr); }
    .analytics-grid { grid-template-columns: repeat(2, 1fr); }
    .prediction-meta { grid-template-columns: 1fr; }
    .meta-item-wide { grid-column: span 1; }
    .bar-row { grid-template-columns: 120px 1fr 44px; }
    #action-row { flex-direction: row; }
}
@media (max-width: 480px) {
    #kpi-row { grid-template-columns: 1fr; }
    .analytics-grid { grid-template-columns: 1fr; }
    .bar-row { grid-template-columns: 100px 1fr 40px; }
    #navbar { flex-direction: column; gap: 12px; align-items: flex-start; }
}
"""

FORCE_DARK_HEAD = """
<script>
document.documentElement.classList.add('dark');
function fillComplaintExample(text) {
    const ta = document.querySelector('#complaint-textarea textarea');
    if (!ta) return;
    const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
    setter.call(ta, text);
    ta.dispatchEvent(new Event('input', { bubbles: true }));
    ta.focus();
}
</script>
"""

with gr.Blocks(theme=DARK_THEME, css=CUSTOM_CSS, head=FORCE_DARK_HEAD, title="Consumer Complaint Classification") as demo:

    # Top Navbar & Hero
    accuracy_card_html = f'<div class="kpi-card"><div class="kpi-label">Accuracy</div><div class="kpi-value">{ACCURACY_VALUE}</div></div>' if ACCURACY_VALUE else ""
    
    gr.HTML(
        f"""
        <div id="navbar">
            <div class="logo">ComplaintAI</div>
            <div class="nav-badges">
                <div class="nav-badge">🤖 {MODEL_LABEL}</div>
                <div class="nav-badge"><span class="status-dot"></span> Online</div>
            </div>
        </div>
        <div id="hero">
            <h1>Consumer Complaint Classification</h1>
            <p>
                AI-powered NLP system that automatically classifies consumer financial
                complaints into the correct CFPB category using a fine-tuned {MODEL_LABEL} model.
            </p>
        </div>
        <div id="kpi-row">
            <div class="kpi-card"><div class="kpi-label">Model</div><div class="kpi-value">{MODEL_LABEL}</div></div>
            {accuracy_card_html}
            <div class="kpi-card"><div class="kpi-label">Categories</div><div class="kpi-value">{NUM_CATEGORIES}</div></div>
            <div class="kpi-card"><div class="kpi-label">Framework</div><div class="kpi-value">{FRAMEWORK_LABEL}</div></div>
            <div class="kpi-card"><div class="kpi-label">Inference</div><div class="kpi-value">Real Time</div></div>
        </div>
        """
    )

    # Section 1: Try the Classifier
    with gr.Column(elem_classes="section-heading"):
        gr.Markdown("## Try the Classifier")
        gr.Markdown("Paste a complaint narrative below and the model will predict its category.")

    with gr.Row(equal_height=True, elem_id="cards-row"):
        with gr.Column(scale=5, elem_classes="glass-card"):
            gr.Markdown('<div class="glass-card-title">📝 Complaint Narrative</div>')
            narrative_input = gr.Textbox(
                lines=8,
                placeholder="Describe your financial complaint here...",
                label=None,
                show_label=False,
                elem_id="complaint-textarea",
            )
            counters = gr.HTML(value=render_character_counter(""))
            with gr.Row(elem_id="action-row"):
                clear_btn = gr.Button("Reset", variant="secondary")
                submit_btn = gr.Button("Predict Category 🚀", variant="primary")

        with gr.Column(scale=5, elem_classes="glass-card"):
            with gr.Column(elem_classes="right-panel-content"):
                pred_output = gr.HTML(value=render_result("")[0])
                probs_output = gr.HTML(value="")

    # Section 2: Complaint Analysis
    with gr.Column(elem_classes="section-heading"):
        gr.Markdown("## Complaint Analysis")

    with gr.Row():
        with gr.Column(elem_classes="glass-card"):
            gr.Markdown('<div class="glass-card-title">📊 Real-Time Narrative Breakdown</div>')
            analytics_output = gr.HTML(value=render_analytics(""))

    # Section 3: Example Complaints
    with gr.Column(elem_classes="section-heading"):
        gr.Markdown("## Example Complaints")
        gr.Markdown("Click a card to auto-fill the narrative field and analyze instantly.")

    gr.HTML(render_example_cards())

    # Footer
    gr.HTML(
        """
        <div id="footer">
            Built with ❤️ using
            <div class="stack">
                <span>Gradio</span><span>PyTorch</span><span>HuggingFace Transformers</span><span>DistilBERT</span>
            </div>
        </div>
        """
    )

    # Handlers & Callbacks
    def on_input(text):
        return render_character_counter(text), render_analytics(text)

    narrative_input.input(fn=on_input, inputs=narrative_input, outputs=[counters, analytics_output])

    def on_submit_start(text):
        # Immediate UI state update when predicting starts
        return render_analytics(text, is_predicting=True)

    submit_btn.click(
        fn=on_submit_start,
        inputs=narrative_input,
        outputs=analytics_output,
    ).then(
        fn=render_result,
        inputs=narrative_input,
        outputs=[pred_output, probs_output, analytics_output]
    )

    narrative_input.submit(
        fn=on_submit_start,
        inputs=narrative_input,
        outputs=analytics_output,
    ).then(
        fn=render_result,
        inputs=narrative_input,
        outputs=[pred_output, probs_output, analytics_output]
    )

    def on_clear():
        return "", render_character_counter(""), render_analytics(""), *render_result("")

    clear_btn.click(
        fn=on_clear,
        inputs=None,
        outputs=[narrative_input, counters, analytics_output, pred_output, probs_output, analytics_output],
    )

if __name__ == "__main__":
    demo.launch(share=False)