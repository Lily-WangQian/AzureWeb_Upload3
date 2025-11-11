from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import re
import string
import pdfplumber
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from PyPDF2 import PdfReader
import yake  # Contextual keyword extraction

app = Flask(__name__)

# ------------ Config ------------
UPLOAD_FOLDER = "uploads"
ALLOWED_EXT = {".pdf"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ------------ Load standards ------------
df = pd.read_csv("standards keywords.csv")
df.columns = df.columns.str.strip()

if "Standards" in df.columns and "Standard" not in df.columns:
    df.rename(columns={"Standards": "Standard"}, inplace=True)
for c in list(df.columns):
    if c.lower().startswith("publication") and "Publication Date" not in df.columns:
        df.rename(columns={c: "Publication Date"}, inplace=True)

EXPECTED = ["Standard", "Publication Date", "TFIDF Keywords", "Contextual Keywords"]
for col in EXPECTED:
    if col not in df.columns:
        df[col] = ""

standards_list = sorted(df["Standard"].dropna().unique().tolist())

# ------------ Utility functions ------------

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
    except Exception:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text.strip()

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.lower()

def extract_tfidf_keywords(text, top_n=10):
    if not text.strip():
        return []
    vectorizer = TfidfVectorizer(stop_words="english", max_features=top_n)
    try:
        X = vectorizer.fit_transform([text])
        return vectorizer.get_feature_names_out().tolist()
    except Exception:
        return []

def extract_contextual_keywords(text, top_n=5):
    """
    Extract top-N bigram contextual keywords using YAKE.
    """
    if not text or not text.strip():
        return []
    kw_extractor = yake.KeywordExtractor(lan="en", n=2, top=top_n)
    keywords = [kw for kw, score in kw_extractor.extract_keywords(text)]
    return keywords

def detect_publication_date(text):
    match = re.search(r"(20\d{2}|19\d{2})", text)
    return match.group(0) if match else "Unknown"

# ------------ Routes ------------

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", standards=standards_list)

@app.route("/analyze", methods=["POST"])
def analyze():
    if "bank_pdf" not in request.files:
        return render_template("index.html", error="Please upload a PDF file.", standards=standards_list)

    file = request.files["bank_pdf"]
    if file.filename == "":
        return render_template("index.html", error="No file selected.", standards=standards_list)

    std = request.form.get("standard")
    if not std:
        return render_template("index.html", error="Please select a standard.", standards=standards_list)

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        return render_template("index.html", error="Invalid file type.", standards=standards_list)

    # Save and extract text
    fname = secure_filename(file.filename)
    fpath = os.path.join(app.config["UPLOAD_FOLDER"], fname)
    file.save(fpath)

    text = extract_text_from_pdf(fpath)
    cleaned_text = clean_text(text)
    preview = text[:1500] + "..." if len(text) > 1500 else text

    # Extract keywords
    bank_pub_date = detect_publication_date(text)
    bank_tfidf_list = extract_tfidf_keywords(cleaned_text)
    bank_contextual_list = extract_contextual_keywords(cleaned_text, top_n=5)

    # Convert to strings
    bank_tfidf_str = ", ".join(bank_tfidf_list)
    bank_contextual_str = ", ".join(bank_contextual_list)

    # Match the selected standard
    std_row = df[df["Standard"] == std].iloc[0] if not df[df["Standard"] == std].empty else None
    std_info = None
    if std_row is not None:
        std_info = {
            "standard": std_row["Standard"],
            "pub_date": std_row["Publication Date"],
            "tfidf": std_row["TFIDF Keywords"],
            "contextual": std_row["Contextual Keywords"],
        }

    result = {
        "filename": fname,
        "standard": std,
        "preview": preview,
        "bank_pub_date": bank_pub_date,
        "bank_tfidf": bank_tfidf_str,
        "bank_contextual": bank_contextual_str,
    }

    return render_template(
        "index.html",
        result=result,
        std_info=std_info,
        standards=standards_list,
        selected=std,
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
