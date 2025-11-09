from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import re
import pdfplumber
import pandas as pd

app = Flask(__name__)

# ------------ Config ------------
UPLOAD_FOLDER = "uploads"
ALLOWED_EXT = {".pdf"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ------------ Load standards ------------
# CSV must contain (or be mappable to):
#   Standard | Publication Date | TFIDF Keywords | Contextual Keywords
df = pd.read_csv("standards keywords.csv")
df.columns = df.columns.str.strip()

# normalize possible variants
if "Standards" in df.columns and "Standard" not in df.columns:
    df.rename(columns={"Standards": "Standard"}, inplace=True)
for c in list(df.columns):
    if c.lower().startswith("publication") and "Publication Date" not in df.columns:
        df.rename(columns={c: "Publication Date"}, inplace=True)

EXPECTED = ["Standard", "Publication Date", "TFIDF Keywords", "Contextual Keywords"]
for col in EXPECTED:
    if col not in df.columns:
        df[col] = ""

standards = (
    df["Standard"].dropna().astype(str).str.strip().sort_values().unique().tolist()
)

# ------------ Helpers ------------
def allowed_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXT

def extract_text_from_pdf(path: str, max_chars: int = 2000) -> str:
    """Quick extraction for preview."""
    parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            parts.append(txt)
            if sum(len(p) for p in parts) >= max_chars:
                break
    raw = "\n".join(parts)
    clean = re.sub(r"\s+", " ", raw).strip()
    return clean[:max_chars] if clean else "(no text extracted)"

def lookup_standard(std_name: str) -> dict | None:
    row = df.loc[df["Standard"].astype(str).str.strip() == std_name].head(1)
    if row.empty:
        return None
    r = row.iloc[0]
    return {
        "standard": r.get("Standard", ""),
        "pub_date": r.get("Publication Date", ""),
        "tfidf": r.get("TFIDF Keywords", ""),
        "contextual": r.get("Contextual Keywords", "")
    }

# ------------ Routes ------------
@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        standards=standards,
        selected=None,
        result=None,
        std_info=None,
        error=None,
        message=None,
    )

@app.route("/analyze", methods=["POST"])
def analyze():
    std = (request.form.get("standard") or "").strip()
    pdf_file = request.files.get("bank_pdf")

    if not std:
        return render_template("index.html", standards=standards, selected=None,
                               result=None, std_info=None,
                               error="Please select a standard.", message=None)

    if not pdf_file or pdf_file.filename == "":
        return render_template("index.html", standards=standards, selected=std,
                               result=None, std_info=None,
                               error="Please upload a bank ESG report (PDF).", message=None)

    if not allowed_file(pdf_file.filename):
        return render_template("index.html", standards=standards, selected=std,
                               result=None, std_info=None,
                               error="The uploaded file should be a PDF.", message=None)

    fname = secure_filename(pdf_file.filename)
    fpath = os.path.join(app.config["UPLOAD_FOLDER"], fname)
    pdf_file.save(fpath)

    preview = extract_text_from_pdf(fpath, max_chars=2000)
    std_info = lookup_standard(std)

    result = {"filename": fname, "standard": std, "preview": preview}
    return render_template("index.html", standards=standards, selected=std,
                           result=result, std_info=std_info, error=None, message=None)

if __name__ == "__main__":
    # Azure uses WEBSITES_PORT in production; 8000 is good locally.
    app.run(host="0.0.0.0", port=8000)
