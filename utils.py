import pdfplumber
import re


def extract_text(file):
    text = ""

    try:
        import fitz
        pdf = fitz.open(stream=file.read(), filetype="pdf")
        for page in pdf:
            text += page.get_text()
    except:
        import pdfplumber
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""

    return text


def preprocess(text):
    text = text.lower()
    
    # 🔥 synonyms
    text = text.replace("rpa", "robotic process automation")
    text = text.replace("ai", "artificial intelligence")
    text = text.replace("ml", "machine learning")

    # remove junk but keep spaces
    text = re.sub(r'[^a-zA-Z ]', ' ', text)

    # remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# 🔥 ADD THIS (IMPORTANT FOR UPGRADE)
def skill_gap(resume, jd):
    resume_words = set(resume.split())
    jd_words = set(jd.split())

    matched = resume_words & jd_words
    missing = jd_words - resume_words

    return list(matched), list(missing)