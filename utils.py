import fitz  
import spacy
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('all-MiniLM-L6-v2')

def read_txt(file):
    return file.read().decode("utf-8")

def read_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return "\n".join([page.get_text() for page in doc])

def extract_text(file):
    if file.name.endswith(".pdf"):
        return read_pdf(file)
    return read_txt(file)

def anonymize(text):
    doc = nlp(text)
    for ent in doc.ents:
        text = text.replace(ent.text, f"[{ent.label_}]")
    text = re.sub(r"\b\d{4}\b", "[YEAR]", text)
    return text

def compute_score(article, synopsis):
    article_embedding = model.encode([article])[0]
    synopsis_embedding = model.encode([synopsis])[0]
    similarity = cosine_similarity([article_embedding], [synopsis_embedding])[0][0]
    content_score = round(similarity * 50)

    sentences = synopsis.split('.')
    clarity_score = 25 if all(len(s.split()) < 30 for s in sentences if s.strip()) else 15
    coherence_score = 25  

    total = content_score + clarity_score + coherence_score
    return {
        "Content Coverage": content_score,
        "Clarity": clarity_score,
        "Coherence": coherence_score,
        "Total": total
    }

def feedback_text(score):
    if score['Total'] > 80:
        return "Excellent synopsis! Great coverage and clarity."
    elif score['Total'] > 60:
        return "Good effort. Consider improving clarity and structure."
    else:
        return "Needs improvement. Try focusing on key points with better flow."