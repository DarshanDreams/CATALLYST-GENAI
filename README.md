
# Gen AI Synopsis Scorer (Privacy Aware)

A Streamlit web app that allows users to upload an article and a synopsis, then scores the synopsis against the article using NLP techniques. The app provides a similarity score, detailed visual score breakdown, and qualitative feedback, helping improve synopsis quality. It uses open-source models and respects user privacy by processing data locally.

---

## Features

- Upload article (.pdf or .txt) and synopsis (.txt) files
- Extract text content from PDFs and text files
- Calculate similarity score using sentence-transformers embeddings and cosine similarity
- Score breakdown into:
  - Content Coverage (cosine similarity-based)
  - Clarity (Flesch Reading Ease score)
  - Coherence (sentence-to-sentence semantic similarity)
- Visual display of score components with progress bars
- Password/token-based access control for app security
- Fully offline and privacy-preserving (no data sent to external servers)

---

## Requirements

- Python 3.7+
- Streamlit
- sentence-transformers
- textstat
- PyMuPDF (`fitz`)
- scikit-learn (for cosine similarity)

---

## Installation

1. Clone this repository or copy the app code.
2. Create a virtual environment and activate it (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```
3. Install dependencies:
   ```bash
   pip install streamlit sentence-transformers textstat pymupdf scikit-learn
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```

---

## Usage

1. Open the app in your browser at the URL shown after running the command.
2. Enter the access token: `Catallyst123` (can be customized in the code).
3. Upload your article file (PDF or TXT).
4. Upload your synopsis file (TXT).
5. View the previews, scores, visual breakdown, and feedback.
6. Use the feedback to improve the synopsis quality.

---

## Code Overview

- `app.py` contains the Streamlit application.
- Uses `SentenceTransformer` model `all-MiniLM-L6-v2` for embedding generation.
- Calculates:
  - Content Coverage: Cosine similarity scaled to 50 points.
  - Clarity: Based on Flesch Reading Ease score scaled to 25 points.
  - Coherence: Average similarity between consecutive synopsis sentences scaled to 25 points.
  - Total score out of 100 with a simple feedback message.
