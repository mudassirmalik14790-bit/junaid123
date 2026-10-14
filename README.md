# 📄 AI Resume ATS Checker

A Streamlit app that scores a resume for Applicant Tracking System (ATS) compatibility
and suggests concrete improvements, powered by Google's **Gemini Flash** model.

## Features
- Upload a resume as **PDF, DOCX or TXT**
- Overall **ATS score (0-100)** plus a breakdown: formatting, keywords, content quality, structure
- Strengths, weaknesses, **missing keywords** and prioritised improvements
- Optional **job description** field for a targeted keyword match
- Download the report as JSON

## Run locally
```bash
pip install -r requirements.txt
export GEMINI_API_KEY="your_key_here"     # Windows PowerShell: $env:GEMINI_API_KEY="your_key_here"
streamlit run app.py
```
Get a free API key at https://aistudio.google.com/apikey.
If no key is set, the app shows a password box in the sidebar.

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub (`app.py`, `requirements.txt`, `README.md`).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, choose the repo, branch `main`, main file `app.py`.
4. Open **Advanced settings → Secrets** and add:
   ```toml
   GEMINI_API_KEY = "your_key_here"
   ```
5. Click **Deploy**.

## Notes
- Scanned (image-only) PDFs can't be read. Use a text-based PDF or DOCX.
- The model name is set by `MODEL_NAME` in `app.py`. Change it if Google retires it.
- Scores are AI estimates, not the output of any real ATS. Use them as guidance.
- Never commit your API key to GitHub.
