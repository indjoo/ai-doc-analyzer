# AI Document Analyzer

A Streamlit web app that analyzes PDF documents using AI models (Claude Sonnet 4.5 or Gemini 2.5 Flash) and returns structured summaries.

## Features

- Upload any PDF up to 20MB
- Choose between Claude Sonnet 4.5 and Gemini 2.5 Flash
- Get structured analysis: executive summary, key points, action items
- See token usage and cost breakdown
- Download results as JSON

## Live Demo

[ai-doc-analyzer.streamlit.app](https://ai-doc-analyzer-u3e5w6gjjfdfdpvgkd4rgb.streamlit.app)

## Run Locally

```bash
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml`:
```toml
ANTHROPIC_API_KEY = "your-key"
GOOGLE_API_KEY = "your-key"
```

Run:
```bash
streamlit run app.py
```

## Tech Stack

Python, Streamlit, Claude API, Gemini API

## Author

Built by Luka — [Upwork](https://www.upwork.com)
