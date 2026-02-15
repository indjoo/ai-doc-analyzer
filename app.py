"""
AI Document Analyzer — Day 3 Build
Streamlit web app that analyzes PDFs using Claude or Gemini.
Portfolio piece + Upwork proof of work.
"""

import streamlit as st
import base64
import json
import time

# ── Page config (must be first Streamlit command) ──
st.set_page_config(
    page_title="AI Document Analyzer",
    page_icon="📄",
    layout="centered"
)

# ── Pricing constants ──
PRICING = {
    "Claude Sonnet 4.5": {"input": 3.00, "output": 15.00},   # per 1M tokens
    "Gemini 2.5 Flash":  {"input": 0.15, "output": 0.60},    # per 1M tokens (paid tier pricing, free tier = $0)
}

# ── Helper: calculate cost ──
def calculate_cost(model, input_tokens, output_tokens):
    p = PRICING[model]
    input_cost  = (input_tokens  / 1_000_000) * p["input"]
    output_cost = (output_tokens / 1_000_000) * p["output"]
    return input_cost + output_cost

# ── Helper: parse JSON from model response ──
def parse_json_response(text):
    """Extract JSON from model response, handling markdown fences."""
    cleaned = text.strip()
    # Strip markdown code fences if present
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first line (```json or ```) and last line (```)
        lines = [l for l in lines[1:] if l.strip() != "```"]
        cleaned = "\n".join(lines)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Last resort: find first { to last }
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(cleaned[start:end])
        raise

# ── Helper: build the analysis prompt ──
SYSTEM_PROMPT = """You are a document analysis assistant. Analyze the provided PDF and return ONLY valid JSON with no other text. Use this exact structure:

{
    "title": "Document title or best guess",
    "author": "Author if found, otherwise 'Unknown'",
    "page_count_estimate": number,
    "executive_summary": "3 sentence summary of the entire document",
    "key_points": [
        "First key point",
        "Second key point",
        "Third key point",
        "Fourth key point",
        "Fifth key point"
    ],
    "action_items": [
        "First action item or recommendation",
        "Second action item or recommendation"
    ]
}

Return ONLY the JSON object. No markdown, no explanation, no preamble."""

# ── Analyze with Claude ──
def analyze_with_claude(pdf_bytes):
    import anthropic
    
    api_key = st.secrets.get("ANTHROPIC_API_KEY")
    if not api_key:
        st.error("Missing ANTHROPIC_API_KEY in secrets. Add it in app settings → Secrets.")
        return None
    
    client = anthropic.Anthropic(api_key=api_key)
    pdf_b64 = base64.standard_b64encode(pdf_bytes).decode("utf-8")
    
    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_b64
                        }
                    },
                    {
                        "type": "text",
                        "text": "Analyze this PDF document and return the structured JSON summary."
                    }
                ]
            }
        ]
    )
    
    raw_text = response.content[0].text
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    
    return {
        "raw": raw_text,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }

# ── Analyze with Gemini ──
def analyze_with_gemini(pdf_bytes):
    import google.generativeai as genai
    
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        st.error("Missing GEMINI_API_KEY in secrets. Add it in app settings → Secrets.")
        return None
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    
    response = model.generate_content(
        [
            {
                "mime_type": "application/pdf",
                "data": pdf_bytes   # Gemini takes raw bytes, no base64 needed
            },
            f"{SYSTEM_PROMPT}\n\nAnalyze this PDF document and return the structured JSON summary."
        ]
    )
    
    raw_text = response.text
    input_tokens = response.usage_metadata.prompt_token_count
    output_tokens = response.usage_metadata.candidates_token_count
    
    return {
        "raw": raw_text,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }

# ── Main UI ──
st.title("📄 AI Document Analyzer")
st.markdown("Upload a PDF and get a structured analysis powered by AI.")

# Sidebar settings
st.sidebar.title("Settings")
model_choice = st.sidebar.selectbox(
    "Choose AI Model",
    ["Claude Sonnet 4.5", "Gemini 2.5 Flash"]
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Built by Luka** · "
    "[GitHub](https://github.com/indjoo) · "
    "[Upwork](https://www.upwork.com/freelancers/YOUR_PROFILE)"
)

# File upload
uploaded_file = st.file_uploader(
    "Upload a PDF (max 20MB)",
    type=["pdf"],
    help="Supported format: PDF files up to 20MB"
)

# Show file info if uploaded
if uploaded_file is not None:
    file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
    st.info(f"**{uploaded_file.name}** — {file_size_mb:.1f} MB")
    
    if file_size_mb > 20:
        st.error("File exceeds 20MB limit. Please upload a smaller PDF.")
    else:
        # Analyze button — gates the API call
        if st.button(f"Analyze with {model_choice}", type="primary"):
            pdf_bytes = uploaded_file.getvalue()
            
            with st.spinner(f"Analyzing with {model_choice}... this may take 15-60 seconds."):
                start_time = time.time()
                
                try:
                    if model_choice == "Claude Sonnet 4.5":
                        result = analyze_with_claude(pdf_bytes)
                    else:
                        result = analyze_with_gemini(pdf_bytes)
                    
                    elapsed = time.time() - start_time
                    
                    if result is None:
                        st.stop()
                    
                    # Parse the JSON response
                    parsed = parse_json_response(result["raw"])
                    
                    # ── Display results ──
                    st.success(f"Analysis complete in {elapsed:.1f}s")
                    
                    # Document info
                    st.header(parsed.get("title", "Untitled Document"))
                    col1, col2 = st.columns(2)
                    col1.metric("Author", parsed.get("author", "Unknown"))
                    col2.metric("Est. Pages", parsed.get("page_count_estimate", "N/A"))
                    
                    # Executive summary
                    st.subheader("Executive Summary")
                    st.write(parsed.get("executive_summary", "No summary available."))
                    
                    # Key points
                    st.subheader("Key Points")
                    for point in parsed.get("key_points", []):
                        st.markdown(f"- {point}")
                    
                    # Action items
                    st.subheader("Action Items")
                    for item in parsed.get("action_items", []):
                        st.markdown(f"- {item}")
                    
                    # ── Cost & usage stats ──
                    st.markdown("---")
                    st.subheader("Usage & Cost")
                    
                    input_tok  = result["input_tokens"]
                    output_tok = result["output_tokens"]
                    cost = calculate_cost(model_choice, input_tok, output_tok)
                    
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Input Tokens", f"{input_tok:,}")
                    c2.metric("Output Tokens", f"{output_tok:,}")
                    c3.metric("Total Cost", f"${cost:.4f}")
                    c4.metric("Time", f"{elapsed:.1f}s")
                    
                    # Download raw JSON
                    st.download_button(
                        label="Download Full JSON",
                        data=json.dumps(parsed, indent=2, ensure_ascii=False),
                        file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}_analysis.json",
                        mime="application/json"
                    )
                    
                except json.JSONDecodeError:
                    st.error("The AI returned an invalid response. Raw output below:")
                    st.code(result["raw"] if result else "No response")
                except Exception as e:
                    st.error(f"Error: {type(e).__name__}: {e}")

else:
    # Empty state
    st.markdown("---")
    st.markdown(
        "**How it works:**\n"
        "1. Upload any PDF document\n"
        "2. Choose an AI model (Claude or Gemini)\n"
        "3. Click Analyze to get a structured summary\n"
        "4. Download the results as JSON"
    )
