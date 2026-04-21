import os
from groq import Groq
from rag import retrieve_for_disease

MODEL = "llama-3.3-70b-versatile"

LANGUAGE_MAP = {
    "english": "English",
    "hindi":   "Hindi",
    "gujarati":"Gujarati"
}

_groq_key = os.environ.get("GROQ_API_KEY")
if not _groq_key:
    raise EnvironmentError(
        "GROQ_API_KEY environment variable is not set.\n"
        "Local:  set GROQ_API_KEY=gsk_...\n"
        "Render: add it in Environment Variables dashboard."
    )
client = Groq(api_key=_groq_key)


# ───────────────────────────────
# Build context from RAG
# ───────────────────────────────

def build_context(disease_name):

    docs = retrieve_for_disease(disease_name)

    context_parts = []

    for doc in docs:
        context_parts.append(doc["text"])

    return "\n\n---\n\n".join(context_parts)


# ───────────────────────────────
# Prompt builder
# ───────────────────────────────

def build_prompt(disease_name, confidence, language, context):

    lang = LANGUAGE_MAP.get(language.lower(), "English")

    return f"""
You are KrushiBandhu Crop Doctor helping cotton farmers.

The AI detected:

Disease: {disease_name}
Confidence: {confidence*100:.1f}%

Use the agricultural knowledge below.

========================
AGRICULTURAL KNOWLEDGE
========================
{context}

========================
TASK
========================

Explain the situation to the farmer.

Write in simple {lang}.

Use this structure:

WHAT IS HAPPENING
Explain the disease or pest.

WHY THIS HAPPENED
Explain likely causes.

IMMEDIATE ACTION
List steps the farmer should take immediately.

ORGANIC REMEDY
Provide natural treatment.

PREVENTION
Explain how to prevent this next season.

WHEN TO SEE A SPECIALIST
Explain when expert help is needed.
"""


# ───────────────────────────────
# Main advisory function
# ───────────────────────────────

def ask_crop_doctor(disease_name, confidence, language="english"):

    context = build_context(disease_name)

    prompt = build_prompt(disease_name, confidence, language, context)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024
    )

    return response.choices[0].message.content


# ───────────────────────────────
# Follow-up Q&A
# ───────────────────────────────

def ask_followup(question, disease_name, language="english"):

    lang = LANGUAGE_MAP.get(language.lower(), "English")

    context = build_context(disease_name)

    prompt = f"""
You are KrushiBandhu Crop Doctor helping cotton farmers.

Disease detected: {disease_name}

Knowledge:
{context}

Farmer question:
{question}

Answer in simple {lang}. Give practical advice farmers can follow.
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512
    )

    return response.choices[0].message.content