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
client = Groq(api_key=_groq_key, timeout=25.0) if _groq_key else None


# ───────────────────────────────
# Build context from RAG
# ───────────────────────────────

def build_context(disease_name):

    docs = retrieve_for_disease(disease_name)

    context_parts = []

    for doc in docs:
        context_parts.append(doc["text"])

    return "\n\n---\n\n".join(context_parts)


def _fallback_advisory(disease_name, confidence, language="english"):
    lang = LANGUAGE_MAP.get(language.lower(), "English")
    confidence_pct = f"{confidence*100:.1f}%"
    return (
        f"WHAT IS HAPPENING\n"
        f"The crop likely shows symptoms of {disease_name}. The model confidence is {confidence_pct}.\n\n"
        f"WHY THIS HAPPENED\n"
        f"Common reasons include humid weather, dense canopy, poor airflow, and delayed field scouting.\n\n"
        f"IMMEDIATE ACTION\n"
        f"1) Remove heavily affected leaves.\n"
        f"2) Keep irrigation balanced and avoid water stress.\n"
        f"3) Follow label-approved control measures recommended for {disease_name}.\n\n"
        f"ORGANIC REMEDY\n"
        f"Use neem-based spray or bio-control options suitable for local extension guidance.\n\n"
        f"PREVENTION\n"
        f"Use clean planting material, improve spacing, monitor weekly, and rotate control methods.\n\n"
        f"WHEN TO SEE A SPECIALIST\n"
        f"If spread continues after 3-5 days or more than 20% of plants are affected, contact an agronomist immediately.\n\n"
        f"(Fallback advisory generated because live AI response was unavailable. Language requested: {lang}.)"
    )


def _fallback_followup(question, disease_name, language="english"):
    lang = LANGUAGE_MAP.get(language.lower(), "English")
    return (
        f"I could not reach the live AI advisor right now. Based on the detected condition ({disease_name}), "
        f"please continue field scouting, remove badly affected leaves, and use locally approved treatment protocols. "
        f"If symptoms are spreading quickly, consult your nearest agriculture extension officer. "
        f"(Language requested: {lang})"
    )


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

    if client is None:
        return _fallback_advisory(disease_name, confidence, language)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
        )
        content = response.choices[0].message.content
        return content or _fallback_advisory(disease_name, confidence, language)
    except Exception:
        return _fallback_advisory(disease_name, confidence, language)


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

    if client is None:
        return _fallback_followup(question, disease_name, language)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
        )
        content = response.choices[0].message.content
        return content or _fallback_followup(question, disease_name, language)
    except Exception:
        return _fallback_followup(question, disease_name, language)