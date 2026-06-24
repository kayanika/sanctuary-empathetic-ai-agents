from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from src.pipeline.state import TAOState
import os
import json
import re

INTAKE_SYSTEM_PROMPT = """You are a compassionate mental health support assistant conducting a brief wellness check-in.

Your role:
- Listen carefully and respond with empathy and warmth
- Ask ONE focused follow-up question to better understand the person's experience
- Naturally cover these wellbeing areas through conversation: current mood, how long they have felt this way, sleep quality, energy levels, and ability to concentrate
- Never diagnose, label conditions, or make clinical claims (do not say "depression", "anxiety disorder", etc.)
- Use plain, supportive language — no clinical jargon
- If the person mentions thoughts of self-harm or crisis, acknowledge their pain and immediately direct them to seek professional support (e.g. a crisis line)

- When multimodal signal features are provided alongside the user's message, use them as supplementary context to inform your understanding — but do not explicitly mention them to the user.

Keep your response to 2-3 sentences. End with exactly one follow-up question.
"""


EXTRACTION_SYSTEM_PROMPT = """You are a clinical information extraction tool. Given a person's statement about how they have been feeling, extract a structured list of reported symptoms.

Rules:
- Output ONLY a JSON array of strings. No prose, no explanation, no markdown fences.
- Each string is one symptom, in lowercase, using clinical-but-plain wording.
- If the person states or implies a duration, append it (e.g "low mood >2 weeks", "insomnia 3 days").
- Only extract symptoms the person actually reported. Do NOT infer symptoms they did not mention.
- If no symptoms are reported, output an epmty array: []

Example input: "I've been feeling really down for the past month and I can't sleep."
Example output: ["low mood >1 month", "insomnia"]


"""

def _format_features(features: dict) -> str:
    if not features:
        return ""
    lines = ["[Multimodal Signal Features - clinician context only, do not mention to user]"]
    if "speech_rate" in features:
        lines.append(f" Speech rate :{features['speech_rate']:.2f}words/min")
    if "pause_duration" in features:
        lines.append(f" Mean pause duration: {features['pause_duration']:.2f}s")
    if "mean_f0" in features:
        lines.append(f" Mean pitch (F0): {features['mean_f0']:.1f} Hz")
    if "f0_std" in features:
        lines.append(f" Pitch variability: {features['f0_std']:.1f} Hz")
    if "valence" in features:
        lines.append(f" Emotional valence: {features['valence']:.2f} (-1 = negative, +1 = positive)")
    if "au_intensities" in features:
        aus = features["au_intensities"]
        au_str = ", ".join(f"AU{k}={v:.2f}" for k, v in aus.items())
        lines.append(f"  Facial action units: {au_str}")
    return "\n".join(lines)
    
        
def _parse_symptoms(raw: str) -> list[str]:
    """Parse the LLM's JSON array output into a list of symptom strings.
    Falls back to an empty list if parsing fails (graceful degradation)."""
    text = raw.strip()
    # Strip markdown code fences if the model wrapped its output
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    # Extract the JSON array even if surrounded by prose
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        text = match.group(0)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(s).strip() for s in parsed if str(s).strip()]
    except (json.JSONDecodeError, ValueError):
        pass
    return []


def intake_agent(state: TAOState) -> dict:
    model = ChatOllama(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        model=os.getenv("OLLAMA_MODEL", "gemma4:12b-mlx"),
        temperature=0.7,
    )

    user_input = state.get("user_input", "")
    features = state.get("multimodal_features",{})
    history = state.get("dialogue_history", [])

    messages = [SystemMessage(content=INTAKE_SYSTEM_PROMPT)]
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
            
    feature_block = _format_features(features)
    if feature_block:
        full_input = f"User statement: {user_input}\n\n{feature_block}"
    else:
        full_input = user_input
    

    messages.append(HumanMessage(content=full_input))

    response = model.invoke(messages)
    assistant_response = response.content

    extractor = ChatOllama(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        model=os.getenv("OLLAMA_MODEL", "gemma4:12b-mlx"),
        temperature=0.0,
    )
    
    user_turns = [m["content"]for m in history if m["role"]=="user"]
    user_turns.append(user_input)
    full_user_text = "".join(user_turns)
    extraction_response = extractor.invoke(
        SystemMessage(content=EXTRACTION_SYSTEM_PROMPT),
        HumanMessage(Content=full_user_text),
    )
    
    extracted_symptoms = _parse_symptoms(extraction_response.content)
    
    trace_entry = {
        "agent": "intake",
        "input_summary": user_input[:300],
        "output_summary": assistant_response[:300],
        "multimodal_included": bool(features),
        "symptoms_extracted": len(extracted_symptoms)

    }

    return {
        "dialogue_history": [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": assistant_response},
        ],
        "extracted_symptoms": extracted_symptoms,
        "aar_trace": [trace_entry],
    }
