from transformers import pipeline
import json

def generate_reasoned_audit(payload: dict) -> str:
    llm = pipeline(
        "text2text-generation",
        model="google/flan-t5-base"
    )

    prompt = f"""
    You are a senior ML auditor.

    Evidence from the system:
    {json.dumps(payload, indent=2)}

    Write a professional audit report that includes:
    1) Executive Summary
    2) Model Risk Level (Low/Medium/High) with justification
    3) Data Drift Interpretation
    4) Reliability Assessment
    5) Fairness Concerns (if any)
    6) Specific Actionable Recommendations
    7) Whether retraining is required (Yes/No + why)

    Keep it professional and analytical.
    """

    result = llm(prompt, max_length=900)[0]["generated_text"]
    return result

