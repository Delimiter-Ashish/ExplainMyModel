from llama_cpp import Llama

def load_llm(model_path: str):
    return Llama(
        model_path=model_path,
        n_ctx=2048,
        n_threads=4,
        verbose=False
    )

def generate_report(llm, summary: dict):
    prompt = f"""
You are a senior ML engineer writing a professional audit report.

System summary:
{summary}

Write a structured markdown audit report with:
- Executive Summary
- Business Risk
- Model Performance
- Drift Risks
- Interpretability
- Recommendations
"""
    res = llm(prompt, max_tokens=1000)
    return res["choices"][0]["text"]
