# model.py

import re
from transformers import pipeline
from typing import List
from langchain.schema import Document
from huggingface_hub import login

# Replace with your actual token



# Load free Hugging Face LLM pipeline (can be replaced with other small models if slow)
llm = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",  # or try Cohere or mistral if access
    device="cpu"  # or "cuda" if you have GPU
)

def extract_query_info(query: str):
    age = re.search(r"(\d+)[ -]?year[- ]?old", query.lower())
    treatment = re.search(r"(surgery|accident|treatment|hospitalization|diagnosis)", query.lower())
    duration = re.search(r"policy.*?(since|for)?\s?(\d+)\s?(months|years)", query.lower())
    location = re.search(r"in\s([A-Za-z]+)", query)

    return {
        "age": age.group(1) if age else None,
        "treatment": treatment.group(1) if treatment else None,
        "duration": duration.group(2) + " " + duration.group(3) if duration else None,
        "location": location.group(1) if location else None,
    }


def get_llm_decision(query: str, docs: List[Document], extracted: dict):
    context = "\n".join([doc.page_content.strip()[:300] for doc in docs[:2]])

    prompt = f"""
User Query:
{query}

Relevant Policy Clauses:
{context}

Answer in this format:
Decision: approved or rejected
Amount: Rs. <amount>
Justification: <why this decision was made>
""".strip()

    raw_output = llm(prompt)[0]["generated_text"]

    try:
        decision_match = re.search(r'Decision:\s*(approved|rejected)', raw_output, re.IGNORECASE)
        amount_match = re.search(r'Amount:\s*Rs\.?\s*([\d,]+)', raw_output, re.IGNORECASE)
        justification_match = re.search(r'Justification:\s*(.+)', raw_output, re.IGNORECASE)

        if not (decision_match and amount_match and justification_match):
            raise ValueError("LLM response does not contain expected fields")

        return {
            "decision": decision_match.group(1).lower(),
            "amount": f"Rs. {amount_match.group(1)}",
            "justification": justification_match.group(1).strip()
        }

    except Exception as e:
        return {
            "raw_result": raw_output,
            "error": str(e)
        }


