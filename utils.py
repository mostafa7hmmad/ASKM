# utils.py
import json

def load_examples(path="few_shot_examples.json"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []

def get_final_prompt() -> str:
    try:
        with open("system_prompt.txt", "r", encoding="utf-8") as f:
            system_prompt = f.read().strip()
    except:
        system_prompt = "You are an Islamic Dawah Assistant. Answer strictly using the retrieved context and the following approved sources only: Duroos Al-Sunniyah, Islam Question & Answer, and IslamWeb. All answers must include clear citations."

    examples = load_examples()
    examples_text = ""
    for ex in examples[:5]:
        examples_text += f"Example Q: {ex.get('question','')}\nExample A: {ex.get('answer','')}\n\n"

    return (
    f"{system_prompt}\n\n"
    f"{examples_text}"
    "CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\n"
    "Answer strictly using ONLY the provided CONTEXT and the following approved sources: "
    "Duroos Al-Sunniyah, Islam Question & Answer, and IslamWeb. "
    "Do NOT use any external knowledge. "
    "If the answer is not found, reply exactly: "
    "\"I'm sorry, my current database does not contain this specific information.\" "
    "Write in clear paragraphs (not lists unless necessary). "
    "Include concise citations in the format: [source: Name (ID)]."
)
