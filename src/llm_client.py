from llama_cpp import Llama

MODEL_PATH = "models/Meta-Llama-3-8B.Q4_K_M.gguf"

# Load model once
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=6
)


def generate_player_summary(player_doc):

    prompt = f"""
You are an expert soccer analyst. Write a concise, factual narrative paragraph
about the career of the player shown below.

RULES:
- 4–6 sentences only
- ONLY soccer career info
- No bullet points, no headings
- No code, no instructions, no citations
- Neutral, polished tone

Player Bio:
Name: {player_doc['name']}
Position: {player_doc['position']}
Clubs: {", ".join(player_doc['clubs'])}
Appearances: {player_doc['appearances']}
Goals: {player_doc['goals']}

Write the summary as a single paragraph:
""".strip()

    response = llm.create_chat_completion(
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=200,
        temperature=0.4,
        stop=["<|end_of_text|>"]
    )

    summary = response["choices"][0]["message"]["content"].strip()
    return summary
