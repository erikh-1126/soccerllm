from llama_cpp import Llama

MODEL_PATH = "models/Meta-Llama-3-8B.Q4_K_M.gguf"

llm = Llama(model_path=MODEL_PATH, n_ctx=2048)

def generate_player_summary(player_doc):
    prompt = f"""
You are an expert soccer analyst. Write a concise, factual, 4–6 sentence summary
of this player's soccer career. Follow ALL rules:

- ONLY mention the player's soccer career.
- DO NOT include essays, instructions, citations, or formatting.
- One paragraph only. No bullet points, no headings.

Player Info:
Name: {player_doc['name']}
Position: {player_doc['position']}
Clubs: {", ".join(player_doc['clubs'])}
Appearances: {player_doc['appearances']}
Goals: {player_doc['goals']}

Now write the summary:
"""

    result = llm.create_chat_completion(
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=200,
        temperature=0.6,
    )

    text = result["choices"][0]["message"]["content"]
    return text.strip()
