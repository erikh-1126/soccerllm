from llama_cpp import Llama

MODEL_PATH = "models/Meta-Llama-3-8B.Q4_K_M.gguf"

# Load the model once at startup
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=6
)


def generate_player_summary(player_doc):
    prompt = f"""
You are an expert soccer analyst. Write a concise paragraph summarizing this player's soccer career.

STRICT OUTPUT RULES:
- Write 4–6 complete factual sentences in ONE paragraph.
- ONLY talk about the player's soccer career.
- NO bullet points.
- NO headings.
- NO lists.
- NO code, no essays, no writing instructions.
- Do NOT repeat these rules or mention them.

Player Bio:
Name: {player_doc['name']}
Position: {player_doc['position']}
Clubs: {", ".join(player_doc['clubs'])}
Appearances: {player_doc['appearances']}
Goals: {player_doc['goals']}

Begin the summary now:
"""

    result = llm.create_chat_completion(
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=200,
        temperature=0.6,
    )

    summary = result["choices"][0]["message"]["content"].strip()

    # Clean weird artifacts
    summary = summary.replace("#", "").strip()
    summary = summary.replace("\n", " ").strip()

    return summary
