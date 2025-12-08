from llama_cpp import Llama

MODEL_PATH = "models/Meta-Llama-3-8B.Q4_K_M.gguf"

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=6
)


def generate_player_summary(player_doc):
    prompt = f"""
You are a soccer analyst. Write 4-6 sentences summarizing this player's career.
Follow the rules:

- Only write about their soccer career.
- Do NOT include instructions, citations, or headings.
- Do NOT include bullet points or markdown.
- The response must be a single normal paragraph.

Player:
Name: {player_doc['name']}
Position: {player_doc['position']}
Clubs: {", ".join(player_doc['clubs'])}
Appearances: {player_doc['appearances']}
Goals: {player_doc['goals']}

Write the summary now:
"""

    response = llm.create_chat_completion(
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=200,
        temperature=0.7,
        stop=["<|end_of_text|>", "\n\n"]
    )

    return response["choices"][0]["message"]["content"].strip()
