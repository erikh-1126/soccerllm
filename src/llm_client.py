from llama_cpp import Llama

MODEL_PATH = "models/Meta-Llama-3-8B.Q4_K_M.gguf"

# Load the model once
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=6
)

def generate_player_summary(player_doc):
    prompt = f"""
Write a SINGLE paragraph summarizing the following soccer player's professional career.

• 4–6 complete sentences only.
• Focus only on soccer career — no personal life, no opinions.
• No lists, no bullet points, no headings, no copying the instructions.
• Do not mention these rules.

Player:
Name: {player_doc['name']}
Position: {player_doc['position']}
Clubs: {", ".join(player_doc['clubs'])}
Appearances: {player_doc['appearances']}
Goals: {player_doc['goals']}

Paragraph:
"""

    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=220,
        temperature=0.4,
        top_p=0.95,
        frequency_penalty=0.4,
        stop=["<|end_of_text|>"]
    )

    text = response["choices"][0]["message"]["content"]

    # Cleanup
    text = text.split("Paragraph:")[-1].strip()
    text = text.replace("\n", " ").replace("#", "").strip()

    return text

