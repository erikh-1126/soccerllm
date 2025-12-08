from llama_cpp import Llama

MODEL_PATH = "models/Meta-Llama-3-8B.Q4_K_M.gguf"

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=6
)


def generate_player_summary(player_doc):
    prompt = f"""
You are an expert soccer analyst. Write a concise, factual narrative paragraph about the soccer career of the player shown below.

STRICT RULES:
- Write 4–6 complete sentences.
- ONLY talk about the player's soccer career.
- NO bullet points.
- NO headings.
- NO code.
- NO essays, no instructions, no analysis guidelines.
- Stay neutral and factual.
- Use the stats only as reference — do not repeat them as a list.

Player Bio:
- Name: {player_doc['name']}
- Position: {player_doc['position']}
- Clubs Played For: {", ".join(player_doc['clubs'])}
- Appearances: {player_doc['appearances']}
- Goals: {player_doc['goals']}

Now write the summary as a polished paragraph:
"""

    return response["choices"][0]["message"]["content"].strip()
