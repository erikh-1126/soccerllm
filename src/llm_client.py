def generate_player_summary(player_doc):
    prompt = f"""
You are an expert soccer analyst. Your task is to write a concise and factual career summary for the player provided.
Follow ALL rules below:

- ONLY write about the player's soccer career.
- DO NOT include any chapters, essays, academic instructions, citations, or unrelated information.
- Write 4-6 sentences in a single paragraph.
- Do NOT add headings, bullet points, or markdown.
- Stay factual, neutral, and concise.

Player Information:
Name: {player_doc['name']}
Position: {player_doc['position']}
Clubs: {", ".join(player_doc['clubs'])}
Appearances: {player_doc['appearances']}
Goals: {player_doc['goals']}

Write the summary now:
"""
    result = llm.create_completion(
        prompt=prompt,
        max_tokens=200,
        temperature=0.7,
        stop=["<|end_of_text|>"]
    )

    summary = result["choices"][0]["text"].strip()
    return summary
