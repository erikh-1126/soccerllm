from flask import Flask, request, jsonify
from pymongo import MongoClient
from llama_cpp import Llama
import datetime
import re

# ----- Flask Setup -----
app = Flask(__name__)

# ----- MongoDB Setup -----
client = MongoClient("mongodb://localhost:27017/")
db = client["soccer_llm"]
collection = db["player_summaries"]

# ----- Llama (llama-cpp) Setup -----
model = Llama(model_path="models/llama-3-8b.gguf", n_ctx=2048)

# ----- Helper: Extract Player Name -----
def extract_name(text):
    match = re.search(r"[A-Z][a-z]+ [A-Z][a-z]+", text)
    return match.group(0) if match else None

@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json()
    raw_text = data.get("text", "")

    player = extract_name(raw_text)
    if not player:
        return jsonify({"error": "No recognizable player name found"}), 400

    # LLM prompt
    prompt = f"Describe the career of the soccer player: {player}. Keep it factual."

    output = model(prompt, max_tokens=500)
    summary = output["choices"][0]["text"]
    tokens_used = output.get("usage", {}).get("total_tokens", None)

    # Save to MongoDB
    document = {
        "player_name": player,
        "input_text": raw_text,
        "summary": summary,
        "model_name": "llama-3-8b.gguf",
        "timestamp": datetime.datetime.utcnow(),
        "tokens_used": tokens_used
    }

    collection.insert_one(document)

    return jsonify({
        "player": player,
        "summary": summary
    })

@app.route("/history/<name>", methods=["GET"])
def history(name):
    docs = list(collection.find({"player_name": {"$regex": f"^{name}$", "$options": "i"}}))
    
    for d in docs:
        d["_id"] = str(d["_id"])
    
    return jsonify(docs)

if __name__ == "__main__":
    app.run(debug=True)
