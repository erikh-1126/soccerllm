from flask import Flask, request, jsonify
from db import get_player, save_summary
from llm_client import generate_summary

app = Flask(__name__)

@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.json
    player_name = data.get("player_name")

    player_doc = get_player(player_name)
    if not player_doc:
        return jsonify({"error": "Player not found in database"}), 404

    summary = generate_summary(player_doc)
    save_summary(player_name, summary)

    return jsonify({"player_name": player_name, "summary": summary})

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
