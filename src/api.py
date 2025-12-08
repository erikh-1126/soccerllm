from flask import Flask, request, jsonify
from pymongo import MongoClient
from llm_client import generate_player_summary
import os

app = Flask(__name__)

mongo_uri = os.getenv("MONGO_URI", "mongodb://mongo:27017")
client = MongoClient(mongo_uri)
db = client["soccerdb"]

@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.json
    name = data.get("player_name")

    player = db.players.find_one({"name": name})

    if not player:
        return jsonify({"error": "Player not found in DB"}), 404

    summary = generate_player_summary(player)

    db.requests.insert_one({
        "player_name": name,
        "summary": summary
    })

    return jsonify({
        "player_name": name,
        "summary": summary
    })

@app.route("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    app.run(port=5000)
