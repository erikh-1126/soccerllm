# Soccer Career Generator LLM
### By Erik Huisman

## 1. Executive Summary
Many soccer fans, myself included, want to easily look at career retrospectives of some of the world's best players, but it is hard to find good summaries consistently outside of Wikipedia. So I have created a model that will generate a paragraph-long overview of a player's career, given that this player is manually inputted into the database. All you need to do is input the player's name into the terminal, and it will spit out a retrospective of its own. All of this can be achieved without using external databases like Wikipedia, so it also erases the risk of repetitive storytelling.

## 2. System Overview
Course Concepts: This project demonstrates core course concepts, including  REST API design (Flask), LLM inference serving (llama-cpp-python), and database integration (MongoDB).

Architecture Diagram: ![Architecture](./assets/architecture.png)

Data/Models/Services: 

  Data: Stored in MongoDB
      Collections: players — static soccer player metadata 
                   requests — logs each summarization request
                   
  Models: Llama 3 8B Instruct (GGUF quantized Q4_K_M)
    Size: ~4 GB
    Source: Meta / QuantFactory (GGUF conversion on Hugging Face)
    Format: GGUF, run with llama-cpp-python
    
  Services: 
    Flask API:
      Routes: /summarize, /health
    MongoDB (Dockerized):
      Port: 27017
    LLM Inference Service:
      Embedded llama.cpp backend launched inside API container

## 3. How to Run (Locally)

### To Build
docker build -t soccer-llm .

### To Run
docker run -p 5000:5000 soccer-llm

### Health Check
curl http://localhost:5000/health

## 4. Design Decisions
Why this Concept: There were many different alternatives I could've taken while making this LLM. For example, I could have used the OpenAI API; however, it isn't reproducible offline. GPU-based inference models also came up in my head, but they aren't necessarily portable, making them harder to grade. SQLite could be used as an alternative to MongoDB, but it is less flexible for structured documents. Finally, FastAPI was also considered instead of Flask, but Flask is just simpler to pull off.

Tradeoffs: Using a CPU for inference instead of a GPU makes it portable, but in turn makes it slower to perform. Using a GGUF model to perform the summaries reduces the size of the file, but its accuracy isn't the best. Picking MongoDB over SQLite makes things more flexible, but it is much heavier than SQLite. Finally, using Docker makes the process more reproducible, but it takes longer to pull off. 

Security/Privacy: No external calls take place here; all inference performed in the summary is local, no personal data is stored at all, and every input is validated by the fact that any player name inputted must match a name inputted into the database.

Ops: All logs are stored in MongoDB, Docker Compose manages a multi-service lifecycle, and some known limitations include a cold start time for the Llama 3 model, a CPU latency of ~300–500ms per request, summaries being cut off early, and a possible chance of hallucinations in some of the responses.

## 5. Results & Evaluation
Response Screenshots: ![Response Screenschot](./assets/ronaldo_response_prompt.png)
![Response Screenshot #2](./assets/messi_response_prompt.png)

Time to Build: ![Build Time](./assets/build_time.png)

Example Players in MongoDB ![MongoDB Storage](./assets/mongodb_storage.png)

## 6. What's Next
In the future, I want to add data retrieval through Wikipedia, so players don't have to be inputted manually into MongoDB. On top of this, retrieval-augmented generation (RAG) is also a good next step, which can also be done via Wikipedia. I also want to add an add_player endpoint, so that including players in the database becomes easier. Finally, to make the process more visually appealing, I want to be able to deploy this process to the cloud and make it a visible chatbox (potentially through Azure).

## 7. Links
GitHub Repo <https://github.com/erikh-1126/soccerllm>
