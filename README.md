# LangGraph Chatbot
## Quick start

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd langgraph-chatbot

# 2. Copy the env template and fill with your real keys
cp .env.example .env
# then edit .env with your favorite editor

# 3. Build the Docker image
docker build -t langgraph-chatbot .

# 4. Run it
docker run --rm -it \
  --env-file .env \
  -v "$(pwd)/logs:/app/logs" \
  langgraph-chatbot
```

To exit, type `quit`, `exit`, `q`, or press `Ctrl+D`.
