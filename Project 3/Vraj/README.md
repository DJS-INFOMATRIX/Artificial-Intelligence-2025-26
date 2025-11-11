# 🏴‍☠️ One Piece RAG Chatbot

A FREE, fully local AI chatbot about **Marineford**, **Wano**, and **Enies Lobby** arcs with RAG, conversation memory, and multiple interfaces!

## ✨ Features

- 🆓 **100% FREE** - No API costs (local LLM + embeddings)
- 🤖 **Smart RAG** - Retrieves relevant context from One Piece knowledge base
- 💬 **Memory** - Remembers conversation history (last 5 exchanges)
- 🎭 **3 Modes** - CLI, REST API (FastAPI), Web UI (Streamlit)
- 🚀 **Fast** - Llama 3.2 1B model via Ollama
- 🎨 **Beautiful UI** - One Piece themed Streamlit + Next.js frontend

## 📋 Requirements

- Python 3.10+
- Docker Desktop
- 8GB+ RAM

## 🚀 Quick Start

### 1️⃣ Install Ollama & Download Model
```powershell
winget install Ollama.Ollama
```
Restart terminal, then:
```powershell
ollama pull llama3.2:1b
```

### 2️⃣ Start Weaviate Vector Database
```powershell
docker-compose up -d
```

### 3️⃣ Setup Python Environment
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 4️⃣ Initialize Database
```powershell
python setup.py
```

### 5️⃣ Run the Chatbot

Choose your preferred mode:

**CLI (Terminal Interface):**
```powershell
python app.py cli
```

**API Server (for React/Next.js frontend):**
```powershell
python app.py api
```
- API: http://localhost:8000
- Docs: http://localhost:8000/docs

**Web Interface (Streamlit):**
```powershell
python app.py web
```
- Web: http://localhost:8501

### 6️⃣ Run Next.js Frontend (Optional)
```powershell
cd New_Frontend
pnpm install
pnpm dev
```
- Frontend: http://localhost:3000

## 💬 Example Questions

- "Was Ace killed in Marineford?"
- "What gear did Luffy use against Kaido?"
- "Who saved Robin at Enies Lobby?"
- "What is Gear 5?"
- "Tell me about Whitebeard's final moments"

## 📂 Project Structure

```
ChatbotAI/
├── app.py                      # 🎯 UNIFIED APP (CLI/API/Web modes)
├── setup.py                    # Database initialization
├── requirements.txt            # Python dependencies
├── docker-compose.yml          # Weaviate configuration
├── onepiece_arcs_data.json    # Knowledge base
├── README.md                   # This file
└── New_Frontend/              # Next.js React frontend
    ├── app/page.tsx
    └── components/
```

## 🛠️ Tech Stack

- **LLM**: Llama 3.2 1B (Ollama)
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 (local)
- **Vector DB**: Weaviate
- **Backend**: FastAPI + LangChain
- **Frontends**: Streamlit + Next.js
- **Framework**: LangChain

## 🎯 How It Works

1. **Document Chunking**: One Piece arcs split into manageable chunks
2. **Embeddings**: Text converted to vectors locally (no API needed)
3. **Vector Search**: Weaviate finds relevant chunks using semantic similarity
4. **Context Building**: Top 5 relevant documents retrieved
5. **LLM Generation**: Llama 3.2 generates detailed answers using context
6. **Memory**: Conversation history tracked for follow-up questions

## 🎨 Enhanced Response Quality

The chatbot provides **immersive, story-like responses** instead of robotic answers:

### Before:
> "Ace died in Marineford."

### After:
> "The shocking fate of Ace - a pivotal moment in One Piece history that continues to captivate audiences worldwide. During the climactic Battle of Marineford, Ace made the ultimate sacrifice to protect his beloved brother Luffy. When Admiral Akainu launched a devastating magma attack aimed at Luffy, Ace threw himself in front of the blow...
> 
> With his final breaths, Ace shared one last message with Luffy: 'I'm proud of you, bro! You're the strongest!' As their bond was sealed by this emotional farewell, it became clear that Ace's sacrifice would not be in vain..."

**Features:**
- ✅ Comprehensive coverage using all context
- ✅ Natural storytelling flow
- ✅ Vivid details (names, abilities, emotions)
- ✅ Logical event connections
- ✅ Epic adventure tone
- ✅ Structured with paragraphs
- ✅ Emotional resonance

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `GET /` | GET | Health check |
| `POST /chat` | POST | Send message, get AI response |
| `POST /clear` | POST | Clear chat history |
| `GET /health` | GET | Server health status |

### Example API Request (PowerShell):
```powershell
$body = @{
    message = "Who saved Robin at Enies Lobby?"
    chat_history = @()
    selected_arc = "Enies Lobby"
} | ConvertTo-Json

Invoke-WebRequest -Uri "http://localhost:8000/chat" -Method POST -Body $body -ContentType "application/json"
```

## 🔧 Configuration

### Adjust LLM Parameters (in `app.py`):
```python
"temperature": 0.75    # Creativity (0.0-1.0)
"num_predict": 512     # Max response length
"top_p": 0.9          # Nucleus sampling
"top_k": 40           # Top-K sampling
```

### Change Retrieval Count:
```python
search_kwargs={"k": 5}  # Number of documents to retrieve
```

### Modify Memory Window:
```python
k=5  # Number of conversation exchanges to remember
```

## 🐳 Docker Commands

```powershell
# Start Weaviate
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Stop Weaviate
docker-compose down

# Reset database
docker-compose down -v
python setup.py
```

## 📝 Notes

- **First run**: Downloads ~90MB embedding model + ~1.3GB Llama model
- **Offline**: Works completely offline after initial setup
- **Free**: No API keys or subscriptions needed
- **Fast**: Responses in 2-5 seconds on modern hardware
- **Scalable**: Easy to add more arcs in `onepiece_arcs_data.json`

## 🚨 Troubleshooting

### Ollama not found
```powershell
ollama --version
# If not found, restart terminal or reinstall Ollama
```

### Weaviate connection error
```powershell
docker-compose ps
# If not running: docker-compose up -d
```

### Import errors
```powershell
pip install -r requirements.txt --upgrade
```

### Port already in use
- API (8000): Change in `app.py` → `uvicorn.run(port=8001)`
- Web (8501): Streamlit auto-assigns new port
- Next.js (3000): Change in `New_Frontend/package.json`

## 🎓 Usage Tips

1. **Be specific**: "What happened to Ace in Marineford?" vs "Tell me about Ace"
2. **Follow-up questions**: "What happened next?" (memory enabled)
3. **Arc filtering**: Use Web/API mode to filter by specific arc
4. **Long conversations**: Clear chat when switching topics

## 🤝 Contributing

Want to improve the chatbot?
1. Add more arcs to `onepiece_arcs_data.json`
2. Enhance prompts in `app.py`
3. Adjust RAG parameters for better retrieval
4. Improve frontend UI/UX

## 📄 License

MIT License - Feel free to use and modify!

---

Made with ❤️ for One Piece fans | Powered by local AI 🏴‍☠️

**Enjoy chatting about One Piece!** 🚢⚓
