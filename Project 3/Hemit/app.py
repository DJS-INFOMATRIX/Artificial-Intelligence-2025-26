"""
Games RAG Chatbot - Unified Application
Supports CLI, API Server (FastAPI), and Web Interface (Streamlit) modes
Run with: python app.py [cli|api|web]
This is adapted from your One Piece project to use a Games domain dataset.
"""

import os
import sys
import time
from typing import List, Optional
from dotenv import load_dotenv
import weaviate
from sentence_transformers import SentenceTransformer
import requests

# NOTE: these imports assume the same custom wrappers you used originally.
# If these wrappers are not available in your environment, swap with equivalent LangChain classes.
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_weaviate import WeaviateVectorStore
    from langchain_ollama import ChatOllama
    from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
    from langchain_classic.chains import ConversationalRetrievalChain
    from langchain_classic.memory import ConversationBufferWindowMemory
except Exception:
    # If the custom/langchain-community wrappers are not installed, you'll still be able to run
    # ingestion and Weaviate; LLM calls will fallback to HTTP Ollama endpoint in query().
    HuggingFaceEmbeddings = None
    WeaviateVectorStore = None
    ChatOllama = None
    PromptTemplate = None
    ChatPromptTemplate = None
    ConversationalRetrievalChain = None
    ConversationBufferWindowMemory = None

load_dotenv()


class GamesChatbot:
    """Unified RAG-powered Games chatbot (mirrors OnePieceChatbot structure)."""
    
    def __init__(self, mode="cli"):
        self.mode = mode
        self.setup_components()
    
    def setup_components(self):
        """Initialize Weaviate, embeddings, LLM and memory (or set placeholders)."""
        print("🔌 Connecting to Weaviate (localhost:8080)...")
        # Use the same helper style you used before. If your weaviate helper differs, change this.
        # Here we create a standard weaviate client connection:
        self.weaviate_client = weaviate.Client(url=os.getenv("WEAVIATE_URL", "http://localhost:8080"))

        print("🧠 Loading embedding model (sentence-transformers/all-MiniLM-L6-v2)...")
        # local SentenceTransformer used for any re-ranking or local embedding needs
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        # If you have HuggingFaceEmbeddings available, wrap it for LangChain usage
        if HuggingFaceEmbeddings is not None:
            self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        else:
            self.embeddings = None
        
        # Vectorstore wrapper (if available)
        if WeaviateVectorStore is not None:
            self.vectorstore = WeaviateVectorStore(
                client=self.weaviate_client,
                index_name="GameKnowledge",
                text_key="content",
                embedding=self.embeddings
            )
        else:
            self.vectorstore = None

        print("🤖 Preparing LLM connection...")
        # We'll primarily use a local Ollama HTTP endpoint if available like your original app did.
        # Keep a ChatOllama instance if you have the wrapper
        if ChatOllama is not None:
            self.llm = ChatOllama(model="llama3.2:1b", temperature=0.5)
        else:
            self.llm = None

        if self.mode == "cli":
            # In CLI mode use an in-memory windowed buffer similar to your original
            if ConversationBufferWindowMemory is not None:
                self.memory = ConversationBufferWindowMemory(
                    memory_key="chat_history",
                    return_messages=True,
                    k=5,
                    output_key="answer"
                )
            else:
                self.memory = None
            self.setup_rag_chain()
        else:
            # API & Web mode will hold chat_history in a simple list so we can pass it to the HTTP Ollama endpoint
            self.chat_history = []
        
        print("✅ Games Chatbot ready!")

    def setup_rag_chain(self):
        """Recreate a ConversationalRetrievalChain equivalent for CLI mode (if wrappers exist)."""
        # Prompt template tuned for games domain (mirrors the One Piece quality target structure)
        prompt_template = """You are a knowledgeable games analyst providing exceptional answers.

🎯 QUALITY TARGET: 9-10/10
Your answer must score:
• Factual Accuracy: 4/4 (every detail correct)
• Context Relevance: 3/3 (focused on the question)
• Clarity & Style: 2/2 (natural, engaging tone)
• Completeness: 1/1 (fully answers with details)

📋 ANSWER STRUCTURE:
1. Direct answer first (1 sentence)
2. Supporting details with specifics (2-4 sentences)
3. Brief insight or summary (1 sentence if relevant)

RULES:
• Use ONLY the context information below
• Be specific: names, exact mechanics, release years, platforms where relevant
• Write naturally - like an expert analyst, not a robot
• Focus on the game/genre most relevant to the question
• If you're not certain, say so clearly

CONTEXT:
{context}

CONVERSATION HISTORY:
{chat_history}

QUESTION: {question}

ANSWER (following the structure above for 9-10/10 quality):"""
        
        if ConversationalRetrievalChain is None:
            self.qa_chain = None
            return
        
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 10}) if self.vectorstore is not None else None,
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": PromptTemplate(
                template=prompt_template,
                input_variables=["context", "chat_history", "question"]
            )}
        )

    def _rerank_documents(self, question: str, docs: list) -> list:
        """Simple keyword overlap re-ranker to improve relevance (keeps your original approach)."""
        import re
        question_keywords = set(re.findall(r'\w+', question.lower()))
        scored_docs = []
        for doc in docs:
            content_lower = (doc.page_content if hasattr(doc, "page_content") else str(doc)).lower()
            score = sum(1 for keyword in question_keywords if keyword in content_lower and len(keyword) > 3)
            if any(keyword in content_lower for keyword in question_keywords if len(keyword) > 5):
                score += 2
            scored_docs.append((score, doc))
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_docs]

    def _build_enhanced_context(self, docs: list) -> str:
        """Build a compact context grouping by 'game' or 'genre' metadata (top relevant chunks)."""
        context = "=== RELEVANT GAME KNOWLEDGE ===\n\n"
        group = {}
        for doc in docs[:8]:
            meta = getattr(doc, "metadata", {}) if hasattr(doc, "metadata") else {}
            game = meta.get("game") or meta.get("title") or meta.get("genre") or "Unknown"
            group.setdefault(game, []).append(getattr(doc, "page_content", str(doc))[:800])
        for game, contents in group.items():
            context += f"📍 {game}:\n"
            for c in contents:
                context += f"{c}\n\n"
        context += "=== END OF KNOWLEDGE ===\n"
        return context

    def query(self, question: str, selected_game: Optional[str] = None):
        """Query the chatbot with the multi-stage pipeline (retrieval → rerank → prompt → LLM)."""
        if self.mode == "cli" and self.qa_chain is not None:
            result = self.qa_chain({"question": question})
            return result["answer"], result.get("source_documents", [])
        else:
            # STAGE 1: RETRIEVAL - get candidates
            if self.vectorstore is not None:
                # mimic your original similarity_search usage
                docs = self.vectorstore.similarity_search(question, k=12)
            else:
                docs = []

            # STAGE 2: optional filter by selected_game
            if selected_game:
                docs = [d for d in docs if selected_game.lower() in (d.metadata.get("game", "") or "").lower()]

            # STAGE 3: re-rank
            docs = self._rerank_documents(question, docs)

            # STAGE 4: context building
            context = self._build_enhanced_context(docs)

            # STAGE 5: system prompt for quality
            system_msg = """You are a games domain expert producing concise, deeply-knowledgeable answers.
Quality: aim for 9-10/10 factual, specific, and well-structured responses. Follow the 3-part structure from the assistant's prompt."""
            
            # Compose messages (include last few chat history items)
            messages = [{"role": "system", "content": system_msg}]
            for msg in self.chat_history[-6:]:
                messages.append({"role": msg["role"], "content": msg["content"]})
            
            user_prompt = f"""{context}

QUESTION: {question}

Provide a 9-10/10 quality answer following the structure and rules above. Be accurate, concise, and cite the games or mechanics used in the context when relevant."""
            messages.append({"role": "user", "content": user_prompt})

            # STAGE 6: call local Ollama HTTP API (like your original did)
            try:
                response = requests.post(
                    "http://localhost:11434/api/chat",
                    json={
                        "model": "llama3.2:1b",
                        "messages": messages,
                        "stream": False,
                        "options": {
                            "temperature": 0.4,
                            "num_predict": 400,
                            "top_p": 0.9,
                            "top_k": 30,
                            "repeat_penalty": 1.15,
                            "num_ctx": 4096
                        }
                    },
                    timeout=60
                )
                answer = response.json().get("message", {}).get("content", "Sorry, I couldn't generate an answer.")
            except Exception as e:
                # If Ollama isn't available, fallback to a short safe reply
                print("⚠️ Ollama HTTP call failed:", str(e))
                answer = "I couldn't reach the local LLM service. Please ensure Ollama (or your LLM endpoint) is running at http://localhost:11434."

            # Update chat history and return
            self.chat_history.append({"role": "user", "content": question})
            self.chat_history.append({"role": "assistant", "content": answer})
            return answer, docs[:5]

    def run_cli(self):
        """Interactive CLI mode identical in feel to your One Piece CLI."""
        print("\n" + "="*70)
        print("🎮  GAMES RAG CHATBOT - CLI MODE")
        print("="*70)
        print("\n💡 Ask about games, mechanics, meta, or lore.")
        print("💡 Type 'quit' or 'exit' to stop\n")

        while True:
            try:
                question = input("🤔 You: ").strip()
                if not question:
                    continue
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Bye!")
                    break
                print("\n🤖 Thinking... ", end="", flush=True)
                answer, sources = self.query(question)
                print("\n" + answer + "\n")
                if sources:
                    print(f"📚 Retrieved {len(sources)} sources (showing titles):")
                    for s in sources[:5]:
                        meta = getattr(s, "metadata", {})
                        print(" -", meta.get("game") or meta.get("title") or meta.get("genre"))
                print()
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")

        # If any explicit close is needed for client, do it
        try:
            self.weaviate_client.close()
        except Exception:
            pass


# ==================== FASTAPI MODE ====================
def run_api_mode():
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn

    app = FastAPI(title="Games RAG Chatbot API")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:3001"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class ChatMessage(BaseModel):
        role: str
        content: str

    class ChatRequest(BaseModel):
        message: str
        chat_history: Optional[List[ChatMessage]] = []
        selected_game: Optional[str] = None

    class ChatResponse(BaseModel):
        response: str
        sources: Optional[List[dict]] = []

    chatbot = None

    @app.on_event("startup")
    async def startup():
        nonlocal chatbot
        print("\n🚀 Starting Games RAG Chatbot API Server...")
        chatbot = GamesChatbot(mode="api")

    @app.get("/")
    async def root():
        return {"message": "Games RAG Chatbot API", "status": "ready"}

    @app.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest):
        try:
            if chatbot is None:
                raise HTTPException(status_code=503, detail="Chatbot not initialized yet. Please wait.")
            # set chat history from request if passed
            chatbot.chat_history = [{"role": m.role, "content": m.content} for m in request.chat_history] if request.chat_history else []
            answer, docs = chatbot.query(request.message, request.selected_game)
            # prepare sources snippet
            sources = []
            for d in docs[:4]:
                meta = getattr(d, "metadata", {})
                sources.append({"game": meta.get("game") or meta.get("title"), "snippet": getattr(d, "page_content", "")[:300]})
            return ChatResponse(response=answer, sources=sources)
        except Exception as e:
            print("❌ Error in /chat endpoint:", str(e))
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/clear")
    async def clear_history():
        if chatbot:
            chatbot.chat_history = []
        return {"message": "Chat history cleared"}

    @app.get("/health")
    async def health():
        return {"status": "healthy", "mode": "api"}

    print("📡 API: http://localhost:8000")
    print("📖 Docs: http://localhost:8000/docs\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)


# ==================== STREAMLIT WEB MODE ====================
def run_web_mode():
    """Streamlit UI adapted for Games theme. Keeps the same dynamic style but domain = games."""
    import streamlit as st
    st.set_page_config(page_title="Games Knowledge Bot", page_icon="🎮", layout="wide")

    # Minimal CSS swap (kept relatively light)
    st.markdown("""
    <style>
        .stApp { background: linear-gradient(135deg,#0f172a 0%,#001219 100%); color: #fff; }
        .stButton > button { background: linear-gradient(135deg,#06b6d4,#3b82f6); color: white; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

    if "chatbot" not in st.session_state:
        st.session_state.chatbot = GamesChatbot(mode="web")
        st.session_state.messages = []

    st.title("🎮 Games Knowledge Vault")
    st.write("Ask about games, mechanics, platform history, or meta. Example: \"How does recoil work in CS:GO?\"")

    with st.sidebar:
        st.markdown("### Controls")
        game_filter = st.text_input("Filter by game or genre (optional)")
        if st.button("Clear chat"):
            st.session_state.messages = []
            st.session_state.chatbot.chat_history = []
            st.experimental_rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about games..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge..."):
                answer, docs = st.session_state.chatbot.query(prompt, selected_game=game_filter or None)
                st.markdown(answer)
                if docs:
                    with st.expander(f"📚 Sources ({len(docs)})"):
                        for i, d in enumerate(docs[:4], 1):
                            meta = getattr(d, "metadata", {})
                            st.markdown(f"**{i}. {meta.get('game') or meta.get('title') or 'Unknown'}**\n\n{getattr(d, 'page_content', '')[:300]}...")
        st.session_state.messages.append({"role": "assistant", "content": answer})


# ==================== MAIN ====================
if __name__ == "__main__":
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "cli"

    if mode == "api":
        run_api_mode()
    elif mode == "web":
        run_web_mode()
    elif mode == "cli":
        chatbot = GamesChatbot(mode="cli")
        chatbot.run_cli()
    else:
        print("Usage: python app.py [cli|api|web]")
        print("  cli - Interactive command-line interface")
        print("  api - FastAPI REST API server")
        print("  web - Streamlit web interface")

