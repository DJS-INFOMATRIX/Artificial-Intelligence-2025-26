"""
One Piece RAG Chatbot - Unified Application
Supports CLI, API Server (FastAPI), and Web Interface (Streamlit) modes
Run with: python app.py [cli|api|web]
"""

import os
import sys
import time
from typing import List, Optional
from dotenv import load_dotenv
import weaviate
from weaviate.classes.init import Auth
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_weaviate import WeaviateVectorStore
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferWindowMemory
import requests

load_dotenv()


class OnePieceChatbot:
    """Unified RAG-powered One Piece chatbot"""
    
    def __init__(self, mode="cli"):
        self.mode = mode
        self.setup_components()
    
    def setup_components(self):
        """Initialize Weaviate, embeddings, LLM, and memory"""
        print("🔌 Connecting to Weaviate...")
        self.weaviate_client = weaviate.connect_to_local(host="localhost", port=8080)
        
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        self.vectorstore = WeaviateVectorStore(
            client=self.weaviate_client,
            index_name="OnePieceArc",
            text_key="content",
            embedding=self.embeddings
        )
        
        print("🤖 Loading LLM...")
        self.llm = ChatOllama(model="llama3.2:1b", temperature=0.5)  # Using smaller 1B model for faster responses
        
        if self.mode == "cli":
            self.memory = ConversationBufferWindowMemory(
                memory_key="chat_history",
                return_messages=True,
                k=5,
                output_key="answer"
            )
            self.setup_rag_chain()
        else:
            self.chat_history = []
        
        print("✅ Chatbot ready!")
    
    def setup_rag_chain(self):
        """Setup conversational RAG chain with enhanced prompt for 9-10/10 quality"""
        prompt_template = """You are a knowledgeable One Piece analyst providing exceptional answers.

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
• Be specific: names, abilities, exact events
• Write naturally - like an expert analyst, not a robot
• Focus on the arc/topic most relevant to the question
• If you're not certain, say so clearly

STYLE:
• Professional yet passionate
• Natural conversation, not academic report
• No phrases like "according to context" or "the document states"
• Sound like a knowledgeable friend explaining One Piece

CONTEXT:
{context}

CONVERSATION HISTORY:
{chat_history}

QUESTION: {question}

ANSWER (following the structure above for 9-10/10 quality):"""
        
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 10}),  # More candidates
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": PromptTemplate(
                template=prompt_template,
                input_variables=["context", "chat_history", "question"]
            )}
        )
    
    def _rerank_documents(self, question: str, docs: list) -> list:
        """Re-rank documents by relevance to question using keyword matching"""
        import re
        
        question_keywords = set(re.findall(r'\w+', question.lower()))
        
        scored_docs = []
        for doc in docs:
            content_lower = doc.page_content.lower()
            # Score based on keyword overlap
            score = sum(1 for keyword in question_keywords if keyword in content_lower and len(keyword) > 3)
            # Bonus for exact phrases
            if any(keyword in content_lower for keyword in question_keywords if len(keyword) > 5):
                score += 2
            scored_docs.append((score, doc))
        
        # Sort by score descending
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_docs]
    
    def _build_enhanced_context(self, docs: list, question: str) -> str:
        """Build rich context with focus on most relevant chunks"""
        context = "=== RELEVANT ONE PIECE KNOWLEDGE ===\n\n"
        
        # Group by arc for clarity
        arc_groups = {}
        for doc in docs[:5]:  # Top 5 most relevant
            arc = doc.metadata.get('arc', 'Unknown')
            if arc not in arc_groups:
                arc_groups[arc] = []
            arc_groups[arc].append(doc.page_content)
        
        # Build context with arc grouping
        for arc, contents in arc_groups.items():
            context += f"📍 {arc} Arc:\n"
            for content in contents:
                context += f"{content}\n\n"
        
        context += "=== END OF KNOWLEDGE ===\n"
        return context
    
    def query(self, question: str, selected_arc: Optional[str] = None):
        """Query the chatbot with enhanced RAG pipeline for 9-10/10 quality"""
        if self.mode == "cli":
            result = self.qa_chain({"question": question})
            return result["answer"], result.get("source_documents", [])
        else:
            # STAGE 1: RETRIEVAL - Get more candidates
            docs = self.vectorstore.similarity_search(question, k=12)  # Increased for better coverage
            
            # STAGE 2: FILTERING - Apply arc filter if specified
            if selected_arc:
                docs = [d for d in docs if selected_arc.lower() in d.metadata.get("arc", "").lower()]
            
            # STAGE 3: RE-RANKING - Semantic re-ranking by relevance
            docs = self._rerank_documents(question, docs)
            
            # STAGE 4: CONTEXT BUILDING - Enhanced structured context
            context = self._build_enhanced_context(docs, question)
            
            # STAGE 5: PROMPT ENGINEERING - Ultra-precise instructions
            system_msg = """You are a knowledgeable One Piece analyst who provides exceptional, detailed answers.

🎯 QUALITY STANDARDS (Must achieve 9-10/10):
• Factual Accuracy: 4/4 - Every detail must be correct
• Context Relevance: 3/3 - Stay focused on the specific topic asked
• Clarity & Style: 2/2 - Natural, engaging, professional tone
• Completeness: 1/1 - Fully answer the question with supporting details

📋 ANSWER STRUCTURE (Follow exactly):
1. START with a direct, clear answer (1 sentence)
2. PROVIDE supporting evidence with specific details (2-4 sentences)
3. END with insight or summary if relevant (1 sentence)

✅ RULES:
• Use ONLY information from the provided knowledge base
• Be specific: mention character names, abilities, exact events
• Write naturally - like a calm, expert analyst explaining to a fan
• Never say "Document", "Context", or "According to" - just tell the story
• If multiple arcs are mentioned, focus on the one most relevant to the question
• If uncertain, clearly state what you know and what you don't

❌ NEVER:
• Mix up different arcs or events
• Make assumptions or guess details
• Give vague or incomplete answers
• Use robotic academic language

🎨 TONE: Professional yet passionate, like a knowledgeable friend who deeply understands both the facts and emotions of One Piece.

EXAMPLE QUALITY:
Question: "Did Luffy fight at Marineford?"
✅ PERFECT: "Yes, Luffy fought his way through Marineford during the war to rescue Ace. He battled countless Marines and even crossed paths with powerful figures like Mihawk and the Admirals, though he avoided prolonged duels since his focus was saving his brother. After Ace's death, Luffy suffered a complete emotional breakdown - one of the most devastating moments in the series."
"""
            
            messages = [{"role": "system", "content": system_msg}]
            
            # Add recent chat history (limited for focus)
            for msg in self.chat_history[-4:]:  # Last 2 exchanges
                messages.append({"role": msg["role"], "content": msg["content"]})
            
            # STAGE 6: QUERY CONSTRUCTION - Clear, focused prompt
            user_prompt = f"""{context}

QUESTION: {question}

Provide a 9-10/10 quality answer following the structure and rules above. Focus on being accurate, detailed, and naturally conversational."""
            
            messages.append({"role": "user", "content": user_prompt})
            
            # STAGE 7: GENERATION - Optimized parameters for quality
            response = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": "llama3.2:1b",
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.5,  # Balanced: factual + natural
                        "num_predict": 400,  # Reduced for smaller model
                        "top_p": 0.9,
                        "top_k": 30,
                        "repeat_penalty": 1.15,  # Reduce repetition
                        "num_ctx": 4096  # Larger context window
                    }
                }
            )
            
            answer = response.json()["message"]["content"]
            
            # STAGE 8: POST-PROCESSING (optional validation)
            # Could add length check, keyword validation, etc.
            
            return answer, docs[:5]  # Return top 5 docs for source reference
    
    def run_cli(self):
        """Run interactive CLI mode"""
        print("\n" + "="*70)
        print("🏴‍☠️  ONE PIECE RAG CHATBOT - CLI MODE")
        print("="*70)
        print("\n💡 Ask about: Marineford, Wano, Enies Lobby")
        print("💡 Type 'quit' or 'exit' to stop\n")
        
        while True:
            try:
                question = input("🤔 You: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Thanks for chatting! Sayonara!")
                    break
                
                print("\n🤖 Luffy's Brain: ", end="", flush=True)
                answer, sources = self.query(question)
                print(answer)
                
                if sources:
                    print(f"\n📚 Retrieved from {len(sources)} sources")
                print()
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
        
        self.weaviate_client.close()


# ==================== FASTAPI MODE ====================
def run_api_mode():
    """Run FastAPI server mode"""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
    
    app = FastAPI(title="One Piece RAG Chatbot API")
    
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
        selected_arc: Optional[str] = None
    
    class ChatResponse(BaseModel):
        response: str
        sources: Optional[List[dict]] = []
    
    chatbot = None
    
    @app.on_event("startup")
    async def startup():
        global chatbot
        print("\n🚀 Starting One Piece RAG Chatbot API Server...")
        chatbot = OnePieceChatbot(mode="api")
    
    @app.get("/")
    async def root():
        return {"message": "One Piece RAG Chatbot API", "status": "ready"}
    
    @app.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest):
        try:
            if chatbot is None:
                raise HTTPException(status_code=503, detail="Chatbot not initialized yet. Please wait.")
            
            # Set chat history from request
            if request.chat_history:
                chatbot.chat_history = [{"role": m.role, "content": m.content} for m in request.chat_history]
            else:
                chatbot.chat_history = []
            
            # Get answer from chatbot
            answer, docs = chatbot.query(request.message, request.selected_arc)
            
            # Update chat history
            chatbot.chat_history.append({"role": "user", "content": request.message})
            chatbot.chat_history.append({"role": "assistant", "content": answer})
            
            # Prepare sources
            sources = [{"arc": d.metadata.get("arc"), "content": d.page_content[:200]} for d in docs[:3]]
            
            return ChatResponse(response=answer, sources=sources)
        except Exception as e:
            print(f"❌ Error in /chat endpoint: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/clear")
    async def clear_history():
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
    """Run Streamlit web interface with epic One Piece theme"""
    import streamlit as st
    
    st.set_page_config(
        page_title="One Piece Knowledge Bot", 
        page_icon="🏴‍☠️", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Epic One Piece CSS with animations
    st.markdown("""
    <style>
        /* Main background - Ocean gradient with waves */
        .stApp {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 25%, #0f3460 50%, #533483 75%, #e94560 100%);
            background-size: 400% 400%;
            animation: gradientWave 15s ease infinite;
        }
        
        @keyframes gradientWave {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        /* Sidebar - Pirate Ship theme */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
            border-right: 5px solid #ffd700;
            box-shadow: 5px 0 20px rgba(255, 215, 0, 0.3);
        }
        
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {
            color: #ffffff !important;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }
        
        /* Chat messages with animation */
        [data-testid="stChatMessage"] {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            border: 2px solid rgba(255, 215, 0, 0.3);
            margin: 10px 0;
            animation: messageSlide 0.5s ease-out;
        }
        
        @keyframes messageSlide {
            from { 
                opacity: 0;
                transform: translateX(-20px);
            }
            to { 
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        /* Input box - Treasure chest style */
        [data-testid="stChatInput"] {
            border: 3px solid #ffd700;
            border-radius: 20px;
            background: rgba(0, 0, 0, 0.3);
            box-shadow: 0 0 20px rgba(255, 215, 0, 0.5);
        }
        
        /* Buttons - Devil Fruit style */
        .stButton > button {
            background: linear-gradient(135deg, #e94560 0%, #533483 100%) !important;
            color: white !important;
            border: 2px solid #ffd700 !important;
            border-radius: 15px !important;
            font-weight: bold !important;
            padding: 10px 25px !important;
            transition: all 0.3s !important;
            box-shadow: 0 5px 15px rgba(233, 69, 96, 0.4) !important;
        }
        
        .stButton > button:hover {
            transform: scale(1.05) !important;
            box-shadow: 0 8px 25px rgba(233, 69, 96, 0.6) !important;
        }
        
        /* Expander - Wanted Poster style */
        .streamlit-expanderHeader {
            background: linear-gradient(135deg, #533483 0%, #0f3460 100%);
            border: 3px solid #ffd700;
            border-radius: 10px;
            color: white !important;
            font-weight: bold;
        }
        
        /* Loading animation container */
        .loading-container {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            z-index: 9999;
        }
        
        .jolly-roger {
            font-size: 120px;
            animation: spin 2s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .loading-text {
            color: #ffd700;
            font-size: 32px;
            font-weight: bold;
            text-shadow: 3px 3px 6px rgba(0,0,0,0.7);
            margin-top: 20px;
            animation: pulse 1.5s ease-in-out infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 0.6; transform: scale(1); }
            50% { opacity: 1; transform: scale(1.1); }
        }
        
        .waves {
            margin-top: 30px;
            font-size: 40px;
            animation: wave 2s ease-in-out infinite;
        }
        
        @keyframes wave {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-15px); }
        }
        
        /* ZORO SLASH SCREEN ANIMATION */
        .slash-screen {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(135deg, #0a1929 0%, #1e3a5f 100%);
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            z-index: 9998;
        }
        
        .start-button {
            font-size: 48px;
            font-weight: bold;
            color: #ffd700;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            border: 5px solid #ffd700;
            border-radius: 20px;
            padding: 30px 80px;
            cursor: pointer;
            transition: all 0.3s;
            box-shadow: 0 10px 40px rgba(255, 215, 0, 0.5);
            animation: buttonPulse 2s ease-in-out infinite;
            text-shadow: 3px 3px 6px rgba(0,0,0,0.8);
        }
        
        .start-button:hover {
            transform: scale(1.1);
            box-shadow: 0 15px 60px rgba(255, 215, 0, 0.8);
        }
        
        @keyframes buttonPulse {
            0%, 100% { transform: scale(1); box-shadow: 0 10px 40px rgba(255, 215, 0, 0.5); }
            50% { transform: scale(1.05); box-shadow: 0 15px 60px rgba(255, 215, 0, 0.8); }
        }
        
        .slash-title {
            color: #ffd700;
            font-size: 56px;
            font-weight: bold;
            text-shadow: 4px 4px 8px rgba(0,0,0,0.8);
            margin-bottom: 30px;
            animation: titleGlow 2s ease-in-out infinite;
        }
        
        @keyframes titleGlow {
            0%, 100% { text-shadow: 4px 4px 8px rgba(0,0,0,0.8), 0 0 20px rgba(255, 215, 0, 0.5); }
            50% { text-shadow: 4px 4px 8px rgba(0,0,0,0.8), 0 0 40px rgba(255, 215, 0, 0.9); }
        }
        
        /* Zoro character animation */
        .zoro-container {
            position: fixed;
            bottom: -500px;
            right: -300px;
            font-size: 200px;
            z-index: 10000;
            animation: zoroSlash 1.5s cubic-bezier(0.68, -0.55, 0.265, 1.55);
            pointer-events: none;
        }
        
        @keyframes zoroSlash {
            0% {
                bottom: -500px;
                right: -300px;
                transform: rotate(0deg);
            }
            50% {
                bottom: 50%;
                right: 50%;
                transform: rotate(45deg) scale(1.5);
            }
            100% {
                bottom: 150%;
                right: 150%;
                transform: rotate(90deg) scale(0.5);
            }
        }
        
        /* Slash effect - diagonal line */
        .slash-line {
            position: fixed;
            top: -10%;
            left: -10%;
            width: 150%;
            height: 10px;
            background: linear-gradient(90deg, transparent 0%, #00ff00 30%, #ffffff 50%, #00ff00 70%, transparent 100%);
            transform: rotate(45deg);
            opacity: 0;
            z-index: 10001;
            box-shadow: 0 0 30px #00ff00, 0 0 60px #00ff00, 0 0 90px #00ff00;
            animation: slashLine 0.3s ease-out 0.7s;
        }
        
        @keyframes slashLine {
            0% { opacity: 0; height: 0px; }
            30% { opacity: 1; height: 15px; }
            100% { opacity: 0; height: 5px; }
        }
        
        /* Screen split effect */
        .screen-split-top {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 50%;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            z-index: 9999;
            transform-origin: bottom right;
            animation: splitTop 0.8s ease-out 1s forwards;
        }
        
        .screen-split-bottom {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            height: 50%;
            background: linear-gradient(135deg, #16213e 0%, #0f3460 100%);
            z-index: 9999;
            transform-origin: top left;
            animation: splitBottom 0.8s ease-out 1s forwards;
        }
        
        @keyframes splitTop {
            0% { transform: translateY(0) translateX(0); opacity: 1; }
            100% { transform: translateY(-100%) translateX(100%); opacity: 0; }
        }
        
        @keyframes splitBottom {
            0% { transform: translateY(0) translateX(0); opacity: 1; }
            100% { transform: translateY(100%) translateX(-100%); opacity: 0; }
        }
        
        /* Flash effect on slash */
        .slash-flash {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: white;
            z-index: 10002;
            opacity: 0;
            animation: flashEffect 0.2s ease-out 0.8s;
        }
        
        @keyframes flashEffect {
            0% { opacity: 0; }
            50% { opacity: 0.8; }
            100% { opacity: 0; }
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Loading screen with One Piece animation
    if "chatbot" not in st.session_state:
        loading_placeholder = st.empty()
        
        with loading_placeholder.container():
            st.markdown("""
            <div class="loading-container">
                <div class="jolly-roger">🏴‍☠️</div>
                <div class="loading-text">⚡ LOADING ONE PIECE KNOWLEDGE ⚡</div>
                <div class="waves">🌊 🌊 🌊 🌊 🌊</div>
                <div style="color: #ffd700; font-size: 18px; margin-top: 20px; text-align: center;">
                    Connecting to Weaviate...<br>
                    Loading Llama 3.2 Model...<br>
                    Preparing RAG Pipeline...
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Initialize chatbot
        st.session_state.chatbot = OnePieceChatbot(mode="web")
        st.session_state.messages = []
        
        # Clear loading screen
        loading_placeholder.empty()
    
    # Epic header with animation
    st.markdown("""
    <h1 style='text-align: center; color: #ffd700; font-size: 4em; text-shadow: 4px 4px 8px rgba(0,0,0,0.8); animation: pulse 2s ease-in-out infinite;'>
        🏴‍☠️ ONE PIECE KNOWLEDGE VAULT 🏴‍☠️
    </h1>
    <p style='text-align: center; color: #ffffff; font-size: 1.5em; text-shadow: 2px 2px 4px rgba(0,0,0,0.7);'>
        ⚡ Ask anything about Marineford, Wano, or Enies Lobby! ⚡
    </p>
    """, unsafe_allow_html=True)
    
    # Sidebar with pirate controls
    with st.sidebar:
        st.markdown("### 🏴‍☠️ PIRATE CONTROLS")
        
        arc_filter = st.selectbox(
            "🗺️ Filter by Arc:",
            ["All Arcs", "Marineford", "Wano", "Enies Lobby"],
            help="Focus on a specific arc for better answers"
        )
        
        st.markdown("---")
        
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chatbot.chat_history = []
            st.rerun()
        
        st.markdown("---")
        st.markdown("""
        <div style='background: rgba(255, 215, 0, 0.1); padding: 15px; border-radius: 10px; border: 2px solid #ffd700;'>
            <h4 style='color: #ffd700; margin: 0;'>⚔️ TIPS</h4>
            <ul style='color: white; font-size: 14px;'>
                <li>Ask specific questions</li>
                <li>Use character names</li>
                <li>Request story details</li>
                <li>Follow-up questions work!</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("<p style='color: #ffd700; text-align: center; font-size: 12px;'>Powered by RAG + Llama 3.2 🚀</p>", unsafe_allow_html=True)
    
    # Chat messages with style
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🤠" if msg["role"] == "user" else "🏴‍☠️"):
            st.markdown(f"<div style='color: white; font-size: 16px;'>{msg['content']}</div>", unsafe_allow_html=True)
    
    # Chat input with epic style
    if prompt := st.chat_input("🗣️ Ask about One Piece..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user", avatar="🤠"):
            st.markdown(f"<div style='color: white; font-size: 16px;'>{prompt}</div>", unsafe_allow_html=True)
        
        with st.chat_message("assistant", avatar="🏴‍☠️"):
            with st.spinner("🤔 Searching the Grand Line..."):
                selected = None if arc_filter == "All Arcs" else arc_filter
                answer, docs = st.session_state.chatbot.query(prompt, selected)
                
                st.markdown(f"<div style='color: white; font-size: 16px;'>{answer}</div>", unsafe_allow_html=True)
                
                if docs:
                    with st.expander(f"📚 Sources ({len(docs)} documents from the archives)"):
                        for i, doc in enumerate(docs[:3], 1):
                            arc = doc.metadata.get('arc', 'Unknown')
                            st.markdown(f"""
                            <div style='background: rgba(255, 215, 0, 0.1); padding: 10px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #ffd700;'>
                                <strong style='color: #ffd700;'>📍 {i}. {arc} Arc</strong><br>
                                <span style='color: white; font-size: 14px;'>{doc.page_content[:300]}...</span>
                            </div>
                            """, unsafe_allow_html=True)
        
        st.session_state.messages.append({"role": "assistant", "content": answer})


# ==================== MAIN ====================
if __name__ == "__main__":
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "cli"
    
    if mode == "api":
        run_api_mode()
    elif mode == "web":
        run_web_mode()
    elif mode == "cli":
        chatbot = OnePieceChatbot(mode="cli")
        chatbot.run_cli()
    else:
        print("Usage: python app.py [cli|api|web]")
        print("  cli - Interactive command-line interface")
        print("  api - FastAPI REST API server")
        print("  web - Streamlit web interface")
