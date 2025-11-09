"""
ChefBot RAG Application - LOCAL & FAST
Domain: Cooking & Culinary Arts
Features: Weaviate vector DB, Microsoft Phi-2 (2.7B model), Fast embeddings
GPU Optimized: 5-15 second responses with 8-bit quantization on 4GB VRAM

Pipeline:
User Question → Fast Local Embedding → Weaviate Search → Top-K Chunks → Phi-2 Generation → Answer
"""

import os
import time
import torch
import weaviate
from typing import List, Dict, Optional
from PIL import Image
import pytesseract
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configure Tesseract path for Windows
if os.name == 'nt':  # Windows
    # Common installation path for Tesseract on Windows
    tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    if os.path.exists(tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
    else:
        # Try alternative path
        alt_path = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
        if os.path.exists(alt_path):
            pytesseract.pytesseract.tesseract_cmd = alt_path
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFacePipeline
from langchain.schema import HumanMessage, AIMessage
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class ChefBotRAG:
    """Main RAG chatbot class for cooking domain - uses Microsoft Phi-2 (2.7B)"""
    
    def __init__(self, weaviate_url: str = "http://localhost:8080"):
        """Initialize the RAG system with Phi-2 and local embeddings"""
        self.weaviate_url = weaviate_url
        self.client = None
        self.llm = None
        self.embeddings = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.collection_name = "CookingKnowledge"
        
        print("🤖 Initializing ChefBot RAG System (Phi-2 + Weaviate)...")
        self._setup_weaviate()
        self._setup_embeddings()
        self._setup_llm()
        print("✅ ChefBot is ready! (Running locally with Phi-2 🚀)")
    
    def _setup_weaviate(self):
        """Connect to Weaviate and create schema"""
        try:
            self.client = weaviate.Client(url=self.weaviate_url)
            
            # Check if collection exists, if not create it
            try:
                self.client.schema.get(self.collection_name)
                print(f"✅ Connected to Weaviate - Collection '{self.collection_name}' exists")
            except:
                # Create collection schema
                schema = {
                    "class": self.collection_name,
                    "description": "Cooking knowledge base including recipes, techniques, and ingredients",
                    "vectorizer": "none",  # We'll provide our own vectors
                    "properties": [
                        {
                            "name": "content",
                            "dataType": ["text"],
                            "description": "The text content"
                        },
                        {
                            "name": "source",
                            "dataType": ["text"],
                            "description": "Source of the content"
                        }
                    ]
                }
                self.client.schema.create_class(schema)
                print(f"✅ Created Weaviate collection: {self.collection_name}")
        except Exception as e:
            print(f"❌ Error connecting to Weaviate: {e}")
            print("Make sure Docker is running: docker-compose up -d")
            raise
    
    def _setup_embeddings(self):
        """Initialize fast local embedding model"""
        print("📦 Loading embedding model (fast & free)...")
        
        # Use fast local embeddings with GPU support
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': device}
        )
        print(f"✅ Embeddings ready on {device}")
    
    def _setup_llm(self):
        """Initialize Phi-2 - high quality 2.7B model perfect for 4GB VRAM"""
        print("🧠 Loading Phi-2 (2.7B, high quality, fits perfectly on 4GB GPU)...")
        
        # Using Phi-2 - excellent quality, small enough for RTX 3050
        model_name = "microsoft/phi-2"
        
        # Determine device with GPU info
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        if device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🎮 GPU: {gpu_name} ({gpu_memory:.1f} GB VRAM)")
            print("📦 Loading in 8-bit with CPU offload (optimized for 4GB)...")
        else:
            print("⚠️  No GPU detected, using CPU (much slower)")
        
        # Load with 8-bit quantization and CPU offloading for 4GB GPU
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        if torch.cuda.is_available():
            # 8-bit quantization with CPU offload for RTX 3050
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_8bit=True,
                device_map="auto",
                trust_remote_code=True,
                llm_int8_enable_fp32_cpu_offload=True
            )
        else:
            # CPU fallback
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=None,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        
        # Set padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        
        # Create pipeline - optimized for clear, concise, understandable responses
        text_gen_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=450,  # Balanced for concise yet complete answers
            temperature=0.7,     # Focused and coherent
            top_p=0.9,           # Quality over diversity
            top_k=50,            # Good variety
            repetition_penalty=1.2,  # Prevent repetition
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_full_text=False,
            no_repeat_ngram_size=3  # Prevent repeating phrases
        )
        
        self.llm = HuggingFacePipeline(pipeline=text_gen_pipeline)
        print(f"✅ LLM loaded: Phi-2 (2.7B, GPU-accelerated with 8-bit)")
    
    def _embed_text(self, text: str) -> List[float]:
        """Generate embedding for text using local model"""
        return self.embeddings.embed_query(text)
    
    def load_knowledge_base(self, file_path: str):
        """Load and index the cooking knowledge base"""
        print(f"📚 Loading knowledge base from: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            chunks = text_splitter.split_text(text)
            
            # Index chunks in Weaviate
            print(f"💾 Indexing {len(chunks)} chunks into Weaviate...")
            
            for i, chunk in enumerate(chunks):
                # Generate embedding using MiniLM
                embedding = self._embed_text(chunk)
                
                # Add to Weaviate
                self.client.data_object.create(
                    data_object={
                        "content": chunk,
                        "source": "cooking_knowledge.txt"
                    },
                    class_name=self.collection_name,
                    vector=embedding
                )
                
                if (i + 1) % 10 == 0:
                    print(f"  Indexed {i + 1}/{len(chunks)} chunks")
            
            print(f"✅ Knowledge base loaded successfully!")
            return len(chunks)
        
        except Exception as e:
            print(f"❌ Error loading knowledge base: {e}")
            raise
    
    def add_document(self, text: str, source: str = "user_upload"):
        """Add a new document to the knowledge base"""
        print(f"📄 Adding document from: {source}")
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )
        chunks = text_splitter.split_text(text)
        
        # Index chunks using local embeddings (MiniLM)
        for chunk in chunks:
            embedding = self._embed_text(chunk)
            self.client.data_object.create(
                data_object={
                    "content": chunk,
                    "source": source
                },
                class_name=self.collection_name,
                vector=embedding
            )
        
        print(f"✅ Added {len(chunks)} chunks from {source}")
        return len(chunks)
    
    def extract_text_from_image(self, image_path: str) -> str:
        """Extract text from image using OCR"""
        try:
            # Check if tesseract is available
            try:
                pytesseract.get_tesseract_version()
            except Exception:
                error_msg = (
                    "Tesseract OCR is not installed or not in your PATH.\n"
                    "Please install Tesseract:\n"
                    "1. Download from: https://github.com/UB-Mannheim/tesseract/wiki\n"
                    "2. Install it (typically to C:\\Program Files\\Tesseract-OCR\\)\n"
                    "3. Restart the application"
                )
                print(f"❌ {error_msg}")
                raise Exception(error_msg)
            
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            print(f"❌ Error extracting text from image: {e}")
            raise
    
    def retrieve_context(self, query: str, top_k: int = 2) -> List[str]:
        """Retrieve relevant context from Weaviate using embeddings"""
        # Generate query embedding
        query_embedding = self._embed_text(query)
        
        # Search in Weaviate
        result = self.client.query.get(
            self.collection_name,
            ["content", "source"]
        ).with_near_vector({
            "vector": query_embedding
        }).with_limit(top_k).do()
        
        # Extract content (limit to 300 chars per chunk)
        contexts = []
        if "data" in result and "Get" in result["data"]:
            items = result["data"]["Get"][self.collection_name]
            for item in items:
                content = item["content"]
                # Limit context length to avoid overwhelming model
                if len(content) > 300:
                    content = content[:300] + "..."
                contexts.append(content)
        
        return contexts
    
    def chat(self, user_input: str) -> Dict:
        """Main chat function with RAG and memory - Always uses RAG"""
        
        start_time = time.time()
        
        # Get conversation history (last 5 user inputs = 10 messages total)
        history = self.memory.load_memory_variables({}).get("chat_history", [])
        history_context = ""
        last_topic = ""
        
        if history:
            recent_messages = history[-10:]  # Last 10 messages (5 turns)
            for msg in recent_messages:
                role = "User" if msg.type == "human" else "ChefBot"
                history_context += f"{role}: {msg.content}\n"
            
            # Get last user question for context expansion
            for msg in reversed(recent_messages):
                if msg.type == "human":
                    last_topic = msg.content
                    break
        
        # Always retrieve context from knowledge base
        # If current query is short/vague, use last topic for better retrieval
        retrieval_start = time.time()
        search_query = user_input
        if last_topic and len(user_input.split()) < 5:  # Short query, likely a follow-up
            search_query = f"{last_topic} {user_input}"  # Combine for better search
        
        retrieved_contexts = self.retrieve_context(search_query)
        retrieval_time = time.time() - retrieval_start
        context = "\n\n".join(retrieved_contexts) if retrieved_contexts else ""
        
        # Detect if user wants a specific detail vs full recipe
        specific_keywords = ['what', 'which', 'spices', 'ingredients', 'temperature', 'time', 'how long', 'how much']
        recipe_keywords = ['how to make', 'recipe for', 'steps to', 'how do i make', 'cook']
        
        is_specific_question = any(keyword in user_input.lower() for keyword in specific_keywords)
        is_recipe_request = any(keyword in user_input.lower() for keyword in recipe_keywords)
        
        # Create prompt - adapt based on question type
        if context:
            if history_context:
                if is_specific_question and not is_recipe_request:
                    # Short, focused answer
                    prompt = f"""You are a helpful cooking assistant. Give clear, concise answers that are easy to understand.

Previous conversation:
{history_context}

Reference information:
{context[:1000]}

Question: {user_input}

Provide a clear, brief answer:"""
                else:
                    # Full recipe with clear instructions
                    prompt = f"""You are a helpful cooking assistant. Explain recipes in a clear, easy-to-understand way with proper steps.

Previous conversation:
{history_context}

Reference information:
{context[:1000]}

Question: {user_input}

Explain clearly and concisely, step-by-step:"""
            else:
                if is_specific_question and not is_recipe_request:
                    # Short, focused answer
                    prompt = f"""You are a helpful cooking assistant. Give clear, concise answers that are easy to understand.

Reference information:
{context[:1000]}

Question: {user_input}

Provide a clear, brief answer:"""
                else:
                    # Full recipe with clear instructions
                    prompt = f"""You are a helpful cooking assistant. Explain recipes in a clear, easy-to-understand way with proper steps.

Reference information:
{context[:1000]}

Question: {user_input}

Explain clearly and concisely, step-by-step:"""
        else:
            if history_context:
                prompt = f"""You are a helpful cooking assistant having a natural conversation. Answer in plain English.

Previous conversation:
{history_context}

Question: {user_input}

Answer:"""
            else:
                prompt = f"""You are a helpful cooking assistant. Answer in plain English sentences.

Question: {user_input}

Answer:"""
        
        # Generate response using local LLM
        try:
            generation_start = time.time()
            response = self.llm(prompt)
            generation_time = time.time() - generation_start
            
            # Clean up response
            response_text = response.strip()
            
            # Remove any system tags and prompt artifacts that might appear
            if "</s>" in response_text:
                response_text = response_text.split("</s>")[0].strip()
            if "<|" in response_text:
                response_text = response_text.split("<|")[0].strip()
            
            # Remove prompt instructions that leaked into response (check multiple times)
            cleanup_phrases = [
                "Do NOT write code.",
                "Do NOT write code",
                "Assistant:",
                "Answer:",
                "(in plain English, not code):",
                "(in plain English, not code)",
                "in plain English:",
                "not code):",
                "(Note:",  # Remove meta-notes about conversation
                "due to constraints",
            ]
            
            for phrase in cleanup_phrases:
                if phrase in response_text:
                    # Take everything before the phrase for notes
                    if phrase in ["(Note:", "due to constraints"]:
                        response_text = response_text.split(phrase)[0].strip()
                    else:
                        # Take everything after the phrase for instructions
                        parts = response_text.split(phrase)
                        if len(parts) > 1:
                            response_text = parts[-1].strip()
            
            # If it starts with "Answer:" remove it
            if response_text.startswith("Answer:"):
                response_text = response_text[7:].strip()
            
            # Fix common formatting issues from the model
            import re
            
            # Fix missing spaces after common words
            response_text = re.sub(r'(the)([A-Z])', r'\1 \2', response_text)  # "theOven" -> "the Oven"
            response_text = re.sub(r'(with)([A-Z])', r'\1 \2', response_text)  # "withSalt" -> "with Salt"
            response_text = re.sub(r'(and)([A-Z])', r'\1 \2', response_text)  # "andPepper" -> "and Pepper"
            response_text = re.sub(r'(of)([A-Z])', r'\1 \2', response_text)  # "ofthe" -> "of the"
            response_text = re.sub(r'(to)([A-Z])', r'\1 \2', response_text)  # "tothe" -> "to the"
            response_text = re.sub(r'(for)([A-Z])', r'\1 \2', response_text)  # "forOne" -> "for One"
            
            # Fix numbered list formatting - ensure space after number
            response_text = re.sub(r'(\d+)\.(\w)', r'\1. \2', response_text)  # "1.Preheat" -> "1. Preheat"
            response_text = re.sub(r'(\d+)\s+([A-Z])', r'\1. \2', response_text)  # "1 Preheat" -> "1. Preheat"
            
            # Remove excessive blank lines (more than 2 in a row)
            while '\n\n\n' in response_text:
                response_text = response_text.replace('\n\n\n', '\n\n')
            
            # Remove code-like formatting if present
            if response_text.startswith("```"):
                # Remove code blocks
                response_text = response_text.replace("```python", "").replace("```", "").strip()
            
            # If response looks like Python code (has = or []), extract natural text
            if "ingredients = [" in response_text or "recipe_" in response_text:
                # Try to extract any natural language before the code starts
                lines = response_text.split('\n')
                natural_lines = []
                for line in lines:
                    if '=' in line or line.strip().startswith('[') or line.strip().startswith('recipe_'):
                        break
                    natural_lines.append(line)
                if natural_lines:
                    response_text = '\n'.join(natural_lines).strip()
                else:
                    # Fallback: create a natural response
                    response_text = "I can help you with that recipe! Let me provide the details in a clear format."
            
            # For specific questions, keep answer concise - stop at first complete thought
            if is_specific_question and not is_recipe_request:
                # If response has numbered steps, only keep first 2-3 relevant points
                lines = response_text.split('\n')
                if len(lines) > 5 and any(line.strip().startswith(str(i)) for i in range(1, 10) for line in lines):
                    # Has numbered steps - keep only first few relevant ones
                    kept_lines = []
                    step_count = 0
                    for line in lines:
                        kept_lines.append(line)
                        if any(line.strip().startswith(f"{i}.") for i in range(1, 10)):
                            step_count += 1
                            if step_count >= 3:  # Keep max 3 steps for specific questions
                                break
                    response_text = '\n'.join(kept_lines).strip()
            
            # Detect and fix repetition loops - ONLY for extreme cases
            # Disabled for now to prevent over-filtering
            # The no_repeat_ngram_size in generation should handle repetition
            pass  # Placeholder
            
            # Final safety check - only if truly empty
            if not response_text:
                response_text = "I apologize, but I had trouble generating a response. Please try asking in a different way."
            
            # Update memory
            self.memory.save_context(
                {"input": user_input},
                {"output": response_text}
            )
            
            total_time = time.time() - start_time
            
            return {
                "response": response_text,
                "total_time": round(total_time, 2),
                "retrieval_time": round(retrieval_time, 2),
                "generation_time": round(generation_time, 2)
            }
        
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return {
                "response": "I apologize, but I encountered an error. Please try again.",
                "total_time": 0,
                "retrieval_time": 0,
                "generation_time": 0
            }
    
    def reset_memory(self):
        """Clear conversation memory"""
        self.memory.clear()
        print("🔄 Conversation memory cleared")
    
    def get_stats(self):
        """Get knowledge base statistics"""
        result = self.client.query.aggregate(self.collection_name).with_meta_count().do()
        count = 0
        if "data" in result and "Aggregate" in result["data"]:
            count = result["data"]["Aggregate"][self.collection_name][0]["meta"]["count"]
        return {"total_chunks": count}


if __name__ == "__main__":
    # Test the system
    bot = ChefBotRAG()
    
    # Load knowledge base if it exists
    kb_path = "cooking_knowledge.txt"
    if os.path.exists(kb_path):
        bot.load_knowledge_base(kb_path)
    
    # Interactive test
    print("\n" + "="*60)
    print("ChefBot Test Mode - Type 'quit' to exit")
    print("="*60 + "\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit']:
            break
        
        result = bot.chat(user_input)
        print(f"ChefBot: {result['response']}")
        print(f"⏱️  Time: {result['total_time']}s (Retrieval: {result['retrieval_time']}s, Generation: {result['generation_time']}s)\n")
