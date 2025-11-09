# ChefBot Architecture Overview

**Last Updated**: November 9, 2025  
**Version**: 2.0 (Enhanced UI with File Uploads)

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       USER INTERFACE                            │
│                  (enhanced_ui.py - Flask)                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  Chat Input  │  │ PDF Upload   │  │ Image Upload │         │
│  │  (AJAX/JSON) │  │  (PyPDF2)    │  │ (Tesseract)  │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  RAG Toggle  │  │ Clear Memory │  │  View Stats  │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ↓
┌────────────────────────────────────────────────────────────────┐
│                      CHEFBOT RAG ENGINE                        │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │             CONVERSATIONAL MEMORY                        │  │
│  │       (ConversationBufferMemory - Last 5 turns)          │  │
│  └──────────────────────────────────────────────────────────┘  │
│                             ↓                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │            QUERY PROCESSING & EMBEDDING                  │  │
│  │  • User question                                         │  │
│  │  • Convert to 384-dim vector                             │  │
│  │  • GPU/CPU acceleration                                  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                             ↓                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │         CONTEXT RETRIEVAL (RAG) - Toggle ON/OFF          │  │
│  │  • Semantic search in Weaviate (cosine similarity)       │  │
│  │  • Retrieve top-2 relevant chunks (configurable)         │  │
│  │  • Include metadata (source, content)                    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                             ↓                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │            PROMPT CONSTRUCTION                           │  │
│  │  System Prompt + Context + History + Question            │  │
│  │  → Formatted for Phi-2                                   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                             ↓                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │       LLM GENERATION (Microsoft Phi-2 - 2.7B)            │  │
│  │  • 8-bit quantization for 4GB VRAM                       │  │
│  │  • CPU offloading enabled                                │  │
│  │  • Max 512 tokens, temp 0.7                              │  │
│  │  • Context-aware response generation                     │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────┬───────────────────────────────────┘
                             │
                             ↓
                     Generated Answer
```

## 🔄 RAG Pipeline Flow

```
User Question: "How do I make pizza dough?"
       │
       ↓
┌─────────────────────┐
│  EMBEDDING MODEL    │
│  (MiniLM-L6-v2)     │  Converts question → [384-dim vector]
│  Device: GPU/CPU    │  e.g., [0.23, -0.15, 0.87, ..., 0.42]
└─────────────────────┘
       │
       ↓
┌─────────────────────┐
│   WEAVIATE DB       │
│  (Vector Search)    │  Cosine similarity search
│  localhost:8080     │
│                     │
│  Chunk 1: 0.89 sim  │  "Pizza dough requires flour, water..."
│  Chunk 2: 0.85 sim  │  "Mix ingredients, knead for 10 mins..."
└─────────────────────┘
       │
       ↓
┌─────────────────────┐
│  CONTEXT BUILDER    │  Combines:
│                     │  • Top-2 retrieved chunks
│                     │  • Last 5 conversation turns
│                     │  • System instructions
└─────────────────────┘
       │
       ↓
┌─────────────────────┐
│   PHI-2 MODEL       │  Generates answer using:
│  (2.7B params)      │  • Retrieved context
│  8-bit quantized    │  • Conversation history
│  4GB VRAM optimized │  • Model's knowledge
└─────────────────────┘
       │
       ↓
Answer: "To make pizza dough, you'll need 3 cups flour..."
```

## 📦 Component Breakdown

### 1. **Weaviate (Vector Database)**
```
┌──────────────────────────────────┐
│         WEAVIATE v1.23.7         │
│  Running on: localhost:8080      │
│  Running in: Docker              │
├──────────────────────────────────┤
│  Collection: CookingKnowledge    │
│                                  │
│  Schema:                         │
│  ├─ content (text)               │
│  │   - Stores text chunks        │
│  │   - Max ~800 characters       │
│  │                               │
│  ├─ source (text)                │
│  │   - Origin filename           │
│  │   - e.g., "cooking_knowledge" │
│  │                               │
│  └─ vector (384 dimensions)      │
│      - Semantic embedding        │
│      - Generated by MiniLM       │
│                                  │
│  Operations:                     │
│  • add_document() - Add content  │
│  • retrieve_context() - Search   │
│  • Cosine similarity matching    │
│  • Sub-second query performance  │
└──────────────────────────────────┘
```

### 2. **Embedding Model**
```
┌──────────────────────────────────┐
│  sentence-transformers/          │
│  all-MiniLM-L6-v2                │
├──────────────────────────────────┤
│  Specifications:                 │
│  • Output: 384 dimensions        │
│  • Model size: ~100MB            │
│  • Speed: ~1000 sentences/sec    │
│  • Device: CUDA or CPU           │
│                                  │
│  Conversion Process:             │
│  Text → Tokenization → Model     │
│       → 384D Vector              │
│                                  │
│  Example:                        │
│  "Pizza dough"                   │
│    ↓                             │
│  [0.23, -0.15, 0.87, ..., 0.42]  │
│                                  │
│  Similarity Calculation:         │
│  cosine_similarity(vec1, vec2)   │
│  → Score between 0 and 1         │
└──────────────────────────────────┘
```

### 3. **LLM (Microsoft Phi-2)**
```
┌────────────────────────────────────┐
│      Microsoft Phi-2               │
│      (2.7 Billion Parameters)      │
├────────────────────────────────────┤
│  Optimizations:                    │
│  ✓ 8-bit quantization (load_in_8bit)│
│  ✓ CPU offloading for 4GB VRAM    │
│  ✓ Device map: auto               │
│  ✓ Trust remote code: enabled     │
│                                    │
│  Hardware Support:                 │
│  • GPU: NVIDIA RTX 3050+ (4GB)    │
│  • CPU: Fallback mode available   │
│  • Memory: ~3GB VRAM + 4GB RAM    │
│                                    │
│  Generation Settings:              │
│  • Max tokens: 512                │
│  • Temperature: 0.7               │
│  • Top-k: 50                      │
│  • Top-p: 0.95                    │
│  • Repetition penalty: 1.1        │
│                                    │
│  Performance:                      │
│  • GPU: 5-15 seconds/response     │
│  • CPU: 30-60 seconds/response    │
│                                    │
│  Input: Formatted prompt           │
│  Output: Generated text response   │
└────────────────────────────────────┘
```

### 4. **Memory System**
```
┌─────────────────────────────────────┐
│   ConversationBufferMemory          │
│   (LangChain)                       │
├─────────────────────────────────────┤
│  Configuration:                     │
│  • Stores last 5 conversation turns│
│  • Return messages: True            │
│  • Memory key: "chat_history"       │
│                                     │
│  Storage Format:                    │
│  Turn 1:                            │
│    Human: "How to make pasta?"      │
│    AI: "Here's how to make pasta..."│
│                                     │
│  Turn 2:                            │
│    Human: "What about sauce?"       │
│    AI: "For the sauce you mentioned│
│         with pasta..."              │
│                                     │
│  Features:                          │
│  • Context continuity               │
│  • Follow-up questions              │
│  • Clear memory function            │
│  • Automatic truncation at 5 turns  │
└─────────────────────────────────────┘
```

### 5. **OCR Engine (Tesseract)**
```
┌─────────────────────────────────────┐
│   Tesseract OCR v5.4.0              │
│   (Google's Open Source OCR)        │
├─────────────────────────────────────┤
│  Installation Path (Windows):       │
│  C:\Program Files\Tesseract-OCR\    │
│                                     │
│  Supported Formats:                 │
│  • PNG, JPG, JPEG                   │
│  • Recipe cards                     │
│  • Printed text                     │
│  • Handwriting (limited)            │
│                                     │
│  Process:                           │
│  Image → PIL.Image.open()           │
│       → pytesseract.image_to_string()│
│       → Extracted text              │
│                                     │
│  Performance:                       │
│  • Speed: 3-8 seconds/image         │
│  • Accuracy: ~90% on clear text     │
│                                     │
│  Error Handling:                    │
│  • Path validation                  │
│  • Clear error messages             │
│  • Installation instructions        │
└─────────────────────────────────────┘
```

## 🔀 Data Flow: Upload PDF

```
User uploads PDF via Flask UI
       │
       ↓
┌─────────────────────┐
│   secure_filename()  │  Sanitize filename
└─────────────────────┘
       │
       ↓
┌─────────────────────┐
│   PyPDF2.PdfReader  │  Extract all text from pages
│                     │  Combines all pages into single text
└─────────────────────┘
       │
       ↓
┌─────────────────────┐
│  Text Splitter      │  RecursiveCharacterTextSplitter
│  Chunk size: 800    │  • Break into 800-char chunks
│  Overlap: 100       │  • 100-char overlap for context
│                     │  • Smart splitting at sentences
└─────────────────────┘
       │
       ↓
┌─────────────────────┐
│  For each chunk:    │
│  1. Generate vector │  MiniLM → [384-dim array]
│  2. Create object   │  {content, source, vector}
│  3. Store in DB     │  Weaviate.batch.add_data_object()
└─────────────────────┘
       │
       ↓
   ✅ Indexed & Searchable
   Total chunks stored in database
```

## 🔀 Data Flow: Upload Image

```
User uploads Image (PNG/JPG) via Flask UI
       │
       ↓
┌─────────────────────┐
│  File Validation    │  Check extension, size
│  secure_filename()  │  Sanitize filename
└─────────────────────┘
       │
       ↓
┌─────────────────────┐
│  PIL.Image.open()   │  Load image
└─────────────────────┘
       │
       ↓
┌─────────────────────┐
│  Tesseract OCR      │  Extract text from image
│  (pytesseract)      │  • Handles multiple fonts
│                     │  • Recipe cards, printed text
└─────────────────────┘
       │
       ↓
┌─────────────────────┐
│  Text Splitter      │  Break into 800-char chunks
│                     │  with 100-char overlap
└─────────────────────┘
       │
       ↓
┌─────────────────────┐
│  Embedding + Store  │  Generate vectors & index
│                     │  Store in Weaviate
└─────────────────────┘
       │
       ↓
   ✅ Indexed & Searchable
```

## 🔀 Data Flow: Chat Query (RAG Mode ON)

```
User: "How to make carbonara?"
       │
       ↓
┌─────────────────────┐
│  Frontend (Flask)   │  POST /chat
│  AJAX Request       │  {message, use_rag: true}
└─────────────────────┘
       │
       ↓
┌─────────────────────┐
│  get_bot()          │  Lazy load ChefBotRAG
│  (Singleton)        │  (30-60s first time)
└─────────────────────┘
       │
       ↓
┌─────────────────────┐
│  query_with_rag()   │  Main RAG pipeline
└─────────────────────┘
       │
       ↓
┌─────────────────────┐
│  Embed Query        │  "carbonara" → [0.15, -0.22, ...]
│  MiniLM Model       │
└─────────────────────┘
       │
       ↓
┌─────────────────────┐
│  Weaviate Search    │  Cosine similarity search
│  Top-2 Results      │  Returns most relevant chunks
└─────────────────────┘
       │
       ↓
┌─────────────────────┐
│  Build Prompt       │  System + Context + History + Q
│                     │  Formatted for Phi-2
└─────────────────────┘
       │
       ↓
┌─────────────────────┐
│  Phi-2 Generation   │  Generate answer (5-15s GPU)
│  Max 512 tokens     │
└─────────────────────┘
       │
       ↓
┌─────────────────────┐
│  Save to Memory     │  Store Q&A in buffer
│  (Last 5 turns)     │
└─────────────────────┘
       │
       ↓
   Response → Flask → Frontend
   Display in chat UI
```

## 🔀 Data Flow: Chat Query (RAG Mode OFF)

```
User: "Tell me about Italian cuisine"
       │
       ↓
┌─────────────────────┐
│  Frontend (Flask)   │  POST /chat
│  AJAX Request       │  {message, use_rag: false}
└─────────────────────┘
       │
       ↓
┌─────────────────────┐
│  query()            │  Direct LLM query
│  (No RAG)           │  Skips vector search
└─────────────────────┘
       │
       ↓
┌─────────────────────┐
│  Build Prompt       │  System + History + Question
│                     │  No context retrieval
└─────────────────────┘
       │
       ↓
┌─────────────────────┐
│  Phi-2 Generation   │  Pure model knowledge
└─────────────────────┘
       │
       ↓
┌─────────────────────┐
│  Save to Memory     │  Store in conversation buffer
└─────────────────────┘
       │
       ↓
   Response → Frontend
```

## 🎯 Why This Architecture?

### 1. **Separation of Concerns**
```
┌─────────────────────────────────┐
│  Layer            │  File        │
├─────────────────────────────────┤
│  Presentation     │  enhanced_ui │  Flask web UI, file handling
│  Business Logic   │  app.py      │  RAG engine, LLM, memory
│  Data Storage     │  Weaviate    │  Vector database
│  Infrastructure   │  Docker      │  Container orchestration
└─────────────────────────────────┘
```

### 2. **Modularity & Extensibility**
- ✅ **Easy LLM swapping**: Change `model_name` in app.py
- ✅ **UI independence**: Flask can be replaced with Gradio/Streamlit
- ✅ **Database flexibility**: Weaviate can be swapped for Pinecone/Chroma
- ✅ **Embedding models**: Change sentence-transformers model easily

### 3. **Scalability Considerations**
```
Current Capacity:
├─ Weaviate: Millions of vectors supported
├─ Embeddings: 1000+ sentences/second
├─ LLM: Stateless (can serve multiple users)
└─ Memory: Isolated per session

Scaling Options:
├─ Horizontal: Multiple Flask instances + load balancer
├─ Vertical: Better GPU (8GB+ VRAM, no quantization)
├─ Distributed: Weaviate cluster for large datasets
└─ Caching: Redis for frequent queries
```

### 4. **Performance Optimizations**
```
┌─────────────────────────────────────┐
│  Optimization         │  Impact     │
├─────────────────────────────────────┤
│  8-bit quantization   │  50% memory │
│  CPU offloading       │  4GB GPU OK │
│  Lazy loading         │  Fast start │
│  Vector indexing      │  <1s search │
│  Batch embeddings     │  3x faster  │
└─────────────────────────────────────┘
```

### 5. **Local-First Philosophy**
```
Why Local?
✅ Privacy: Data never leaves your machine
✅ No API costs: Zero ongoing fees
✅ Offline capable: Works without internet
✅ Customizable: Full control over models
✅ Fast: No network latency
✅ Secure: No data breaches possible
```

## 📊 Performance Characteristics

### First-Time Startup (Cold Start)
```
┌──────────────────────────────────────────────┐
│  Task                 │  Time    │  Size     │
├──────────────────────────────────────────────┤
│  Phi-2 Download       │  10-20m  │  5.5 GB   │
│  MiniLM Download      │  1-2m    │  100 MB   │
│  Weaviate Start       │  10-30s  │  Docker   │
│  Model Loading (GPU)  │  30-45s  │  RAM      │
│  Model Loading (CPU)  │  45-90s  │  RAM      │
│  Schema Creation      │  1-2s    │  N/A      │
│  Initial Index        │  30-60s  │  1000 docs│
├──────────────────────────────────────────────┤
│  TOTAL (first time)   │  15-25m  │  ~6 GB    │
└──────────────────────────────────────────────┘
```

### Subsequent Runs (Warm Start)
```
┌──────────────────────────────────────────────┐
│  Task                 │  Time    │  Notes    │
├──────────────────────────────────────────────┤
│  Weaviate Start       │  5-10s   │  If stopped│
│  Model Loading (GPU)  │  20-30s  │  From disk│
│  Model Loading (CPU)  │  30-60s  │  From disk│
│  Flask Server Start   │  2-3s    │  Instant  │
│  First Query (lazy)   │  30-45s  │  +load bot│
├──────────────────────────────────────────────┤
│  TOTAL (warm start)   │  30-60s  │  Typical  │
└──────────────────────────────────────────────┘
```

### Runtime Performance (GPU - RTX 3050)
```
┌──────────────────────────────────────────────┐
│  Operation            │  Time    │  Notes    │
├──────────────────────────────────────────────┤
│  Chat query (RAG ON)  │  5-15s   │  Normal   │
│  Chat query (RAG OFF) │  3-8s    │  Faster   │
│  Embedding generation │  10-50ms │  Per query│
│  Vector search        │  50-200ms│  Weaviate │
│  LLM generation       │  4-12s   │  Main time│
│  PDF upload (10 pages)│  3-8s    │  Extract  │
│  Image OCR            │  3-8s    │  Per image│
│  Memory save          │  <10ms   │  Fast     │
└──────────────────────────────────────────────┘
```

### Runtime Performance (CPU - Fallback)
```
┌──────────────────────────────────────────────┐
│  Operation            │  Time    │  Notes    │
├──────────────────────────────────────────────┤
│  Chat query (RAG ON)  │  30-90s  │  Slower   │
│  Chat query (RAG OFF) │  20-60s  │  Still slow│
│  Embedding generation │  100-300ms│ Manageable│
│  Vector search        │  50-200ms│  Same     │
│  LLM generation       │  25-80s  │  Bottleneck│
└──────────────────────────────────────────────┘
```

### Memory Usage
```
┌──────────────────────────────────────────────┐
│  Component            │  GPU     │  RAM      │
├──────────────────────────────────────────────┤
│  Phi-2 (8-bit)        │  3.2 GB  │  1 GB     │
│  Phi-2 (16-bit, CPU)  │  N/A     │  6 GB     │
│  MiniLM Embeddings    │  0.3 GB  │  0.5 GB   │
│  Weaviate (100K docs) │  N/A     │  2-4 GB   │
│  Flask + System       │  N/A     │  1-2 GB   │
├──────────────────────────────────────────────┤
│  TOTAL (GPU mode)     │  ~3.5 GB │  ~5 GB    │
│  TOTAL (CPU mode)     │  0 GB    │  ~10 GB   │
└──────────────────────────────────────────────┘
```

## 🔐 Security & Privacy Considerations

### Current Setup: 100% Local
```
┌─────────────────────────────────────────────┐
│  Component         │  Location  │  Network  │
├─────────────────────────────────────────────┤
│  Weaviate DB       │  localhost │  :8080    │
│  Flask Web Server  │  localhost │  :5000    │
│  Phi-2 Model       │  Local GPU │  None     │
│  Embeddings        │  Local GPU │  None     │
│  User Files        │  Temp dir  │  None     │
│  Conversations     │  RAM only  │  None     │
└─────────────────────────────────────────────┘

✅ No external API calls
✅ No data leaves your machine
✅ No telemetry or tracking
✅ Works fully offline (after setup)
✅ No authentication needed (single user)
```

### Security Features
```
File Upload Security:
├─ Filename sanitization (secure_filename)
├─ Extension validation (.pdf, .png, .jpg only)
├─ Size limit (16MB max)
├─ Temporary storage only
└─ No arbitrary code execution

Database Security:
├─ Local-only access (no external exposure)
├─ Anonymous access (single-user mode)
├─ No persistent credentials
└─ Data in Docker volume (isolated)

Web Server Security:
├─ Development server (not production)
├─ Localhost binding only
├─ CSRF protection (built-in Flask)
└─ No user authentication (local use)
```

### Production Hardening (If Deploying)
```
⚠️  Current setup is for LOCAL USE ONLY

For production deployment, add:
├─ Authentication: User login, session management
├─ HTTPS: SSL/TLS certificates
├─ Weaviate Security: API keys, authentication
├─ Rate Limiting: Prevent abuse
├─ Input Validation: Sanitize all inputs
├─ WSGI Server: Gunicorn/uWSGI instead of Flask dev
├─ Reverse Proxy: Nginx for load balancing
├─ Monitoring: Logging, alerting, metrics
├─ Backup: Database backup strategy
└─ Updates: Regular security patches
```

### Privacy Guarantees
```
✅ Your recipe uploads stay on your computer
✅ Your questions never sent to external APIs
✅ No cookies or tracking
✅ No data collection
✅ No model fine-tuning on your data
✅ Memory cleared on restart
✅ No persistent conversation logs (unless you save them)
```

## 🔄 Typical Conversation Flow Examples

### Example 1: Basic Recipe Query (RAG Mode)
```
1. User: "How do I sauté vegetables?"
   
   Process:
   ├─ Embed: "sauté vegetables" → vector
   ├─ Search: Weaviate finds relevant chunks
   │   • "Sautéing is a cooking technique..." (0.91 similarity)
   │   • "Heat oil in pan, add vegetables..." (0.87 similarity)
   ├─ Build prompt with context
   └─ Phi-2 generates: 
       "To sauté vegetables, heat 2 tbsp oil in a large pan over 
        medium-high heat. Add harder vegetables first (carrots, 
        broccoli), cook 3-4 mins. Add softer ones (peppers, 
        mushrooms) and cook until tender-crisp, about 5-7 mins 
        total. Season with salt and pepper."

2. User: "What temperature should I use?"
   
   Process:
   ├─ Memory recalls: Previous Q about sautéing
   ├─ Search: "sauté temperature" in context
   ├─ Phi-2 (with context):
       "For sautéing vegetables that you asked about, use 
        medium-high heat, around 350-375°F (175-190°C). This 
        allows quick cooking while maintaining texture."

3. User: "Any tips for beginners?"
   
   Process:
   ├─ Full conversation context maintained
   ├─ Builds on sautéing topic
   └─ Phi-2 generates:
       "For sautéing vegetables as a beginner: 1) Don't 
        overcrowd the pan, 2) Keep ingredients moving, 3) Have 
        all ingredients prepped, 4) Use high smoke-point oils 
        like vegetable or canola."
```

### Example 2: PDF Upload → Query Flow
```
1. User uploads: "italian_recipes.pdf" (15 pages)
   
   Process:
   ├─ Extract text: 12,450 characters
   ├─ Split into chunks: 18 chunks created
   ├─ Generate embeddings: 18 vectors
   ├─ Store in Weaviate: Success
   └─ Response: "✅ Added 18 chunks from italian_recipes.pdf"

2. User: "What's the authentic carbonara recipe?"
   
   Process:
   ├─ Search newly indexed content
   ├─ Find: Carbonara recipe chunk (0.94 similarity)
   └─ Phi-2 generates:
       "According to the Italian recipe guide you uploaded, 
        authentic carbonara uses guanciale (cured pork jowl), 
        eggs, Pecorino Romano cheese, and black pepper. No 
        cream! Mix egg yolks with grated cheese, cook pasta, 
        crisp guanciale, toss together off heat."
```

### Example 3: Image OCR → Query Flow
```
1. User uploads: "grandmas_recipe_card.jpg"
   
   Process:
   ├─ Tesseract OCR extracts text:
   │   "Grandma's Apple Pie
   │    Ingredients: 6 apples, 2 cups flour, 1 cup sugar...
   │    Instructions: Peel and slice apples..."
   ├─ Split into chunks: 3 chunks
   ├─ Index in Weaviate
   └─ Response: "✅ Extracted 245 characters, added 3 chunks"

2. User: "What apples did grandma use?"
   
   Process:
   ├─ Search: Finds recipe card chunk
   └─ Phi-2 generates:
       "Based on the recipe card image you uploaded, your 
        grandma's apple pie used 6 apples. The recipe suggests 
        Granny Smith or Honeycrisp for best results."
```

### Example 4: RAG Toggle Comparison
```
RAG Mode ON (Default):
User: "How to make bread dough?"
→ Searches knowledge base
→ Uses retrieved cooking instructions
→ Answer: Detailed recipe with measurements from KB

RAG Mode OFF:
User: "How to make bread dough?"
→ No knowledge base search
→ Pure Phi-2 knowledge
→ Answer: General bread-making info from model training
→ May be less specific or domain-focused
```

### Example 5: Memory Clear Demo
```
Session 1:
User: "My favorite cuisine is Thai"
Bot: "Thai cuisine is wonderful! Would you like recipes?"
User: "What's my favorite cuisine?"
Bot: "You mentioned Thai cuisine is your favorite."

[User clicks "Clear Memory"]

Session 2:
User: "What's my favorite cuisine?"
Bot: "I don't have information about your favorite cuisine. 
      Could you tell me?"
```

---

**This architecture balances:**
- ⚡ Speed (small model)
- 🎯 Accuracy (RAG retrieval)
- 💾 Memory (conversation context)
- 🔧 Flexibility (easy to extend)
