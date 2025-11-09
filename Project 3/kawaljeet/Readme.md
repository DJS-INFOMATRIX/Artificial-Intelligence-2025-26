# 🍳 ChefBot - AI-Powered Cooking Assistant

A **Retrieval-Augmented Generation (RAG)** chatbot specialized in cooking and culinary arts, running entirely on your local machine with GPU acceleration.

**Tech Stack**:
- 🧠 **LLM**: Microsoft Phi-2 (2.7B parameters)
- 📊 **Vector DB**: Weaviate (Docker-based)
- 🔤 **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
- 🌐 **Web UI**: Flask with AJAX
- 👁️ **OCR**: Tesseract v5.4.0

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Features

- 🤖 **Smart Cooking Assistant** - Ask any cooking question and get expert answers
- 🔍 **RAG-Powered Search** - Retrieves relevant context from a curated cooking knowledge base
- 💬 **Conversational Memory** - Remembers your conversation history (last 5 exchanges)
- 📄 **PDF Upload** - Extract recipes and cooking tips from PDF files
- 🖼️ **Image OCR** - Extract text from recipe images using Tesseract OCR
- 🚀 **GPU Accelerated** - Optimized for RTX 3050 (4GB VRAM) with 8-bit quantization
- 🌐 **Modern Web UI** - Clean, responsive Flask-based interface

## 🏗️ Architecture

### Core Components

1. **Vector Database**: Weaviate (Docker) - Stores cooking knowledge as embeddings
2. **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` - Fast local embeddings
3. **LLM**: Microsoft Phi-2 (2.7B parameters) - High-quality reasoning with 4GB VRAM support
4. **OCR Engine**: Tesseract - Extract text from images
5. **Web Framework**: Flask - Modern responsive UI

### RAG Pipeline

```
User Question → Embedding → Vector Search (Weaviate) → Top-K Context Retrieval 
    → Prompt Construction → Phi-2 Generation → Answer
```

## 📋 Prerequisites

### Required Software

1. **Python 3.8+** - [Download](https://www.python.org/downloads/)
2. **Docker Desktop** - [Download](https://www.docker.com/products/docker-desktop/)
3. **Tesseract OCR** - [Download for Windows](https://github.com/UB-Mannheim/tesseract/wiki)
4. **CUDA Toolkit** (Optional, for GPU acceleration) - [Download](https://developer.nvidia.com/cuda-downloads)

### Hardware Requirements

**Minimum (CPU only)**:
- CPU: 4+ cores
- RAM: 8GB
- Storage: 10GB free space

**Recommended (GPU accelerated)**:
- GPU: NVIDIA RTX 3050 or better (4GB+ VRAM)
- RAM: 16GB
- Storage: 15GB free space

## 🚀 Quick Start

### 1. Clone & Navigate

```powershell
git clone <repository-url>
cd ChefBot
```

### 2. Install Dependencies

```powershell
# Install Python packages
pip install -r requirements.txt
```

### 3. Install Tesseract OCR

Download and install from: https://github.com/UB-Mannheim/tesseract/wiki

Default installation path: `C:\Program Files\Tesseract-OCR\`

Verify installation:
```powershell
& "C:\Program Files\Tesseract-OCR\tesseract.exe" --version
```

### 4. Start Weaviate Database

```powershell
# Start Weaviate in Docker
docker-compose up -d

# Verify it's running
docker ps
```

You should see Weaviate running on port 8080.

### 5. Launch ChefBot

```powershell
python enhanced_ui.py
```

**First startup takes 30-60 seconds** to:
- Download Phi-2 model (~5GB)
- Load embeddings (~100MB)
- Initialize knowledge base

### 6. Access the UI

Open your browser: **http://127.0.0.1:5000**

## 📖 Usage Guide

### Basic Chat

1. Type your cooking question in the input box
2. Click "Send" or press Enter
3. Wait for ChefBot to respond (5-15 seconds)

**Example Questions**:
- "How do I make homemade pasta?"
- "What's the difference between baking soda and baking powder?"
- "Give me a recipe for chocolate chip cookies"

### File Uploads

#### Upload PDF
1. Click "Choose File"
2. Select a PDF with recipes or cooking information
3. Click "Upload PDF"
4. The content is extracted and added to the knowledge base
5. Ask questions about the uploaded content

#### Upload Image
1. Click "Choose File" 
2. Select an image (PNG/JPG) with recipe text
3. Click "Upload Image"
4. Tesseract extracts text from the image
5. Content is added to knowledge base
6. Ask questions about the recipe

### Settings

- **RAG Mode Toggle**: 
  - ON (default): Uses knowledge base for context-aware answers
  - OFF: Direct LLM responses without retrieval
  
- **Clear Memory**: Reset conversation history

- **View Stats**: Check database statistics
  - Total chunks in knowledge base
  - Memory usage
  - GPU information

## 🔧 Configuration

### Customize Tesseract Path

If Tesseract is installed in a non-standard location, edit `app.py`:

```python
# Around line 21
tesseract_path = r'C:\Your\Custom\Path\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = tesseract_path
```

### Adjust GPU Memory

For GPUs with different VRAM, modify `app.py`:

```python
# Line ~149 - For 8GB+ VRAM, disable 8-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=False,  # Change to False for full precision
    device_map="auto",
    trust_remote_code=True
)
```

### Change LLM Model

To use a different model, edit `app.py` line ~111:

```python
# Options:
# model_name = "microsoft/phi-2"              # Current (2.7B)
# model_name = "TinyLlama/TinyLlama-1.1B"    # Faster, smaller
# model_name = "mistralai/Mistral-7B-v0.1"   # Better quality, needs more VRAM
```

## 📁 Project Structure

```
ChefBot/
├── app.py                  # Core RAG engine & ChefBotRAG class
├── enhanced_ui.py          # Flask web interface
├── requirements.txt        # Python dependencies
├── docker-compose.yml      # Weaviate configuration
├── cooking_knowledge.txt   # Initial knowledge base
├── ARCHITECTURE.md         # Detailed system architecture
├── TESSERACT_SETUP.md      # OCR setup guide
└── README.md              # This file
```

## 🐛 Troubleshooting

### Weaviate Connection Error

```
❌ Error connecting to Weaviate
```

**Solution**: Start Docker and Weaviate
```powershell
docker-compose up -d
```

### Tesseract Not Found

```
❌ tesseract is not installed or it's not in your PATH
```

**Solution**: 
1. Install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
2. Verify installation path in `app.py` (line 21)
3. Restart the application

### CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solution**: The app already uses 8-bit quantization. If still occurring:
1. Close other GPU applications
2. Reduce batch size in generation (line ~150)
3. Use CPU mode (automatic fallback)

### Slow Response Times

**On CPU**: 30-60 seconds per response (normal)
**On GPU**: 5-15 seconds per response (normal)

**Speed up**:
- Ensure GPU is being used (check console output)
- Update CUDA drivers
- Close other applications

### Model Download Issues

If model download fails:

```powershell
# Manually download via Python
python -c "from transformers import AutoModel; AutoModel.from_pretrained('microsoft/phi-2')"
```

## 🎯 Advanced Usage

### Add Custom Knowledge

1. Create a text file with cooking information
2. Use the built-in upload function, or
3. Programmatically add via Python:

```python
from app import ChefBotRAG

bot = ChefBotRAG()
bot.add_document("your_knowledge.txt", "custom_source")
```

### Check Database Stats

Access the stats page through the UI or via Python:

```python
stats = bot.get_stats()
print(f"Total chunks: {stats['total_documents']}")
```

### Export Conversation

Conversation history is stored in memory. To export:

```python
history = bot.memory.chat_memory.messages
for msg in history:
    print(f"{msg.type}: {msg.content}")
```

## 🔒 Privacy & Security

- ✅ **100% Local** - All processing happens on your machine
- ✅ **No API Keys** - No external API calls
- ✅ **No Data Sent** - Your recipes and questions stay private
- ✅ **Offline Capable** - Works without internet (after initial model download)

## 📊 Performance Benchmarks

**System**: RTX 3050 (4GB VRAM), i7-11800H, 16GB RAM

| Operation | Time | Notes |
|-----------|------|-------|
| First message (cold start) | 30-45s | Model loading |
| Follow-up questions | 5-15s | Normal operation |
| PDF extraction | 2-5s | Depends on size |
| Image OCR | 3-8s | Depends on quality |
| Knowledge base search | <1s | Vector search |

## 🛠️ Development

### Run in Debug Mode

```powershell
# Edit enhanced_ui.py, change debug mode
python enhanced_ui.py
```

### View Logs

Detailed logs are printed to console. To save:

```powershell
python enhanced_ui.py 2>&1 | Tee-Object -FilePath chefbot.log
```

### Reset Database

```powershell
# Stop and remove Weaviate data
docker-compose down -v

# Restart fresh
docker-compose up -d
python enhanced_ui.py  # Will recreate schema
```

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional cooking knowledge sources
- UI enhancements
- Performance optimizations
- New features (voice input, meal planning, etc.)

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- **Weaviate** - Vector database
- **Hugging Face** - Transformers and models
- **Microsoft** - Phi-2 model
- **Tesseract** - OCR engine
- **LangChain** - RAG framework

## 📞 Support

For issues and questions:
1. Check [ARCHITECTURE.md](ARCHITECTURE.md) for technical details
2. Review troubleshooting section above
3. Open an issue on GitHub

---

**Made with ❤️ for cooking enthusiasts and AI lovers**

*Last Updated: November 9, 2025*
