# Music RAG Chatbot

A local music theory chatbot using RAG (Retrieval-Augmented Generation) with Weaviate, LangChain, and Google Gemini.

## Overview

This chatbot answers questions about music theory strictly from a custom dataset using:
- **Weaviate v1.23.12** - Vector database for semantic search
- **Google Gemini 2.5 Flash** - Large Language Model
- **Google text-embedding-004** - Text embeddings
- **LangChain** - RAG framework
- **Streamlit** - Web interface

## Architecture

```
User Question → Embedding → Vector Search → Context Retrieval → LLM → Answer
```

The system converts questions to vectors, searches the database for similar content, and uses retrieved context to generate accurate answers without hallucinations.

## Setup

### Prerequisites
- Python 3.10+
- Docker Desktop
- Google Gemini API key

### Installation

1. **Clone and navigate to project:**
```bash
cd "Project 3/Aadi"
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Create `.env` file with your API key:**
```
GEMINI_API_KEY=your_api_key_here
```

4. **Start Weaviate database:**
```bash
docker-compose up -d
```

5. **Ingest music data (first time only):**
```bash
python ingest_data.py
```

6. **Run the Streamlit app:**
```bash
streamlit run streamlit_app.py
```

7. **Open browser to:** http://localhost:8501

## Usage

Ask questions about:
- Music theory (scales, chords, rhythm)
- Music history (rock, jazz, classical, hip hop)
- Instruments and techniques
- Any topic in the music_data.txt file

The chatbot maintains conversation memory for follow-up questions.

## Project Structure

```
.
├── docker-compose.yml     # Weaviate configuration
├── streamlit_app.py      # Main chatbot application
├── ingest_data.py        # Data loading script
├── music_data.txt        # Knowledge base
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Technical Details

### RAG Pipeline
1. **Text Chunking**: Splits music_data.txt into 800-character chunks with 100-character overlap
2. **Embedding**: Converts chunks to 768-dimensional vectors using Google's embedding model
3. **Vector Storage**: Stores embeddings in Weaviate for fast similarity search
4. **Retrieval**: Finds top 4 most relevant chunks for each question
5. **Generation**: Gemini generates answers using only retrieved context

### Key Features
- **Conversational Memory**: Remembers chat history for context-aware responses
- **Strict Context Adherence**: Only answers from provided data
- **Semantic Search**: Uses vector similarity instead of keyword matching
- **Persistent Storage**: Weaviate data survives container restarts

## Stopping the Application

```bash
# Stop Weaviate
docker-compose down
```

## Troubleshooting

**Port 8080 already in use:**
- Change port in docker-compose.yml and update Python files

**Connection refused:**
- Wait 5-10 seconds for Weaviate to initialize
- Check: `docker ps` to verify container is running

**API errors:**
- Verify GEMINI_API_KEY in .env file
- Check Google Cloud quota limits

## Technologies

| Component | Technology | Version |
|-----------|-----------|---------|
| Vector DB | Weaviate | 1.23.12 |
| LLM | Google Gemini | 2.5 Flash |
| Embeddings | Google | text-embedding-004 |
| Framework | LangChain | 0.3+ |
| UI | Streamlit | 1.51+ |
| Container | Docker | Latest |

## Author

Aadi - AI Project 3

