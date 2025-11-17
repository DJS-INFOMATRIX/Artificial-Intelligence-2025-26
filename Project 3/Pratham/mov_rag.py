# -*- coding: utf-8 -*-
# Bollywood Movies RAG Chatbot with LangChain + Weaviate
# Domain: 5 Popular Indian Movies

import weaviate
import json
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import warnings
warnings.filterwarnings('ignore')

# Connect to Weaviate
print("Connecting to Weaviate...")
try:
    client = weaviate.connect_to_local(host="localhost", port=8080)
    print("Connected to Weaviate!")
except Exception as e:
    print(f"ERROR: Could not connect to Weaviate: {e}")
    print("Run: docker run -d -p 8080:8080 -p 50051:50051 --name weaviate semitechnologies/weaviate:latest")
    exit(1)


print("Setting up Weaviate schema...")
try:
    if client.collections.exists("BollywoodMovie"):
        client.collections.delete("BollywoodMovie")
except:
    pass

from weaviate.classes.config import Configure, Property, DataType

client.collections.create(
    name="BollywoodMovie",
    vectorizer_config=Configure.Vectorizer.none(),
    properties=[
        Property(name="content", data_type=DataType.TEXT),
        Property(name="movie_title", data_type=DataType.TEXT),
        Property(name="chunk_id", data_type=DataType.INT)
    ]
)
print("Schema created!")

# Load movie data from JSON
print("Loading movie data from JSON file...")
def load_movies_from_json(json_path):
    if not os.path.exists(json_path):
        print(f"JSON file not found: {json_path}")
        return {}
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = {}
    
    if "sections" in data:
        # Collect all paragraphs
        all_paragraphs = []
        for section in data["sections"]:
            for content_item in section.get("content", []):
                if content_item["type"] == "paragraph":
                    all_paragraphs.append(content_item["text"])
        
        full_text = "\n\n".join(all_paragraphs)
        
        # Split by main movie titles
        movie_titles = ["Dangal", "3 Idiots", "Zindagi Na Milegi Dobara", "Dilwale", "Jawan"]
        
        for i, title in enumerate(movie_titles):
           
            start_idx = full_text.find(title)
            if start_idx != -1:
                
                if i + 1 < len(movie_titles):
                    end_idx = full_text.find(movie_titles[i + 1], start_idx + len(title))
                    if end_idx == -1:
                        end_idx = len(full_text)
                else:
                    end_idx = len(full_text)
                
                
                movie_content = full_text[start_idx:end_idx].strip()
                if movie_content:
                    results[title] = movie_content
    
    return results

json_path = os.path.join(os.path.dirname(__file__), 'movie_structured.json')
movie_texts = load_movies_from_json(json_path)

if not movie_texts:
    print("ERROR: Could not load movie data!")
    exit(1)

for title, text in movie_texts.items():
    print(f"Loaded {title} ({len(text)} characters)")

print("Chunking documents...")
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100, separators=["\n\n", "\n", ". ", " "])

# Load embedding model
print("Loading embedding model...")
embeddings_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Store in Weaviate
print("Storing chunks in Weaviate...")
total_chunks = 0
collection = client.collections.get("BollywoodMovie")

for movie_title, text in movie_texts.items():
    chunks = splitter.split_text(text)
    print(f"{movie_title}: {len(chunks)} chunks")
    
    for i, chunk in enumerate(chunks):
        embedding = embeddings_model.embed_query(chunk)
        collection.data.insert(
            properties={
                "content": chunk,
                "movie_title": movie_title,
                "chunk_id": i
            },
            vector=embedding
        )
        total_chunks += 1

print(f"Stored {total_chunks} total chunks across {len(movie_texts)} movies")

# Load LLM
print("Loading language model (Flan-T5-Base - better quality)...")
from transformers import AutoModelForSeq2SeqLM

model_name = "google/flan-t5-base"  # Better quality, ~1GB

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

text_gen_pipeline = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=150,
    do_sample=True,
    temperature=0.7
)
print("Model loaded!")

conversation_history = []
print("Memory initialized (keeping last 3 turns)")


def retrieve_context(query, k=3):
    query_embedding = embeddings_model.embed_query(query)
    collection = client.collections.get("BollywoodMovie")
    
    response = collection.query.near_vector(
        near_vector=query_embedding,
        limit=k,
        return_properties=["content", "movie_title"]
    )
    
    contexts = []
    for obj in response.objects:
        contexts.append({
            "content": obj.properties.get("content", ""),
            "movie": obj.properties.get("movie_title", "Unknown")
        })
    
    return contexts


def chat_with_bot(user_query):
    print(f"\n{'='*60}")
    print(f"USER: {user_query}")
    print(f"{'='*60}")
    
    print("Retrieving relevant information...")
    contexts = retrieve_context(user_query, k=3)
    
    if not contexts:
        retrieved_text = "No relevant information found in the knowledge base."
    else:
        retrieved_text = "\n\n".join([f"[From {ctx['movie']}]: {ctx['content']}" for ctx in contexts])
    
    print(f"Retrieved {len(contexts)} relevant chunks")
    
    # Load conversation history 
    global conversation_history
    history = "\n".join([f"User: {h['user']}\nAssistant: {h['assistant']}" for h in conversation_history[-2:]])  # Only 2 turns to save space
    
    
    prompt = f"Answer this question based on the context.\n\nContext: {retrieved_text}\n\nQuestion: {user_query}\n\nAnswer:"
    
    print("Generating response...")
    generated = text_gen_pipeline(prompt)[0]['generated_text']
    
    response = generated.strip()
    
    
    conversation_history.append({"user": user_query, "assistant": response})
    
    print(f"\nAns:\n{response}")
    return response

# Main chatbot
print("\n" + "="*60)
print("BOLLYWOOD MOVIES RAG CHATBOT")
print("="*60)
print("Ask me anything about these movies:")
for title in movie_texts.keys():
    print(f"  - {title}")
print("\nType 'quit' to exit")
print("="*60)


while True:
    user_input = input("\nYou: ").strip()
    if user_input.lower() in ['quit', 'exit', 'bye']:
        print("Goodbye!")
        break
    if user_input:
        chat_with_bot(user_input)

print("\nChatbot session ended!")
