from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Weaviate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import weaviate, os

load_dotenv()

client = weaviate.Client("http://localhost:8080")

with open("music_data.txt", "r", encoding="utf-8") as f:
    text = f.read()

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
docs = splitter.split_text(text)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

Weaviate.from_texts(
    texts=docs,
    embedding=embeddings,
    client=client,
    index_name="MusicRAG",
)

print("Data ingested successfully")
