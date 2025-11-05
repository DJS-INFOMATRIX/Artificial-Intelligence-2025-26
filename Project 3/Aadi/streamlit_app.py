import streamlit as st
from dotenv import load_dotenv
import os, weaviate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Weaviate
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain

load_dotenv()

st.set_page_config(page_title="🎵 Aadi's Music Assistant", page_icon="🎧", layout="centered")

st.markdown(
    "<h1 style='text-align:center;'>🎧 Aadi's Music Assistant</h1>",
    unsafe_allow_html=True,
)
st.caption("💡 Answers strictly from your custom dataset — powered by Gemini + Weaviate.")

client = weaviate.Client("http://localhost:8080")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

vectorstore = Weaviate(
    client=client,
    index_name="MusicRAG",
    text_key="text",
    embedding=embeddings,
    by_text=False
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

system_prompt = """
You are Aadi's personal music assistant trained strictly on the provided context.

<context>
{context}
</context>

Rules:
- Use ONLY the above context or previous chat history.
- If insufficient info, reply: "I don't have enough information from the provided data to answer that."
- Do not use general or external knowledge.
- Keep responses concise, natural, and human-like.

Chat History:
{chat_history}

User: {question}
Answer using only the context and chat history.
"""

prompt = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=system_prompt
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=st.session_state.memory,
    combine_docs_chain_kwargs={"prompt": prompt}
)

with st.sidebar:
    st.header("⚙️ Controls")
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.session_state.memory.clear()
        st.experimental_rerun()
    st.markdown("---")
    st.write("🧠 Chat memory remains active until cleared.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask about ragas, notes, or music theory..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(f"**You:** {user_input}")

    with st.chat_message("assistant"):
        with st.spinner("🎶 Thinking..."):
            response = qa_chain({"question": user_input})
            answer = response["answer"]
            st.markdown(f"**Assistant:** {answer}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
