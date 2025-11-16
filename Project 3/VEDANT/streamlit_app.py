import streamlit as st
import requests

st.set_page_config(page_title="Indian Car Advisor", page_icon="🚗", layout="wide")
st.title("🚗 Indian Car RAG Chatbot")
st.caption("Ask about Indian cars - specs, comparisons, recommendations!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about Indian cars..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                res = requests.post("http://localhost:8000/chat", json={"message": prompt}, timeout=30)
                res.raise_for_status()
                result = res.json()
                st.markdown(result.get("response",""))
                if result.get("sources"):
                    with st.expander("📚 Sources"):
                        for src in result["sources"][:5]:
                            st.write(f"- {src.get('manufacturer','')} {src.get('model','')} ({src.get('year','')})")
                st.session_state.messages.append({"role":"assistant","content": result.get("response","")})
            except Exception as e:
                st.error("Error querying API: " + str(e))

with st.sidebar:
    st.header("Controls")
    if st.button("Clear Chat"):
        try:
            requests.post("http://localhost:8000/clear", timeout=5)
        except Exception:
            pass
        st.session_state.messages = []
        st.experimental_rerun()