import streamlit as st
import requests
import json
import time
import os
# Default to localhost (Native mode). Docker will override this via Environment Variable.
API_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000/predict")
# API_URL = "http://backend:8000/predict"

# --- Page Config ---
st.set_page_config(
    page_title="Mini-RAG Demo",
    page_icon="🤖",
    layout='centered'
)

# --- Custom CSS for appealing look ---
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    [data-testid="stChatMessageContent"] p {
        font-size: 1.1rem;
    }
    .source-box {
        border-left: 4px solid #f63366;
        background-color: #f0f2f6;
        padding: 10px;
        margin-top: 5px;
        border-radius: 5px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.title("🧠 Mini-RAG System")
st.caption("Production-style Retrieval Augmented Generation demo. Ask questions about the ingested document.")
st.divider()

# --- Sidebar State ---
with st.sidebar:
    st.header("System Status")
    st.success("Backend Connected", icon="✅")
    st.info("Model: FLAN-T5-Large (CPU)", icon="💻")
    st.info("Retriever: FAISS (MiniLM-L6)", icon="🔍")
    
    st.divider()
    top_k = st.slider("Retrieval Depth (Top-K)", min_value=1, max_value=5, value=3)

# --- Main Logic ---
# Initialize chat history if not present
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
if prompt := st.chat_input("Ask a question about the document..."):
    # 1. Display user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Call Backend API
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking... 🧠")
        
        try:
            payload = {"query": prompt, "top_k": top_k}
            
            with st.spinner("Retrieving & Generating..."):
                response = requests.post(API_URL, json=payload, timeout=120)
                response.raise_for_status() # Raise exception for bad status codes
                data = response.json()

            # 3. Display Answer
            answer = data["answer"]
            latency = data["latency_seconds"]
            sources = data["sources"]

            # Update the placeholder with the final answer
            message_placeholder.markdown(answer)
            
            # Add metadata below the answer
            st.caption(f"⏱️ Generated in {latency:.2f}s")

            with st.expander("📚 View Sources & Citations"):
                for i, source in enumerate(sources):
                    st.markdown(f"""
                    <div class="source-box">
                        <b>Source {i+1}</b> (Page {source['page']}, Score: {source['score']:.2f})<br>
                        <i>"...{source['preview']}..."</i>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Save assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": answer})

        except requests.exceptions.ConnectionError:
            message_placeholder.error("❌ Connection Error: Could not connect to backend API. Is Docker running?")
        except Exception as e:
            message_placeholder.error(f"An error occurred: {e}")