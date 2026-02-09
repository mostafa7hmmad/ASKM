import streamlit as st
import requests
import json
import uuid

API_URL="https://coky-unsown-maryanna.ngrok-free.dev/ask"

st.set_page_config(page_title="IAM Dawah Assistant", layout="wide")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("⚙️ Controls")
    if st.button("Clear Chat & Memory"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

st.title("🌙 IAM Islamic Dawah Preacher")

# Display conversation history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# User Chat Input
if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_text = ""
        
        # --- LOADING ICON START ---
        with st.spinner("thinking..."):
            try:
                response = requests.post(
                    API_URL, 
                    json={"question": prompt, "session_id": st.session_state.session_id}, 
                    stream=True
                )

                if response.status_code == 200:
                    for line in response.iter_lines():
                        if line:
                            data = json.loads(line.decode())
                            if data["type"] == "token":
                                full_text += data["data"]
                                placeholder.markdown(full_text + "▌") # Streaming cursor
                            elif data["type"] == "sources":
                                with st.sidebar.expander("References Used", expanded=False):
                                    for s in data["data"]: st.write(f"- {s['source']}")
                else:
                    st.error("Server Error")
            except Exception as e:
                st.error(f"Connection Error: {e}")
        # --- LOADING ICON END ---

        placeholder.markdown(full_text)

        st.session_state.messages.append({"role": "assistant", "content": full_text})

