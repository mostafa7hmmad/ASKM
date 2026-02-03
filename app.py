import streamlit as st
import requests
import time

API_URL = "http://localhost:8000/ask"

st.set_page_config(page_title="Islamic RAG", layout="wide")
st.title("📚 Islamic RAG Assistant (Streaming Simulation)")

query = st.text_input("Ask your question:")

if st.button("Ask") and query:
    placeholder = st.empty()  # container to update answer
    sources_placeholder = st.empty()  # container for sources
    with st.spinner("Thinking..."):
        try:
            # إرسال السؤال للـ API
            response = requests.post(API_URL, json={"question": query}, timeout=120)

            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "")
                sources = data.get("sources", [])

                # تقسيم الإجابة إلى أسطر أو جمل لتحديث تدريجي
                chunks = [line for line in answer.split("\n") if line.strip() != ""]
                displayed_text = ""
                for chunk in chunks:
                    displayed_text += chunk + "\n\n"
                    placeholder.markdown(displayed_text)
                    time.sleep(0.2)  # Delay بسيط لمحاكاة Streaming

                # عرض المصادر بعد الانتهاء
                with sources_placeholder.expander("📌 Sources"):
                    for src in sources:
                        st.write(f"- **{src['source']}** (ID: {src['id']})")

            else:
                st.error(f"API Error: {response.text}")

        except Exception as e:
            st.error(f"Connection Error: {e}")
