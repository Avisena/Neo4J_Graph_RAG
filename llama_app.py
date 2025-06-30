import streamlit as st
import requests
import time
import json

# Configuration
RUNPOD_API_KEY = st.secrets["RUNPOD_API_KEY"]
RUNPOD_ENDPOINT_ID = st.secrets["RUNPOD_ENDPOINT_ID"]

# RunPod URLs
RUN_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/run"
STATUS_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/status/"

# Streamlit UI setup
st.set_page_config(page_title="Tacia Chatbot", layout="wide")
st.title("🧾 Tacia Llama3.2-1B EnforceA")

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User prompt input
prompt = st.chat_input("Tanyakan sesuatu tentang pajak...")

if prompt:
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Prepare RunPod payload
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RUNPOD_API_KEY}"
    }

    data = {
        "input": {
            "question": prompt,
            "max_new_tokens": 512,
            "eos_bias": 4.134,
            "temperature": 0.4,
            "repetition_penalty": 0.9
        }
    }

    # Send job to RunPod
    with st.spinner("📤 Mengirim pertanyaan ke model..."):
        try:
            response = requests.post(RUN_URL, headers=headers, json=data)
            response.raise_for_status()
            job_id = response.json().get("id")
            if not job_id:
                raise Exception("Gagal mendapatkan job ID.")
        except Exception as e:
            st.error(f"❌ Gagal mengirim permintaan: {e}")
            st.stop()

    # Polling for result
    output = None
    status = ""
    spinner_text = st.empty()  # For dynamic message

    start_time = time.time()
    for _ in range(150):  # Max 150 polls (5 mins)
        elapsed = time.time() - start_time

        try:
            status_res = requests.get(f"{STATUS_URL}{job_id}", headers=headers)
            status_data = status_res.json()
            status = status_data.get("status")

            # Update spinner based on status
            if status == "IN_QUEUE" and elapsed < 3:
                spinner_text.markdown("Processing your question...")
            elif status == "IN_QUEUE" and elapsed > 3:
                spinner_text.markdown("🧊 Cold starting... May take 2 minutes ⏳")
            elif status == "IN_PROGRESS":
                spinner_text.markdown("🤖 Tacia is typing...")
            elif status == "COMPLETED":
                spinner_text.markdown("")
                output = status_data.get("output")
                break
            elif status in ("FAILED", "CANCELLED"):
                output = f"❌ Job gagal diproses. Status: {status}"
                break

        except Exception as e:
            output = f"❌ Gagal mengambil status: {e}"
            break

        time.sleep(2)

    if not output:
        output = "⏳ Waktu tunggu habis. Silakan coba lagi."


    # Parse and display output
    try:
        parsed_output = json.loads(output) if isinstance(output, str) else output
        answer_text = parsed_output.get("answer", "❌ Tidak ada jawaban.")
        context_list = parsed_output.get("context", [])
    except Exception as e:
        answer_text = "❌ Gagal membaca output."
        context_list = []
        st.error(f"Parsing error: {e}")

    # Build full assistant markdown message
    assistant_md = f"**Jawaban:**\n\n{answer_text}"

    if context_list:
        assistant_md += "\n\n---\n\n### 📚 Sumber Hukum yang Digunakan\n"
        for item in context_list:
            name = item.get("name", "Sumber tidak diketahui")
            full_text = item.get("full_text", "")
            assistant_md += f"**{name}**\n\n> {full_text}\n\n---\n"

    # Display in chat
    with st.chat_message("assistant"):
        st.markdown(assistant_md)

    # Store complete message so it's preserved in chat history
    st.session_state.messages.append({
        "role": "assistant",
        "content": assistant_md
    })
