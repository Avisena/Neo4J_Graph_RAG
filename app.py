import streamlit as st
from graph_rag import chain  # replace with your actual script name (without .py)

st.set_page_config(page_title="Tacia with Graph", layout="wide")
st.title("🧠 Tacia with Graph Knowledge")
st.markdown("### Dibuat dengan dokumen:")
st.markdown(
    "#### [PER-5/PJ/2023 - Pengembalian Kelebihan Pajak](https://drive.google.com/file/d/1Sss1-LiQtR5Snv29d7wrhSKasefO_H9H/view?usp=sharing)",
    unsafe_allow_html=True
)
st.markdown(
    "#### [PER-28/PJ/2018 - Surat Keterangan Domisili untuk Penerapan P3B](https://drive.google.com/file/d/1jMCYfcAiKeNsUztyWfUURn2DSZmWZh8V/view?usp=sharing)",
    unsafe_allow_html=True
)

st.markdown(
    "#### [ PERUBAHAN KETIGA ATAS UNDANG.UNDANG NOMOR 19 TAHUN 2OO3 TENTANG BADAN USAHA MILIK NEGARA](https://drive.google.com/file/d/11MSm1jdYLd59MvexWpPbedTgqB0Vz0IY/view?usp=sharing)",
    unsafe_allow_html=True
)

st.markdown(
    "#### [ PERATURAN MENTERI KEUANGAN REPUBLIK INDONESIA NOMOR 72 TAHUN 2023 ](https://drive.google.com/file/d/1yoGy4MfLCM9wnvaKI7qaDW4o611ROG0w/view?usp=sharing)",
    unsafe_allow_html=True
)

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input form
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_area("Your question", height=100)
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    # Prepare input for the chain
    chain_input = {"question": user_input}
    if st.session_state.chat_history:
        chain_input["chat_history"] = st.session_state.chat_history

    # Run the chain
    response = chain.invoke(chain_input)

    # Update chat history
    st.session_state.chat_history.append((user_input, response))

# Display conversation history
st.markdown("### 💬 Chat History")
for i, (user_q, bot_a) in enumerate(st.session_state.chat_history):
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_q)
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(bot_a)


# Option to clear chat
if st.button("🔄 Clear Chat"): 
    st.session_state.chat_history = []
    st.rerun() 
