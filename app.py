
import streamlit as st
from langchain.callbacks import get_openai_callback  # ✅ Add this
from crag_runnable import CRAG

st.set_page_config(page_title="Tacia", layout="wide")
st.title("🧠 Tacia Your Tax Assistant")
# st.markdown("### Dibuat dengan dokumen:")
# st.markdown(
#     "#### [PER-5/PJ/2023 - Pengembalian Kelebihan Pajak](https://drive.google.com/file/d/1Sss1-LiQtR5Snv29d7wrhSKasefO_H9H/view?usp=sharing)",
#     unsafe_allow_html=True,
# )
# st.markdown(
#     "#### [PER-28/PJ/2018 - Surat Keterangan Domisili untuk Penerapan P3B](https://drive.google.com/file/d/1jMCYfcAiKeNsUztyWfUURn2DSZmWZh8V/view?usp=sharing)",
#     unsafe_allow_html=True,
# )
# st.markdown(
#     "#### [ PERUBAHAN KETIGA ATAS UNDANG.UNDANG NOMOR 19 TAHUN 2OO3 TENTANG BADAN USAHA MILIK NEGARA](https://drive.google.com/file/d/11MSm1jdYLd59MvexWpPbedTgqB0Vz0IY/view?usp=sharing)",
#     unsafe_allow_html=True,
# )
# st.markdown(
#     "#### [ PERATURAN MENTERI KEUANGAN REPUBLIK INDONESIA NOMOR 72 TAHUN 2023 ](https://drive.google.com/file/d/1yoGy4MfLCM9wnvaKI7qaDW4o611ROG0w/view?usp=sharing)",
#     unsafe_allow_html=True,
# )

# ✅ Initialize chat history and token usage in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "prompt_tokens" not in st.session_state:
    st.session_state.prompt_tokens = 0
if "completion_tokens" not in st.session_state:
    st.session_state.completion_tokens = 0
if "total_cost_usd" not in st.session_state:
    st.session_state.total_cost_usd = 0.0
if "total_cost_rupiah" not in st.session_state:
    st.session_state.total_cost_rupiah = 0.0

# Input form
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_area("Your question", height=100)
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    # Prepare input for the chain
    chain_input = {"question": user_input}
    if st.session_state.chat_history:
        chain_input["chat_history"] = st.session_state.chat_history

    crag = CRAG()
    print(f"CHAIN INPUT: {chain_input}")

    # ✅ Track usage with OpenAI callback
    from langchain.callbacks import get_openai_callback
    with get_openai_callback() as cb:
        response = crag.run(chain_input)

        # ✅ Accumulate token and cost stats
        st.session_state.prompt_tokens += cb.prompt_tokens
        st.session_state.completion_tokens += cb.completion_tokens
        st.session_state.total_cost_usd += cb.total_cost
        st.session_state.total_cost_rupiah = st.session_state.total_cost_usd * 16500
    # Update chat history
    st.session_state.chat_history.append((user_input, response))

# ✅ Show usage stats
st.markdown("### 📊 Total Biaya Pemakaian")
st.markdown(f"- **Prompt Tokens**: {st.session_state.prompt_tokens}")
st.markdown(f"- **Completion Tokens**: {st.session_state.completion_tokens}")
st.markdown(f"- **Total Cost (USD)**: ${st.session_state.total_cost_usd:.6f}")
st.markdown(f"- **Total Cost (Rupiah kurs 16.500)**: Rp. {int(st.session_state.total_cost_rupiah)}")
st.markdown("---" * 50)
st.markdown("\n")

# Display conversation history
st.markdown("### 💬 Chat History")
for i, (user_q, bot_a) in enumerate(reversed(st.session_state.chat_history)):
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_q)
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(bot_a)

# Option to clear chat
if st.button("🔄 Clear Chat"): 
    st.session_state.chat_history = []
    st.session_state.prompt_tokens = 0
    st.session_state.completion_tokens = 0
    st.session_state.total_cost = 0.0
    st.rerun()
