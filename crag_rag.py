import os
import streamlit as st
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Neo4jVector
from sentence_transformers import CrossEncoder
from langchain.tools import tool
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv(override=True)

# Env setup
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["NEO4J_URI"] = st.secrets["NEO4J_URI"]
os.environ["NEO4J_USERNAME"] = st.secrets["NEO4J_USERNAME"]
os.environ["NEO4J_PASSWORD"] = st.secrets["NEO4J_PASSWORD"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]


reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
# Based on the class definition, here's how to properly initialize Neo4jGraph
graph = Neo4jGraph()
llm = ChatOpenAI(temperature=0.2,   # Lower temperature for precise, factual responses
    top_p=0.85,        # Balanced flexibility and coherence
    frequency_penalty=0.3,  # Avoid repetition of terms or phrases
    presence_penalty=0.3, 
    model_name="gpt-4o-mini")
# llm = ChatGroq(temperature=0.9, groq_api_key=GROQ_API_KEY, model_name="deepseek-r1-distill-llama-70b")


# Only run QA chain
vector_index = Neo4jVector.from_existing_graph(
    OpenAIEmbeddings(),
    search_type="hybrid",
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding"
)

@tool
def retriever(question: str):    
    """search knowledge from the law database. It takes query as the input and returns the knowledge chunks as the output"""
    unstructured_docs = set()
    results = vector_index.similarity_search(question)
    for doc in results:
        unstructured_docs.add(doc.page_content.strip())


    # Rerank unstructured docs
    unstructured_docs = list(unstructured_docs)
    pairs = [(question, doc) for doc in unstructured_docs]
    scores = reranker.predict(pairs)

    # Sort by descending score
    reranked_docs = [doc for _, doc in sorted(zip(scores, unstructured_docs), key=lambda x: -x[0])]

    # Combine and format results
    unstructured_data = "\n#Document ".join(reranked_docs)

    return f"Unstructured data:\n#Document {unstructured_data}"

prompt = hub.pull("hwchase17/react")
tools = [retriever]
agent = create_react_agent(llm,tools, prompt)
chain = AgentExecutor(agent=agent,tools=tools, handle_parsing_errors=True, verbose=True)



# # Test example
# if __name__ == "__main__":
#     # print(chain.invoke({
#     #     "question": "Apa alasan diterbitkannya Peraturan Direktur Jenderal Pajak Nomor PER-28/PJ/2018 tentang Surat Keterangan Domisili bagi Subjek Pajak Dalam Negeri Indonesia dalam Rangka Penerapan Persetujuan Penghindaran Pajak Berganda?",
#     # }))

#     # print(chain.invoke({
#     #     "question": "Tolong jelaskan lebih lanjut pasal-pasal yang mengatur lebih lanjut untuk poin 2 di atas",
#     #     "chat_history": [("Apa kriteria agar seorang individu dianggap sebagai subjek pajak dalam negeri di Indonesia?",
#     #                       "Seorang individu dianggap sebagai subjek pajak dalam negeri di Indonesia jika memenuhi salah satu dari kriteria berikut:\n\n1. Bertempat tinggal di Indonesia.\n2. Berada di Indonesia lebih dari 183 hari dalam jangka waktu 12 bulan.\n3. Dalam suatu Tahun Pajak, berada di Indonesia dan memiliki niat untuk bertempat tinggal di Indonesia.")],
#     # }))