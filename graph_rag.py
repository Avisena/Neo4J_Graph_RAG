import os
import streamlit as st
from typing import List

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import TokenTextSplitter

from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import LLMChain

# Load environment variables
load_dotenv(override=True)
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["NEO4J_URI"] = st.secrets["NEO4J_URI"]
os.environ["NEO4J_USERNAME"] = st.secrets["NEO4J_USERNAME"]
os.environ["NEO4J_PASSWORD"] = st.secrets["NEO4J_PASSWORD"]

graph = Neo4jGraph()
llm = ChatOpenAI(temperature=0.2, model_name="gpt-4o-mini")

def preprocess_documents(pdf_path: str = "file (77).pdf"):
    loader = PDFPlumberLoader(pdf_path)
    raw_documents = loader.load()
    text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
    documents = text_splitter.split_documents(raw_documents)

    llm_transformer = LLMGraphTransformer(llm=ChatOpenAI(temperature=0, model_name="gpt-4o-mini"))
    graph_documents = llm_transformer.convert_to_graph_documents(documents)
    graph.add_graph_documents(graph_documents, baseEntityLabel=True, include_source=True)

    Neo4jVector.from_existing_graph(
        OpenAIEmbeddings(),
        search_type="hybrid",
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding"
    )

# Create vector index
vector_index = Neo4jVector.from_existing_graph(
    OpenAIEmbeddings(),
    search_type="hybrid",
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding"
)

# Create FULLTEXT index
graph.query("CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")

def generate_full_text_query(input: str) -> str:
    words = [el for el in remove_lucene_chars(input).split() if el]
    return ' '.join([f"{word}~2 AND" for word in words[:-1]] + [f"{words[-1]}~2"]).strip()

def structured_retriever(question: str) -> str:
    result = ""
    response = graph.query(
    """
    CALL db.index.fulltext.queryNodes('entity', $query, {limit:10})
    YIELD node,score
    CALL {
      WITH node
      MATCH (node)-[r:!MENTIONS]->(neighbor)
      RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
      UNION ALL
      WITH node
      MATCH (node)<-[r:!MENTIONS]-(neighbor)
      RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
    }
    RETURN output LIMIT 50
    """,
    {"query": generate_full_text_query(question)}  # question is a string
)
    result += "\n".join([el['output'] for el in response]) + "\n"
    return result.strip()

def retriever(question: str) -> List[Document]:
    structured_data = structured_retriever(question)
    unstructured_docs = vector_index.similarity_search(question)
    docs = [Document(page_content=f"Structured data:\n{structured_data}", metadata={"source": "graph"})]
    docs.extend(unstructured_docs)
    return docs

# Prompt templates
initial_prompt = PromptTemplate.from_template(
    """You are a legal assistant. Based on the following document, provide a concise answer to the question:
----------
{context}
----------
Question: {question}
Answer:"""
)

refine_prompt = PromptTemplate.from_template(
    """You are refining a legal analysis. Here's the existing summary:
----------
{existing_answer}
----------
And here's an additional context document:
----------
{context}
----------
Refine the answer with the new information (or say 'unchanged' if nothing new):"""
)

# Chains
initial_chain = LLMChain(llm=llm, prompt=initial_prompt)
refine_chain = LLMChain(llm=llm, prompt=refine_prompt)

def refine_documents_chain(input_documents: List[Document], question: str) -> str:
    context_docs = [doc.page_content for doc in input_documents]

    # Step 1: Initial response with first document
    current_answer = initial_chain.run({
        "context": context_docs[0],
        "question": question
    })

    # Step 2: Refine with each remaining document
    for context in context_docs[1:]:
        current_answer = refine_chain.run({
            "existing_answer": current_answer,
            "context": context
        })

    return current_answer

# Final RAG logic
def rag_refine(question: str) -> str:
    docs = retriever(question)
    return refine_documents_chain(docs, question)

# Run for testing
if __name__ == "__main__":
    answer = rag_refine(
        "Apa alasan diterbitkannya Peraturan Direktur Jenderal Pajak Nomor PER-28/PJ/2018 tentang Surat Keterangan Domisili bagi Subjek Pajak Dalam Negeri Indonesia dalam Rangka Penerapan Persetujuan Penghindaran Pajak Berganda?"
    )
    print(answer)
