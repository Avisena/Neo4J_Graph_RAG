# -*- coding: utf-8 -*-
"""
Script version of enhancing_rag_with_graph.ipynb
"""

import os
from typing import Tuple, List

from langchain_core.runnables import (
    RunnableBranch, RunnableLambda, RunnableParallel, RunnablePassthrough
)
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.graphs import Neo4jGraph
from langchain.document_loaders import WikipediaLoader
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.vectorstores import Neo4jVector
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv(override=True)

# Env setup
os.environ["OPENAI_API_KEY"] = "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"
os.environ["NEO4J_URI"] = "neo4j+s://0f38f5f3.databases.neo4j.io"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "Qs8ejTd-eP0hgj3z5Exxy4SkTE2tUSi_3COOza4CeEs"

# Based on the class definition, here's how to properly initialize Neo4jGraph
graph = Neo4jGraph()
llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")


def preprocess_documents(pdf_path: str = "file (77).pdf"):
    # Load and split PDF
    loader = PDFPlumberLoader(pdf_path)
    raw_documents = loader.load()
    text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
    documents = text_splitter.split_documents(raw_documents)

    # Generate graph documents
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
    llm_transformer = LLMGraphTransformer(llm=llm)
    graph_documents = llm_transformer.convert_to_graph_documents(documents)
    graph.add_graph_documents(graph_documents, baseEntityLabel=True, include_source=True)

    # Create vector index
    Neo4jVector.from_existing_graph(
        OpenAIEmbeddings(),
        search_type="hybrid",
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding"
    )

# Only run QA chain
vector_index = Neo4jVector.from_existing_graph(
    OpenAIEmbeddings(),
    search_type="hybrid",
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding"
)


# Create fulltext index
graph.query("CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")

class Entities(BaseModel):
    names: List[str] = Field(..., description="All the person, organization, or business entities that appear in the text")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are extracting organization and person entities from the text."),
    ("human", "Use the given format to extract information from the following input: {question}"),
])

entity_chain = prompt | llm.with_structured_output(Entities)

def generate_full_text_query(input: str) -> str:
    words = [el for el in remove_lucene_chars(input).split() if el]
    return ' '.join([f"{word}~2 AND" for word in words[:-1]] + [f"{words[-1]}~2"]).strip()

def structured_retriever(question: str) -> str:
    result = ""
    entities = entity_chain.invoke({"question": question})
    for entity in entities.names:
        response = graph.query(
            """
            CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
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
            {"query": generate_full_text_query(entity)}
        )
        result += "\n".join([el['output'] for el in response]) + "\n"
    return result.strip()

def retriever(question: str):
    structured_data = structured_retriever(question)
    unstructured_data = [el.page_content for el in vector_index.similarity_search(question)]
    return f"Structured data:\n{structured_data}\nUnstructured data:\n#Document ".join(unstructured_data)

# Condense follow-up questions
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(
    """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question,
in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
)

def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer

_search_query = RunnableBranch(
    (RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(run_name="HasChatHistoryCheck"),
     RunnablePassthrough.assign(chat_history=lambda x: _format_chat_history(x["chat_history"]))
     | CONDENSE_QUESTION_PROMPT
     | ChatOpenAI(temperature=0)
     | StrOutputParser()),
    RunnableLambda(lambda x: x["question"]),
)

# Final RAG chain
template = """Answer the question based only on the following context:
{context}

Question: {question}
Use natural language and be concise.
Answer:"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
    RunnableParallel({
        "context": _search_query | retriever,
        "question": RunnablePassthrough(),
    })
    | prompt
    | llm
    | StrOutputParser()
)

# Test example
if __name__ == "__main__":
    print(chain.invoke({"question": "Apa alasan diterbitkannya Peraturan Direktur Jenderal Pajak Nomor PER-28/PJ/2018 tentang Surat Keterangan Domisili bagi Subjek Pajak Dalam Negeri Indonesia dalam Rangka Penerapan Persetujuan Penghindaran Pajak Berganda?"}))

    # print(chain.invoke({
    #     "question": "Tolong jelaskan lebih lanjut pasal-pasal yang mengatur lebih lanjut untuk poin 2 di atas",
    #     "chat_history": [("Apa kriteria agar seorang individu dianggap sebagai subjek pajak dalam negeri di Indonesia?",
    #                       "Seorang individu dianggap sebagai subjek pajak dalam negeri di Indonesia jika memenuhi salah satu dari kriteria berikut:\n\n1. Bertempat tinggal di Indonesia.\n2. Berada di Indonesia lebih dari 183 hari dalam jangka waktu 12 bulan.\n3. Dalam suatu Tahun Pajak, berada di Indonesia dan memiliki niat untuk bertempat tinggal di Indonesia.")],
    # }))
