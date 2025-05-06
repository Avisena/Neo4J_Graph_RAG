# -*- coding: utf-8 -*-
"""
Enhanced Legal Chatbot with Chain-of-Thought Legal Reasoning
"""

import os
import streamlit as st
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

# Load environment variables
load_dotenv(override=True)
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["NEO4J_URI"] = st.secrets["NEO4J_URI"]
os.environ["NEO4J_USERNAME"] = st.secrets["NEO4J_USERNAME"]
os.environ["NEO4J_PASSWORD"] = st.secrets["NEO4J_PASSWORD"]

graph = Neo4jGraph()
llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")

# Initialize Vector Store
vector_index = Neo4jVector.from_existing_graph(
    OpenAIEmbeddings(),
    search_type="hybrid",
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding"
)

graph.query("CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")

class Entities(BaseModel):
    names: List[str] = Field(..., description="All person, organization, or legal entities in the text")

prompt_entities = ChatPromptTemplate.from_messages([
    ("system", "Extract key legal entities (laws, sections, terms) from the question."),
    ("human", "Input: {question}"),
])

entity_chain = prompt_entities | llm.with_structured_output(Entities)

def generate_full_text_query(input: str) -> str:
    words = [el for el in remove_lucene_chars(input).split() if el]
    return ' '.join([f"{word}~2 AND" for word in words[:-1]] + [f"{words[-1]}~2"]).strip()

def structured_retriever(question: str) -> str:
    result = ""
    entities = entity_chain.invoke({"question": question})
    for entity in entities.names:
        response = graph.query(
            """
            CALL db.index.fulltext.queryNodes('entity', $query, {limit:10})
            YIELD node,score
            MATCH (node)-[*1..2]-(related)
            RETURN DISTINCT related.id AS output LIMIT 50
            """, {"query": generate_full_text_query(entity)}
        )
        result += "\n".join([el['output'] for el in response]) + "\n"
    return result.strip()

def retriever(question: str):
    structured_data = structured_retriever(question)
    unstructured_data = [el.page_content for el in vector_index.similarity_search(question)]
    return f"Structured data:\n{structured_data}\nUnstructured data:\n#Document ".join(unstructured_data)

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

# LEGAL CHAIN-OF-THOUGHT PROMPT
COT_PROMPT = ChatPromptTemplate.from_template("""
You are a legal assistant. Use structured legal reasoning to answer.

Step 1: Identify the key legal issue in the question.
Step 2: Identify the main law(s) or regulation(s) relevant to the issue.
Step 3: Retrieve related articles, sections, or contradictory regulations.
Step 4: Explore the broader legal implication, intention of the law, or comparisons.
Step 5: Provide the final interpretation clearly.

Use the following context to support your answer:
{context}

Question: {question}
Answer:
""")

# Final Chain
chain = (
    RunnableParallel({
        "context": _search_query | retriever,
        "question": RunnablePassthrough(),
    })
    | COT_PROMPT
    | llm
    | StrOutputParser()
)

# Test
if __name__ == "__main__":
    print(chain.invoke({
        "question": "Apa alasan diterbitkannya Peraturan Direktur Jenderal Pajak Nomor PER-28/PJ/2018 tentang Surat Keterangan Domisili bagi Subjek Pajak Dalam Negeri Indonesia dalam Rangka Penerapan Persetujuan Penghindaran Pajak Berganda?"
    }))
