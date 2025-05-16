import os
import sys
import argparse
import streamlit as st

from typing import Tuple, List

from langchain_core.runnables import (
    RunnableBranch, RunnableLambda, RunnableParallel, RunnablePassthrough
)
from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.tools import DuckDuckGoSearchResults
from langchain_community.utilities import SearxSearchWrapper
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Neo4jVector, Chroma, AstraDB
from langchain_astradb import AstraDBVectorStore
from helper_functions import encode_pdf
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from exa_py import Exa
import json

sys.path.append(os.path.abspath(
    os.path.join(os.getcwd(), '..')))  # Add the parent directory to the path since we work with notebooks

# Load environment variables from a .env file
load_dotenv()
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["NEO4J_URI"] = st.secrets["NEO4J_URI"]
os.environ["NEO4J_USERNAME"] = st.secrets["NEO4J_USERNAME"]
os.environ["NEO4J_PASSWORD"] = st.secrets["NEO4J_PASSWORD"]
os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]

exa = Exa(api_key="c8f98386-429b-4c92-8580-bfdd2099c256")


index_name = "tacia"
pc = Pinecone()
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # Depends on the embedding size you're using (1536 for text-embedding-ada-002)
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(index_name)

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
vectorstore_dir = "chroma_db"

class RetrievalEvaluatorInput(BaseModel):
    """
    Model for capturing the relevance score of a document to a query.
    """
    relevance_score: float = Field(..., description="Relevance score between 0 and 1, "
                                                    "indicating the document's relevance to the query.")


class QueryRewriterInput(BaseModel):
    """
    Model for capturing a rewritten query suitable for web search.
    """
    query: str = Field(..., description="The query rewritten for better web search results.")


class KnowledgeRefinementInput(BaseModel):
    """
    Model for extracting key points from a document.
    """
    key_points: str = Field(..., description="Key information extracted from the document in bullet-point form.")


class CRAG:
    """
    A class to handle the CRAG process for document retrieval, evaluation, and knowledge refinement.
    """

    def __init__(self, model="gpt-4o-mini", max_tokens=1000, temperature=0.7, lower_threshold=0.3,
                 upper_threshold=0.7):
        """
        Initializes the CRAG Retriever by encoding the PDF document and creating the necessary models and search tools.

        Args:
            path (str): Path to the PDF file to encode.
            model (str): The language model to use for the CRAG process.
            max_tokens (int): Maximum tokens to use in LLM responses (default: 1000).
            temperature (float): The temperature to use for LLM responses (default: 0).
            lower_threshold (float): Lower threshold for document evaluation scores (default: 0.3).
            upper_threshold (float): Upper threshold for document evaluation scores (default: 0.7).
        """
        print("\n--- Initializing CRAG Process ---")

        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold

        # Encode the PDF document into a vector store
        # self.vectorstore = Neo4jVector.from_existing_graph(
        #     OpenAIEmbeddings(),
        #     search_type="hybrid",
        #     node_label="Document",
        #     text_node_properties=["text"],
        #     embedding_node_property="embedding"
        # )
        # self.vector_store_astra = AstraDB(
        #     embedding=OpenAIEmbeddings(),
        #     collection_name="enforcea_tax", 
        #     api_endpoint="https://4db977cb-ea15-4f3d-b94f-32d5823e8d0b-us-east-2.apps.astra.datastax.com",
        #     token="AstraCS:iZFFdgOnZljUUBPikZYPbLbO:5be8829f4cf7a3cecd40ba40bd19d7ddf9158a3f5b85cface8b415a1b575d27a"
        # )

        self.pinecone_vector_store = PineconeVectorStore(
            index=index,             # This is the Pinecone index handle
            embedding=OpenAIEmbeddings(),     # OpenAI embeddings
            text_key="text"          # Make sure you use the same key used when storing text
        )

        # Initialize OpenAI language model
        self.llm = ChatOpenAI(model=model, max_tokens=max_tokens, temperature=temperature)
        # chat_groq = ChatGroq(temperature=0.9, groq_api_key=GROQ_API_KEY, model_name="deepseek-r1-distill-llama-70b")
        # self.llm = chat_groq.with_structured_output(include_raw=True)
        # Initialize search tool
        self.search = DuckDuckGoSearchResults()
        self.search_searx = SearxSearchWrapper(searx_host="http://127.0.0.1:8888")

    @staticmethod
    def retrieve_documents(question, vstore, k=20):
        """
        Retrieve documents based on a query using a FAISS index.

        Args:
            query (str): The query string to search for.
            faiss_index (FAISS): The FAISS index used for similarity search.
            k (int): The number of top documents to retrieve. Defaults to 3.

        Returns:
            List[str]: A list of the retrieved document contents.
        """
        unstructured_docs = set()
        results = vstore.similarity_search(question, k=k)
        print(results)
        for doc in results:
            unstructured_docs.add(doc.page_content.strip())

        # Step 2: Convert to list
        unstructured_docs = list(unstructured_docs)

        # Step 3: Rerank using a cross-encoder (optional)
        pairs = [(question, doc) for doc in unstructured_docs]
        scores = reranker.predict(pairs)  # assumes reranker is defined globally

        # Step 4: Sort by descending score and take top 5
        reranked_docs = [doc for _, doc in sorted(zip(scores, unstructured_docs), key=lambda x: -x[0])][:10]

        # Step 5: Return List[str]
        print(f"RERANKED DOCS: {reranked_docs}")
        return reranked_docs

    def evaluate_documents(self, query, documents):
        return [self.retrieval_evaluator(query, doc) for doc in documents]

    def retrieval_evaluator(self, query, document):
        prompt = PromptTemplate(
            input_variables=["query", "document"],
            template="Dalam skala dari 0 hingga 1, seberapa relevan dokumen berikut terhadap pertanyaan?"
                     "Pertanyaan: {query}\nDokumen: {document}\nSkor relevansi:"
        )
        chain = prompt | self.llm.with_structured_output(RetrievalEvaluatorInput)
        input_variables = {"query": query, "document": document}
        result = chain.invoke(input_variables).relevance_score
        return result

    def knowledge_refinement(self, document):
        prompt = PromptTemplate(
            input_variables=["document"],
            template="Ekstrak informasi penting dari dokumen berikut dalam bentuk poin-poin:"
                     "\n{document}\nPoin-poin penting:"
        )
        chain = prompt | self.llm.with_structured_output(KnowledgeRefinementInput)
        input_variables = {"document": document}
        result = chain.invoke(input_variables).key_points
        return [point.strip() for point in result.split('\n') if point.strip()]

    def rewrite_query(self, query):
        prompt = PromptTemplate(
            input_variables=["query"],
            template="Tulis ulang kueri berikut agar lebih sesuai untuk pencarian web:\n{query}\nKueri yang telah ditulis ulang:"
        )
        chain = prompt | self.llm.with_structured_output(QueryRewriterInput)
        input_variables = {"query": query}
        return chain.invoke(input_variables).query.strip()

    @staticmethod
    def parse_search_results(results_string):
        try:
            results = json.loads(results_string)
            return [(result.get('title', 'Untitled'), result.get('link', '')) for result in results]
        except json.JSONDecodeError:
            print("Error parsing search results. Returning empty list.")
            return []

    def perform_web_search(self, query):
        rewritten_query = self.rewrite_query(query)
        print("REWRITTEN QUERY: ", rewritten_query)
        web_results = self.search.run(rewritten_query)
        web_knowledge = self.knowledge_refinement(web_results)
        sources = self.parse_search_results(web_results)
        return web_knowledge, sources

    def generate_response(self, query, knowledge, sources):
        response_prompt = PromptTemplate(
            input_variables=["query", "knowledge", "sources"],
            template="Berdasarkan pengetahuan berikut, jawablah pertanyaan."
                    "Sertakan Referensi beserta tautannya (jika tersedia) di akhir jawaban Anda:"
                    "\nPertanyaan: {query}\nPengetahuan: {knowledge}\nReferensi: {sources}\nJawaban:"
        )
        input_variables = {
            "query": query,
            "knowledge": knowledge,
            "sources": "\n".join([f"{title}: {link}" if link else title for title, link in sources])
        }
        response_chain = response_prompt | self.llm
        return response_chain.invoke(input_variables).content

    def run(self, input_data):
        CONDENSE_QUESTION_PROMPT = PromptTemplate(
            input_variables=["chat_history", "question"],
            template="""
        Riwayat percakapan:
        {chat_history}

        Pertanyaan saat ini:
        {question}

        Reformulasikan pertanyaan agar berdiri sendiri, berdasarkan riwayat percakapan di atas.
        """,
        )

        def _format_chat_history(chat_history):
            return "\n".join([f"User: {q}\nAI: {a}" for q, a in chat_history])

        preprocess_chain = RunnableBranch(
            (
                RunnableLambda(lambda x: bool(x.get("chat_history"))),
                RunnablePassthrough.assign(
                    chat_history=lambda x: _format_chat_history(x["chat_history"])
                )
                | CONDENSE_QUESTION_PROMPT
                | self.llm
                | StrOutputParser()
            ),
            # Else: return original question
            RunnableLambda(lambda x: x["question"]),
        )
        processed_query = preprocess_chain.invoke(input_data)
        print(f"REFORMULATED MEMORY: {processed_query}")
        final_response = self.run_context(processed_query)
        return final_response



    def run_context(self, query):
        print(f"\nProcessing query: {query}")

        # Retrieve and evaluate documents
        retrieved_docs = self.retrieve_documents(query, self.pinecone_vector_store)
        eval_scores = self.evaluate_documents(query, retrieved_docs)

        print(f"\nRetrieved {len(retrieved_docs)} documents")
        print(f"Evaluation scores: {eval_scores}")

        # Determine action based on evaluation scores
        max_score = max(eval_scores)
        sources = []

        if max_score > self.upper_threshold:
            print("\nAction: Correct - Using retrieved document")
            best_doc = retrieved_docs[eval_scores.index(max_score)]
            final_knowledge = retrieved_docs
            sources.append(("Retrieved document", ""))
        elif max_score < self.lower_threshold:
            print("\nAction: Incorrect - Performing web search")
            final_knowledge, sources = self.perform_web_search(query)
        else:
            print("\nAction: Ambiguous - Combining retrieved document and web search")
            best_doc = retrieved_docs[eval_scores.index(max_score)]
            retrieved_knowledge = self.knowledge_refinement(retrieved_docs)
            web_knowledge, web_sources = self.perform_web_search(query)
            final_knowledge = "\n".join(retrieved_knowledge + web_knowledge)
            sources = [("Retrieved document", "")] + web_sources

        print("\nFinal knowledge:")
        print(final_knowledge)

        print("\nSources:")
        for title, link in sources:
            print(f"{title}: {link}" if link else title)

        print("\nGenerating response...")
        response = self.generate_response(query, final_knowledge, sources)
        print("\nResponse generated")
        return response


# Function to validate command line inputs
def validate_args(args):
    if args.max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer.")
    if args.temperature < 0 or args.temperature > 1:
        raise ValueError("temperature must be between 0 and 1.")
    return args


# Function to parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="CRAG Process for Document Retrieval and Query Answering.")
    parser.add_argument("--path", type=str, default="../data/Understanding_Climate_Change.pdf",
                        help="Path to the PDF file to encode.")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="Language model to use (default: gpt-4o-mini).")
    parser.add_argument("--max_tokens", type=int, default=1000,
                        help="Maximum tokens to use in LLM responses (default: 1000).")
    parser.add_argument("--temperature", type=float, default=0,
                        help="Temperature to use for LLM responses (default: 0).")
    parser.add_argument("--query", type=str, default="What are the main causes of climate change?",
                        help="Query to test the CRAG process.")
    parser.add_argument("--lower_threshold", type=float, default=0.3,
                        help="Lower threshold for score evaluation (default: 0.3).")
    parser.add_argument("--upper_threshold", type=float, default=0.7,
                        help="Upper threshold for score evaluation (default: 0.7).")

    return validate_args(parser.parse_args())


# Main function to handle argument parsing and call the CRAG class
def main(args):
    # Initialize the CRAG process
    crag = CRAG(
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        lower_threshold=args.lower_threshold,
        upper_threshold=args.upper_threshold
    )

    # Process the query
    response = crag.run(args.query)
    print(f"Query: {args.query}")
    print(f"Answer: {response}")


if __name__ == '__main__':
    main(parse_args())
