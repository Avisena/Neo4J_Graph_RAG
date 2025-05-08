import os
import sys
import streamlit as st
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import ChatOpenAI
from sentence_transformers import CrossEncoder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.tools import DuckDuckGoSearchResults

# Original path append replaced for Colab compatibility
from helper_functions import *
from evalute_rag import *

# Load environment variables from a .env file
# Load environment variables from a .env file
load_dotenv(override=True)

# Env setup
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["NEO4J_URI"] = st.secrets["NEO4J_URI"]
os.environ["NEO4J_USERNAME"] = st.secrets["NEO4J_USERNAME"]
os.environ["NEO4J_PASSWORD"] = st.secrets["NEO4J_PASSWORD"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

graph = Neo4jGraph()
llm = ChatOpenAI(temperature=0.2,   # Lower temperature for precise, factual responses
    top_p=0.85,        # Balanced flexibility and coherence
    frequency_penalty=0.3,  # Avoid repetition of terms or phrases
    presence_penalty=0.3, 
    model_name="gpt-4o-mini")

vector_index = Neo4jVector.from_existing_graph(
    OpenAIEmbeddings(),
    search_type="hybrid",
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding"
)
search = DuckDuckGoSearchResults()

# Retrieval Evaluator
class RetrievalEvaluatorInput(BaseModel):
    relevance_score: float = Field(..., description="The relevance score of the document to the query. the score should be between 0 and 1.")
def retrieval_evaluator(query: str, document: str) -> float:
    prompt = PromptTemplate(
        input_variables=["query", "document"],
        template="On a scale from 0 to 1, how relevant is the following document to the query? Query: {query}\nDocument: {document}\nRelevance score:"
    )
    chain = prompt | llm.with_structured_output(RetrievalEvaluatorInput)
    input_variables = {"query": query, "document": document}
    result = chain.invoke(input_variables).relevance_score
    return result

# Knowledge Refinement
class KnowledgeRefinementInput(BaseModel):
    key_points: str = Field(..., description="The document to extract key information from.")
def knowledge_refinement(document: str) -> List[str]:
    prompt = PromptTemplate(
        input_variables=["document"],
        template="Extract the key information from the following document in bullet points:\n{document}\nKey points:"
    )
    chain = prompt | llm.with_structured_output(KnowledgeRefinementInput)
    input_variables = {"document": document}
    result = chain.invoke(input_variables).key_points
    return [point.strip() for point in result.split('\n') if point.strip()]

# Web Search Query Rewriter
class QueryRewriterInput(BaseModel):
    query: str = Field(..., description="The query to rewrite.")
def rewrite_query(query: str) -> str:
    prompt = PromptTemplate(
        input_variables=["query"],
        template="Rewrite the following query to make it more suitable for a web search:\n{query}\nRewritten query:"
    )
    chain = prompt | llm.with_structured_output(QueryRewriterInput)
    input_variables = {"query": query}
    return chain.invoke(input_variables).query.strip()



def parse_search_results(results_string: str):
    """
    Parse a JSON string of search results into a list of title-link tuples.

    Args:
        results_string (str): A JSON-formatted string containing search results.

    Returns:
        List[Tuple[str, str]]: A list of tuples, where each tuple contains the title and link of a search result.
                               If parsing fails, an empty list is returned.
    """
    try:
        # Attempt to parse the JSON string
        results = json.loads(results_string)
        # Extract and return the title and link from each result
        return [(result.get('title', 'Untitled'), result.get('link', '')) for result in results]
    except json.JSONDecodeError:
        # Handle JSON decoding errors by returning an empty list
        print("Error parsing search results. Returning empty list.")
        return []
    
def retrieve_documents(question: str) -> List[str]:
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
    return(unstructured_data)

def evaluate_documents(query: str, documents: List[str]) -> List[float]:
    """
    Evaluate the relevance of documents based on a query.

    Args:
        query (str): The query string.
        documents (List[str]): A list of document contents to evaluate.

    Returns:
        List[float]: A list of relevance scores for each document.
    """
    return [retrieval_evaluator(query, doc) for doc in documents]

def perform_web_search(query: str):
    """
    Perform a web search based on a query.

    Args:
        query (str): The query string to search for.

    Returns:
        Tuple[List[str], List[Tuple[str, str]]]: 
            - A list of refined knowledge obtained from the web search.
            - A list of tuples containing titles and links of the sources.
    """
    rewritten_query = rewrite_query(query)
    web_results = search.run(rewritten_query)
    web_knowledge = knowledge_refinement(web_results)
    sources = parse_search_results(web_results)
    return web_knowledge, sources

def generate_response(query: str, knowledge: str, sources):
    """
    Generate a response to a query using knowledge and sources.

    Args:
        query (str): The query string.
        knowledge (str): The refined knowledge to use in the response.
        sources (List[Tuple[str, str]]): A list of tuples containing titles and links of the sources.

    Returns:
        str: The generated response.
    """
    response_prompt = PromptTemplate(
        input_variables=["query", "knowledge", "sources"],
        template="Based on the following knowledge, answer the query. Include the sources with their links (if available) at the end of your answer:\nQuery: {query}\nKnowledge: {knowledge}\nSources: {sources}\nAnswer:"
    )
    input_variables = {
        "query": query,
        "knowledge": knowledge,
        "sources": "\n".join([f"{title}: {link}" if link else title for title, link in sources])
    }
    response_chain = response_prompt | llm
    return response_chain.invoke(input_variables).content



def crag_process(query: str) -> str:
    """
    Process a query by retrieving, evaluating, and using documents or performing a web search to generate a response.

    Args:
        query (str): The query string to process.
        faiss_index (FAISS): The FAISS index used for document retrieval.

    Returns:
        str: The generated response based on the query.
    """
    print(f"\nProcessing query: {query}")

    # Retrieve and evaluate documents
    retrieved_docs = retrieve_documents(query)
    eval_scores = evaluate_documents(query, retrieved_docs)
    
    print(f"\nRetrieved {len(retrieved_docs)} documents")
    print(f"Evaluation scores: {eval_scores}")

    # Determine action based on evaluation scores
    max_score = max(eval_scores)
    sources = []
    
    if max_score > 0.7:
        print("\nAction: Correct - Using retrieved document")
        best_doc = retrieved_docs[eval_scores.index(max_score)]
        final_knowledge = best_doc
        sources.append(("Retrieved document", ""))
    elif max_score < 0.3:
        print("\nAction: Incorrect - Performing web search")
        final_knowledge, sources = perform_web_search(query)
    else:
        print("\nAction: Ambiguous - Combining retrieved document and web search")
        best_doc = retrieved_docs[eval_scores.index(max_score)]
        # Refine the retrieved knowledge
        retrieved_knowledge = knowledge_refinement(best_doc)
        web_knowledge, web_sources = perform_web_search(query)
        final_knowledge = "\n".join(retrieved_knowledge + web_knowledge)
        sources = [("Retrieved document", "")] + web_sources

    print("\nFinal knowledge:")
    print(final_knowledge)
    
    print("\nSources:")
    for title, link in sources:
        print(f"{title}: {link}" if link else title)

    # Generate response
    print("\nGenerating response...")
    response = generate_response(query, final_knowledge, sources)

    print("\nResponse generated")
    return response
