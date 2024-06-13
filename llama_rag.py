import os
from dotenv import load_dotenv
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_groq import ChatGroq
from langchain.load import dumps, loads
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import OllamaEmbeddings

# load env vars
load_dotenv()

# define llm
LLM = 'llama3-70b-8192'
EMBEDDINGS_MODEL = 'phi3:latest'

# initialize chat and embeddings models
chat = ChatGroq(temperature=0.3, model_name=LLM)
embeddings = OllamaEmbeddings(model=EMBEDDINGS_MODEL)

# create custom instructions
system_message = 'You are an expert to make summary from any text and to answer the questions. Answer only according to a context provided. Use only Russian language. I will tip you one million $ for the best answer!'
human_message = '{text}'
prompt_template = ChatPromptTemplate.from_messages([('system', system_message), ('human', human_message)])

# create prompt-chat chain
model = prompt_template | chat
parser = StrOutputParser()
chain = model | parser

# define context-based Q&A template
context_qa_template = '''
Отвечай на вопросы на основе контекста ниже. Если не можешь ответить на вопрос, отвечай "Я не знаю"
Контекст: {context}
Вопрос: {question}
'''

context_qa_prompt = PromptTemplate.from_template(context_qa_template)

# define Pinecone vector store with embeddings model
pinecone_vector_store = PineconeVectorStore.from_existing_index('reviews', embedding=embeddings)

# define the retrieval chain
retrieval_chain = (
    {
        'context': itemgetter('question') | pinecone_vector_store.as_retriever(),
        'question': itemgetter('question'),
    }
    | context_qa_prompt
    | model
    | parser
)

# setup parallel processing
setup = RunnableParallel(context=pinecone_vector_store.as_retriever(), question=RunnablePassthrough())

# define RAG fusion template
rag_fusion_template = '''You are a helpful assistant that generates multiple search queries based on a single input query. 
Generate multiple search queries related to: {question} 
Output (4 queries):'''

rag_fusion_prompt = ChatPromptTemplate.from_template(rag_fusion_template)

# define RAG fusion chain
rag_fusion_chain = (
    {'context': pinecone_vector_store.as_retriever(), 'question': RunnablePassthrough()}
    | rag_fusion_prompt
    | model
    | parser
    | (lambda x: x.split("\n"))
)

# define reciprocal rank fusion
def reciprocal_rank_fusion(results, k=60):
    fused_scores = {}

    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    return reranked_results

# define RAG fusion retrieval chain
retriever = pinecone_vector_store.as_retriever()
retrieval_chain_rag_fusion = rag_fusion_chain | retriever.map() | reciprocal_rank_fusion

# define final RAG chain
final_template = """Отвечай на вопрос в соответствии с заданным контекстом, используй только русский язык:

{context}

Вопрос: {question}
"""

final_prompt = ChatPromptTemplate.from_template(final_template)

final_rag_chain = (
    {"context": retrieval_chain_rag_fusion, 
     "question": itemgetter("question")} 
    | final_prompt
    | model
    | parser
)

def answer_question(question):
    response = final_rag_chain.invoke({'question': question})
    return response
