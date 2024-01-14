import os
import openai
import pinecone

import streamlit as st
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone

from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
from langchain.chat_models import ChatOpenAI

# Page title
st.set_page_config(page_title='📄 Service Bot Q&A')
# st.image('logo2.png')
st.title('🤖 Corporate Internet Banking Service bot 📃')
st.warning('Service Bot is equipped with Bill Payments, Fixed Deposits, Home Tiles, Loan Service, Account Summery, CIB Pay Roll, User Information Manual, CIB General Services, Interbank Fund Transfer Manuals')
st.divider()


NEO4J_URL = os.getenv('NEO4J_URL')
NEO4J_username = os.getenv('NEO4J_USERNAME')
NEO4J_password = os.getenv('NEO4J_PASSWORD')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
openai_api_key = os.environ.get('OPENAI_API_KEY')


graph = Neo4jGraph(
    url=NEO4J_URL, username=NEO4J_username, password=NEO4J_password
)

chain = GraphCypherQAChain.from_llm(
    ChatOpenAI(temperature=0), graph=graph
)

saved_questions = ['How do I add a new favorite biller in Corporate Internet Banking?', 'How do I view my bill payment history in Corporate Internet Banking?', 'How Fixed Deposit Placement works ?', 'How to do a Fixed Deposit Withdrawal ?','I want all the information about the load repayments']
saved_queries = st.selectbox('Select a predefined question (Demo):', saved_questions)

query_text = st.text_input('Enter your question:', placeholder = 'How to change my password of the Peoples Bank Mobile app ?')
# Form input and query
result = []
 

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
index_name_pdf = 'service-bot'
index_pdf = pinecone.GRPCIndex(index_name_pdf)

# Initialize Embedding Model
embeddings = OpenAIEmbeddings()
docsearch_pdf = Pinecone.from_existing_index(index_name_pdf, embeddings)

qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model='gpt-4', temperature=0), chain_type="stuff", retriever=docsearch_pdf.as_retriever(search_type="similarity", search_kwargs={"k": 10}))
static_prompt = "Extract the information from the document and answer the question, and elaborate with the given answer, Questions are most likely to be ask from bank customers about the mobile and online banking, app configs. Question:"


if saved_queries is not None:
    answer_Vec = qa.run(static_prompt+saved_queries+ " Answer :")
    answer_KD = chain.run(static_prompt+saved_queries)

    with st.expander("Vector Base RAG Answer:"):
      result.append(answer_Vec)
      st.info(answer_Vec)

    with st.expander("Knowledge graph base RAG Answer:"):
      result.append(answer_KD)
      st.info(answer_KD)
elif query_text is not None:
    answer_Vec = qa.run(static_prompt+saved_queries+ " Answer :")
    answer_KD = chain.run(static_prompt+saved_queries)

    with st.expander("Vector Base RAG Answer:"):
      result.append(answer_Vec)
      st.info(answer_Vec)

    with st.expander("Knowledge graph base RAG Answer:"):
      result.append(answer_KD)
      st.info(answer_KD)
else:
    st.info('Please enter a question to get an answer')
