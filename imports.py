import os
import subprocess
import logging
from dotenv import find_dotenv, load_dotenv
import openai
import streamlit as st
from streamlit_chat import message
import pinecone
from pinecone import Pinecone, ServerlessSpec
from plotly.figure_factory import create_scatterplotmatrix as ff
import numpy as np
from langchain import PromptTemplate
from langchain import hub
from langchain.chains import ConversationalRetrievalChain
from langchain.agents import AgentExecutor, create_tool_calling_agent, create_openai_functions_agent
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.tools import PythonREPLTool

from langchain_pinecone import PineconeVectorStore
from cpp_langchain import CppSubprocessTool
from ns3Agent import *
from PIL import Image
import base64
import html
import requests
from bs4 import BeautifulSoup



