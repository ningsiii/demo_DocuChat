from tempfile import NamedTemporaryFile
from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma   
#导入嵌入模型
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_openai.chat_models.base import BaseChatOpenAI

#from utils.project_config import API_KEY, EMBEDDING_PROVIDER

def ingest_pdf(
        file_or_buffer,
        index_path: str = "faiss_index",
        api_key=None,
        provider=None)->List[str]:
    #1.加载PDF
    if isinstance(file_or_buffer,(str,Path)):
        tmp_path=file_or_buffer
    else:
        tmp=NamedTemporaryFile(delete=False,suffix=".pdf")
        tmp.write(file_or_buffer.getbuffer())
        tmp_path=tmp.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    #2.切分
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n","\n","。","！","？","，","、",""]
       )
    texts=text_splitter.split_documents(docs)
    #3.选嵌入模型embedding
    if provider=="openai":
        embeddings=OpenAIEmbeddings(api_key=api_key,model="text-embedding-3-large")
    elif provider=="dashscope":
        embeddings=DashScopeEmbeddings(dashscope_api_key=api_key,model="text-embedding-v2")
    elif provider=="deepseek":
        embeddings = BaseChatOpenAI(api_key=api_key,model='deepseek-chat')
    else:
        raise ValueError(f"不支持的嵌入提供商: {provider}")
    # 4. 建索引 & 保存
    db = Chroma.from_documents(                      
        texts,
        embeddings,
        persist_directory=index_path                 
    )
    db.persist()                                     

    return [d.page_content for d in texts]






