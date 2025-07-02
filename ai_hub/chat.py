from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatTongyi
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

DOC_ONLY_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
你是一名 **严谨可靠的中文文档问答助手**。  
**所有回复必须使用简体中文**，不得出现任何其他语言。  
只能依赖「文档内容」作答；如果文档中没有答案，直接回复：  
> 对不起，我在文档中未找到相关信息。  
禁止使用预训练知识、外部网址或任何编造的内容。

文档内容：
{context}

用户问题：
{question}

回答：
"""
)

def qa_agent(provider, api_key, contexts, question, memory=None):
    # ———— 1. 使用外部传入的 memory，如无则新建 ————
    if memory is None:
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

    # ———— 2. 文档准备 ————
    documents = [
        Document(page_content=t, metadata={})
        for t in contexts
    ]
#---------选嵌入模型embedding-------
    if provider=="openai":
        embeddings=OpenAIEmbeddings(api_key=api_key,model="text-embedding-3-large")
    elif provider=="dashscope":
        embeddings=DashScopeEmbeddings(dashscope_api_key=api_key,model="text-embedding-v2")
    elif provider=="deepseek":
        embeddings = BaseChatOpenAI(api_key=api_key,model='deepseek-chat')
    else:
        raise ValueError(f"不支持的嵌入提供商: {provider}")
        # ———— 4. 构建向量库 & Retriever ————
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # ———— 5. 选择 LLM 模型 ————
    if provider == "openai":
        model = ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key, temperature=1.0)
    elif provider == "dashscope":
        model = ChatTongyi(model="qwen-plus-2025-04-28", api_key=api_key)
    elif provider == "deepseek":
        model = BaseChatOpenAI(model="deepseek-chat", api_key=api_key, temperature=1.3)
    else:
        raise ValueError(f"不支持的provider：{provider}")

    # ———— 6. 构建 RAG 对话链 ————
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        memory=memory,
        output_key="answer",
        combine_docs_chain_kwargs={
        "prompt": DOC_ONLY_PROMPT
    }
    )

    # ———— 7. 执行问答，Chain 会自动读取并更新 memory ————
    response = qa_chain({"question": question})

    # ———— 8. 返回回答 & 最新 memory ————
    return {
        "answer": response["answer"],
        "memory": memory
    }