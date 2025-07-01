import streamlit as st
from demos.ai_hub.ingest    import ingest_pdf
from demos.ai_hub.summary import build_summary_chain, human_messages
from demos.ai_hub.chat  import qa_agent
from langchain.chains import LLMChain
from demos.ai_hub.text_utils import postprocess_docs
from demos.ai_hub.vector_store import build_vector_store, top_k_context
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage

# 0.会话状态
for key in ("docs","insights","outlines","report","chat_history","metrics"):
    if key not in st.session_state:
        if key=="metrics":
            st.session_state[key]={}
        elif key=="report":
            st.session_state[key]=""
        else:
            st.session_state[key]=[]

# 1. 侧边栏
st.sidebar.title("🔧 配置 & 监控")
# 1.1 APIkey+Provider
with st.sidebar:
    api_key=st.text_input("🔑请输入API密钥",type="password")
    #provider=st.selectable("嵌入服务商",["OpenAI","通义千问","DeepSeek"],index=None)
    provider_display = {
        "OpenAI": "openai",
        "通义千问": "dashscope",
        "DeepSeek": "deepseek",
    }
    provider_name = st.selectbox(
        "嵌入服务商",
        options=list(provider_display.keys()),
        index=None
    )
    provider = provider_display.get(provider_name)
# 1.2 监控（折叠）
with st.sidebar.expander("📊 监控",expanded=False):
    m=st.session_state.metrics
    st.metric("索引 Tokens",     m.get("ingest",  0))
    st.metric("摘要 Tokens",     m.get("research",0))
    st.metric("对话 Tokens",     m.get("chat",    0))
    st.markdown(f"**总计**：{sum(m.values())}Tokens")
st.sidebar.header("ℹ️ 使用小贴士 & 技能亮点")
st.sidebar.markdown(
    """
    欢迎体验本 Demo！  
    - **上传 PDF**，一键提取关键信息并生成中文摘要（支持外文文档）  
    - **检索增强对话**，用中文与文档内容自由交流（支持历史消息查看）

    技术栈：FAISS 向量检索 • LangChain RAG • 大模型集成 • Streamlit 快速部署  
    """
)
"## 向量检索+LLM：智能摘要与对话问答"


#2 数据采集
"##### 数据采集"
with st.expander("# 请上传需要读取的文档，文档语言不限",expanded=True):
    col_input,col_upload=st.columns([3,1])
    uploaded_file = st.file_uploader("上传 PDF", type="pdf", key="upload")
    prepare=st.button("准备数据")
    if prepare is True:
        if not uploaded_file:
            st.warning("请上传文件")
            st.stop()
        if not provider or not api_key:
            st.warning("请在侧边栏输入API KEY并选择服务商")
            st.stop()
    #2.2:布局——搜索输入、搜索按钮、文件上传
        with st.spinner("数据读取中，请稍等……"):
            docs = []
            if uploaded_file:
                pdf_chunks = ingest_pdf(
                    file_or_buffer=uploaded_file,
                    api_key=api_key,
                    provider=provider,
                )
                docs += [{"content": c, "source": "pdf"} for c in pdf_chunks]
                st.write("PDF 已建索引")
            docs_clean=postprocess_docs(docs,min_len=80)
            st.session_state.docs=docs_clean
            vec_store=build_vector_store(
                docs_clean,provider=provider,api_key=api_key
            )
            st.session_state.vec_store=vec_store
            st.session_state.docs_json = None
            st.success(f"已准备好{len(docs_clean)}条文档片段")
if st.session_state.get("docs"):
    with st.expander("📄 查看已准备好的文档片段", expanded=False):
         for item in st.session_state.docs:
            st.markdown(f"[{item['source'].upper()}] {item['content']}")
st.divider()
"##### 功能板块"
#3.2 主面板
tab1,tab2=st.tabs(["✍️智能摘要","💬文档智能问答"])
with tab1:
    docs = st.session_state.get("docs", [])
    if not docs:
        st.info("请先在『准备数据中』完成准备")
        st.stop()
    # ----摘要----
    num_points=st.slider(
        label="请选择要点数量",
        min_value=1,
        max_value=10,
        value=5
    )
    if st.button("生成智能摘要"):
        with st.spinner("数据读取中，请稍等……"):
            chain=build_summary_chain(provider,api_key)
            full_text="\n\n".join([d["content"]for d in docs])
            summary=chain.run(num_points=num_points,context=full_text)
            st.markdown(summary)
            st.session_state["summary"]=summary
    if "summary" in st.session_state:
        st.subheader("📑 摘要结果")
        st.markdown(st.session_state["summary"])
    else:
        st.info("摘要尚未生成，点击上方按钮试试。")
with tab2:
    if "memory"not in st.session_state:
        st.session_state["memory"]=ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history",
        )
    question=st.text_input("对文件内容进行提问",disabled=not uploaded_file)
    if st.button("提问"):
        if not docs:
            st.warning("请先准备好文档")
        elif not question:
            st.warning("请输入问题内容")
        else:
            with st.spinner("AI正在思考中，请稍等……"):
                contexts=[d["content"]for d in docs]
                response = qa_agent(
                provider = provider,
                api_key = api_key,
                contexts = contexts,
                question = question,
                memory=st.session_state["memory"]
                )
               # 提取并展示回答
                st.session_state["memory"] = response["memory"]
                answer = response["answer"]
                st.write("### 回答")
                st.write(answer)
    if st.session_state["memory"].chat_memory.messages:
        history = st.session_state["memory"].chat_memory.messages
        if history:
            with st.expander("历史消息"):
                for msg in history:
                    if isinstance(msg, HumanMessage):
                        st.markdown(f"**你**：{msg.content}")
                    elif isinstance(msg, AIMessage):
                        st.markdown(f"**AI**：{msg.content}")
                    else:
                        # 其它类型（如 SystemMessage）也一并显示
                        st.markdown(f"**{msg.__class__.__name__}**：{msg.content}")

















