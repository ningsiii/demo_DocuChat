from os.path import exists
from typing import List,Dict

from langchain.chains.llm import LLMChain
from langchain.chains.question_answering.map_reduce_prompt import system_template
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatTongyi
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain.prompts import(ChatPromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate)

#-------统一拿到chatllm---------
def get_llm(provider,api_key):
   if provider =="openai":
      return ChatOpenAI(model="gpt-3.5-turbo",
                         api_key=api_key,
                         temperature=1.0)

   elif provider =="dashscope":
       return ChatTongyi(model="qwen-plus-2025-04-28",
                        api_key=api_key)

   elif provider=="deepseek":
       return BaseChatOpenAI(model='deepseek-chat',
                             api_key=api_key)
   else:
       raise ValueError(f"不支持的provider：{provider}")
   # -------prompt模板---------
system_text="""
你是一名专业的技术工作者，精通多语言阅读与翻译。
请你 **通读整篇文章**，从“全局视角”总结 {{ num_points }} 条最重要的要点，每条以 “- ” 开头。
- 将下面的文章原文（无论什么语言）在内部转换为中文，以便理解；只用简体中文输出。
- 不要输出全文翻译，也不要逐句摘抄。
- 不得捏造不存在的事实或链接。
"""
human_text="""
文章原文：
<content>
{context}
</content>
"""

system_messages=SystemMessagePromptTemplate.from_template(system_text)
human_messages=HumanMessagePromptTemplate.from_template(human_text)
summary_prompt=ChatPromptTemplate.from_messages([system_messages,human_messages])
messages=summary_prompt.format_messages(
    num_points="num_points",
    context="context"
)
# -------构建chain---------
def build_summary_chain(provider,api_key)->LLMChain:
    llm=get_llm(provider,api_key)
    return LLMChain(llm=llm,prompt=summary_prompt,output_key="summary")

