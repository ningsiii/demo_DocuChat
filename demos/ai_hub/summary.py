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
       return ChatTongyi(model="qwen-plus",
                        api_key=api_key)

   elif provider=="deepseek":
       return BaseChatOpenAI(model='deepseek-chat',
                             api_key=api_key)
   else:
       raise ValueError(f"不支持的provider：{provider}")
   # -------prompt模板---------
system_text="""
  你是一名专业的技术工作者，精通多语言阅读与翻译。
任务流程：
1. 将下面的文章原文（无论什么语言）在内部转换为中文，以便理解；
2. **不要**输出翻译后的全文；只需用**简体中文**总结 {num_points} 条要点，每条用 “- ” 开头。
请严格只使用**简体中文**输出，不得捏造不存在的信息/链接。
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

