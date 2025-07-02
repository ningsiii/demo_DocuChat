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

☑️ 规则：
1. 先在内部把全文转为中文再理解；**最终回复只能出现简体中文**，不得包含任何英文字母、其他语言或拼音。
2. **通读整篇文章**，从全局视角提炼 {{ num_points }} 条最重要的要点，每条以 “- ” 开头。
3. 不要输出全文翻译，也不要逐句摘抄；不得捏造不存在的事实或链接。
4. 如果无法找到要点，直接回复“对不起，我无法总结本文。”（依旧只用简体中文）。

🚫 如有任何非中文字符（a-z / A-Z / other scripts）将视为违规。
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

