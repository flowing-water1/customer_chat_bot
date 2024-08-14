from langchain import hub
from langchain.agents import create_structured_chat_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI

import os

model = ChatOpenAI(model = "claude-3-sonnet-20240229",
                   openai_api_key = "sk-boFgxLeJNEtzKNgo1b0c6b9f35684cFc90Ed3bDaDe970a74",
                   openai_api_base = "https://api.claude-plus.top/v1")

model.invoke([HumanMessage(content="'君不见黄河之水天上来奔流到海不复回'，这句话的字数是多少？")])

class TextLengthTool(BaseTool):
    name = "文本字数计算工具"
    description = "当你被要求计算文本的字数时，使用此工具"

    def _run(self,text):
        return len(text)

tools = [TextLengthTool()]

prompt = hub.pull("hwchase17/structured-chat-agent")
print(prompt)

agent = create_structured_chat_agent(
    llm =model,
    tools = tools,
    prompt = prompt
)

memory = ConversationBufferMemory(
    memory_key = 'chat_history',
    return_messages = True
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent =agent, tools = tools,memory= memory,verbose = True, handle_parsing_errors = True
)

print(agent_executor.invoke({"input": "'君不见黄河之水天上来奔流到海不复回'，这句话的字数是多少？"}))

print(agent_executor.invoke({"input": "请你充当我的物理老师，告诉我什么是量子力学"}))