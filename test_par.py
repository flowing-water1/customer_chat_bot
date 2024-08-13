import gradio as gr
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain.tools import tool
from typing import TypedDict, Annotated, Any
from datetime import datetime
import os
import uuid
import sqlite3
import time

class State(TypedDict):
    messages: Annotated[list[Any], lambda x: x]
    user_info: str

class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            result = self.runnable.invoke(state)
            if not result.tool_calls and (not result.content or isinstance(result.content, list) and not result.content[0].get("text")):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}

def fetch_user_info(state: State):
    return {"user_info": fetch_user_flight_information.invoke({})}

def display_messages(event, _printed, chat):
    for s in event['messages']:
        if s.id not in _printed:
            if isinstance(s, HumanMessage):
                chat.append(("human", s.content))
            elif isinstance(s, AIMessage):
                if not s.content and 'tool_calls' in s.additional_kwargs:
                    try:
                        tool_calls = s.additional_kwargs['tool_calls']
                        tool_names = [tool_call['function']['name'] for tool_call in tool_calls if 'function' in tool_call and 'name' in tool_call['function']]
                        chat.append(("ai", f"I will use the following tools: {', '.join(tool_names)}"))
                    except Exception as e:
                        chat.append(("error", f"Error processing AIMessage tool calls: {str(e)}"))
                else:
                    chat.append(("ai", s.content))
            elif isinstance(s, ToolMessage):
                try:
                    if 'HTTPError' in s.content:
                        chat.append(("error", f"Tool调用失败：{s.content}，请检查你的API key是否正确"))
                    else:
                        tool_info = {"Tool": s.name, "Content": s.content}
                        chat.append(("tool", tool_info))
                except Exception as e:
                    chat.append(("error", f"处理 ToolMessage 时出错: {str(e)}"))
            _printed.add(s.id)

def run_chat(model_name, api_key, api_base, tutorial_questions):
    llm = ChatOpenAI(model=model_name, api_key=api_key, base_url=api_base)
    assistant_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful customer support assistant for Swiss Airlines. Use the provided tools to search for flights, company policies, and other information to assist the user's queries. When searching, be persistent. Expand your query bounds if the first search returns no results. If a search comes up empty, expand your search before giving up.\n\nCurrent user:\n\n{user_info}\n\nCurrent time: {time}."),
        ("placeholder", "{messages}"),
    ]).partial(time=datetime.now())

    part_2_tools = [
        TavilySearchResults(max_results=1),
        fetch_user_flight_information,
        search_flights,
        lookup_policy,
        update_ticket_to_new_flight,
        cancel_ticket,
        search_car_rentals,
        book_car_rental,
        update_car_rental,
        cancel_car_rental,
        search_hotels,
        book_hotel,
        update_hotel,
        cancel_hotel,
        search_trip_recommendations,
        book_excursion,
        update_excursion,
        cancel_excursion,
    ]
    part_2_assistant_runnable = assistant_prompt | llm.bind_tools(part_2_tools)

    builder = StateGraph(State)
    builder.add_node("fetch_user_info", fetch_user_info)
    builder.add_edge(START, "fetch_user_info")
    builder.add_node("assistant", Assistant(part_2_assistant_runnable))
    builder.add_node("tools", create_tool_node_with_fallback(part_2_tools))
    builder.add_edge("fetch_user_info", "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    memory = SqliteSaver.from_conn_string(":memory:")
    part_2_graph = builder.compile(checkpointer=memory, interrupt_before=["tools"])

    _printed = set()
    chat = []
    try:
        for question in tutorial_questions:
            events = part_2_graph.stream({"messages": ("user", question)}, config, stream_mode="values")
            for event in events:
                display_messages(event, _printed, chat)
            snapshot = part_2_graph.get_state(config)
            while snapshot.next:
                user_input = gr.Textbox(label="Do you approve of the above actions? Type 'y' to continue; otherwise, explain your requested change.", visible=True).launch()
                if user_input is not None:
                    if user_input.strip() == "y":
                        result = part_2_graph.invoke(None, config)
                    else:
                        result = part_2_graph.invoke(
                            {"messages": [ToolMessage(tool_call_id=event["messages"][-1].tool_calls[0]["id"], content=f"API call denied by user. Reasoning: '{user_input}'. Continue assisting, accounting for the user's input.")]}
                        , config)
                snapshot = part_2_graph.get_state(config)
    except Exception as e:
        chat.append(("error", f"运行出错：{str(e)}"))

    return chat

def main_interface():
    with gr.Blocks() as demo:
        # 展示API Key输入部分
        openai_api_key = gr.Textbox(label="OpenAI API Key", placeholder="sk-...")
        zhipu_api_key = gr.Textbox(label="Zhipu API Key", placeholder="sk-...")
        langchain_api_key = gr.Textbox(label="LangChain API Key", placeholder="sk-...")
        tavily_api_key = gr.Textbox(label="Tavily API Key", placeholder="sk-...")

        # 构建运行库部分
        # 这部分原版是构建向量库的代码

        model_name = gr.Textbox(label="输入你要使用的模型:", value="gpt-4", placeholder="gpt-4")
        start_button = gr.Button("开始")
        chat_output = gr.Chatbot(label="Chat")

        def on_start(openai_api_key, zhipu_api_key, langchain_api_key, tavily_api_key, model_name):
            api_key = openai_api_key if openai_api_key else zhipu_api_key
            api_base = "YOUR_API_BASE"  # 根据实际情况设置api_base
            tutorial_questions = ["What is your flight policy?", "How can I book a flight?"]  # Add your questions here

            chat = run_chat(model_name, api_key, api_base, tutorial_questions)
            return chat

        start_button.click(on_start, inputs=[openai_api_key, zhipu_api_key, langchain_api_key, tavily_api_key, model_name], outputs=chat_output)

    return demo

if __name__ == "__main__":
    demo = main_interface()
    demo.launch()



def main_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Customer Chat bot (Including Flight, Car, Hotel, Etc.)")

        with gr.Row():
            build_button = gr.Button("构建向量库")
            status_text = gr.Textbox(label="状态", interactive=False)

        with gr.Column(visible=False) as api_keys_section:
            with gr.Tab("OpenAI"):
                openai_api_key = gr.Textbox(label="OpenAI API Key", type="password", value=session_state["openai_api_key"])
                openai_api_base = gr.Textbox(label="OpenAI API Base", value="https://api.gpts.vin/v1")
                gr.Markdown("[获取OpenAI API key](https://platform.openai.com/account/api-keys)")
                gr.Markdown("[OpenAI API文档](https://platform.openai.com/docs/api-reference/introduction)")
                gr.Markdown("要用直连原版的API话，要开VPN，端口设置为7890。用中转的不用开VPN，已测试过中转的跟直连的效果一样。")

            with gr.Tab("智谱"):
                zhipu_api_key = gr.Textbox(label="智谱AI的API Key", type="password", value=session_state["zhipu_api_key"])
                zhipu_api_base = gr.Textbox(label="智谱AI的API Base")
                gr.Markdown("[获取智谱AI的API key](https://www.zhipuai.cn/)")
                gr.Markdown("国产的LLM模型基本上无法完成任务，但是可能可以通过修改prompt完成任务")

            with gr.Row():
                langchain_api_key = gr.Textbox(label="Langchain API Key", type="password", value=session_state["langchain_api_key"])
                tavily_api_key = gr.Textbox(label="Tavily API Key", type="password", value=session_state["tavily_api_key"])

            gr.Button("设置 API 密钥").click(
                set_api_keys,
                inputs=[langchain_api_key, tavily_api_key, openai_api_key, zhipu_api_key],
                outputs=[gr.Text(), gr.Text(), gr.Text(), gr.Text()]
            )

        build_button.click(
            initialize_vector_store,
            inputs=[],
            outputs=[status_text, api_keys_section]
        )

    return demo

# 启动Gradio应用
if __name__ == "__main__":
    demo = main_interface()
    demo.launch(share=True)

