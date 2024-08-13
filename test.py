from langchain_core.tools import tool
import os


# os.environ["http_proxy"] = f"http://127.0.0.1:7890"
# os.environ["https_proxy"] = f"http://127.0.0.1:7890"
@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b


tools = [add, multiply]

from langchain_openai import ChatOpenAI

# llm = ChatOpenAI(model="gpt-4o",
#                  api_key="sk-usqU5KXzBCpOP2T881D9A21838Fe47Ed8f64Ce753e192aE3",
#                  base_url="https://api.gpts.vin/v1")

llm = ChatOpenAI(model="gpt-4o-2024-05-13",
                 api_key="sk-GfWFJo25weXH3eUS0f08C3A842A241F29cE8BfA171643a00",
                 base_url="https://gtapi.xiaoerchaoren.com:8932/v1")
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)
print(llm_with_tools.invoke("Please call the first tool two times").tool_calls)
