import getpass
import os
import time

import pytz
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.utils.math import cosine_similarity
from langchain_openai import ChatOpenAI
from langgraph.constants import END, START
import streamlit as st

# 智谱的
# b27f36511b74905f6adfd4f1e7cf1f72.r90Sh9TOsQ3JcQaf
# https://open.bigmodel.cn/api/paas/v4/

# sk-9oYJRePIyAbz7wNj955dBbC98f0c44F8B91bF7779d38B131
# https://gtapi.xiaoerchaoren.com:8932
# https://gtapi.xiaoerchaoren.com:8932/v1
# https://gtapi.xiaoerchaoren.com:8932/v1/chat/completions

# https://tavily.com/
# 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
# https://smith.langchain.com/settings
# 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

# llm = ChatOpenAI(model="gpt-4o",
# api_key="sk-usqU5KXzBCpOP2T881D9A21838Fe47Ed8f64Ce753e192aE3",
# base_url="https://api.gpts.vin/v1")

# 看看st.sessionstate中还有什么技术可以用，直接用part2的代码调试调用一下
#https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state
#运行出错: There are multiple widgets with the same key='user_input'.

#To fix this, please make sure that the key argument is unique for each widget you create.

from streamlit_check import (

    validate_and_set_keys,
    set_api_keys,
    delete_environment_variable,
    set_environment_variable
)

# ————————————————————————————————————————————————————————————————————————————————————————————————————————————————————#
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["TAVILY_API_KEY"] = "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"


with st.sidebar:
    tab1, tab2 = st.tabs(["OPENAI", "智谱"])
    with tab1:
        openai_api_key = st.text_input("OpenAI API Key:","sk-usqU5KXzBCpOP2T881D9A21838Fe47Ed8f64Ce753e192aE3", type="password")
        openai_api_base = st.text_input("OpenAI API Base:","https://api.gpts.vin/v1")
        # openai_port = st.text_input("VPN的端口：")
        st.markdown("[获取OpenAI API key](https://platform.openai.com/account/api-keys)")
        st.markdown("[OpenAI API文档](https://platform.openai.com/docs/api-reference/introduction)")
        st.markdown("要用直连原版的API话，要开VPN，端口设置为7890。")
        st.markdown("用中转的不用开VPN，已测试过中转的跟直连的效果一样。")

    with tab2:
        zhipu_api_key = st.text_input("智谱AI的API Key:", type="password")
        zhipu_api_base = st.text_input("智谱AI的API Base:")
        st.markdown("[获取智谱AI的API key](https://www.zhipuai.cn/)")
        st.markdown("国产的LLM模型基本上无法完成任务，但是可能可以通过修改prompt完成任务")

st.title("Customer Chat bot\n(Incuding Flight, Car, Hotel, Etc.)")
st.divider()
with st.expander("**:red[使用前必看信息]**"):
    st.write('''**请注意**，Mutil-Agents是delicate的，因为LLM模型内部是:blue-background[黑箱].''')

    st.write(
        ''':blue-background[所以可能会有意想不到的bug。就算是同样的API，prompt和模型，可能结果也会有差距，这是不可避免的。]''')

    st.write('''***:red[模型推荐gpt-4/gpt-4o/gpt-4-1106-preview/claude-3]***.''')

    st.write('''实测这些模型可以完成任务（仍然会受prompt影响可能报错）''')

    st.write('''**:red[tavily是负责搜索的，langchain是监控流程和提供支持的，一定要填]**''')

st.divider()

# 初始化 session_state
if "tavily_api_key_set" not in st.session_state:
    st.session_state.tavily_api_key_set = bool(os.environ.get("TAVILY_API_KEY"))
    print("tavily_api_key_set:", st.session_state.tavily_api_key_set)

if "langchain_api_key_set" not in st.session_state:
    st.session_state.langchain_api_key_set = bool(os.environ.get("LANGCHAIN_API_KEY"))
    print("langchain_api_key_set:", st.session_state.langchain_api_key_set)

if "rerun" not in st.session_state:
    st.session_state.rerun = False


def set_environment_variable(key, value):
    os.environ[key] = value



# 检测并显示环境变量状态
column_check_tavily, column_check_langchain = st.columns([1, 1])

import os

# 初始化 session_state

if "langchain_api_key_set" not in st.session_state:
    st.session_state.langchain_api_key_set = bool(os.environ.get("LANGCHAIN_API_KEY"))

if "tavily_api_key_set" not in st.session_state:
    st.session_state.tavily_api_key_set = bool(os.environ.get("TAVILY_API_KEY"))

if "show_dialog" not in st.session_state:
    st.session_state.show_dialog = False

if "dialog_active" not in st.session_state:
    st.session_state.dialog_active = False

if "langchain_toast_shown" not in st.session_state:
    st.session_state.langchain_toast_shown = False

if "tavily_toast_shown" not in st.session_state:
    st.session_state.tavily_toast_shown = False

if "langchain_api_key" not in st.session_state:
    st.session_state.langchain_api_key = ""

if "tavily_api_key" not in st.session_state:
    st.session_state.tavily_api_key = ""

if "tavily_tool" not in st.session_state:
    st.session_state.tavily_tool = None

if "show_rerun_button" not in st.session_state:
    st.session_state.show_rerun_button = False

if "embeddings" not in st.session_state:
    st.session_state.embeddings = True

# 检测并显示环境变量状态
if not st.session_state.langchain_api_key_set and not os.environ.get(
        "LANGCHAIN_API_KEY"):
    langchain_info = st.info("未设置 LANGCHAIN_API_KEY")
elif os.environ.get(
        "LANGCHAIN_API_KEY") and not st.session_state.dialog_active and not st.session_state.langchain_toast_shown:
    st.toast("LANGCHAIN_API_KEY 已在环境变量中配置成功", icon='🌟')
    st.toast("LANGSMITH 配置成功", icon='🌟')
    st.session_state.langchain_toast_shown = True

if not st.session_state.tavily_api_key_set and not os.environ.get(
        "TAVILY_API_KEY"):
    tavily_info = st.info("未设置 TAVILY_API_KEY")
elif os.environ.get(
        "TAVILY_API_KEY") and not st.session_state.dialog_active and not st.session_state.tavily_toast_shown:
    st.toast("TAVILY_API_KEY 已在环境变量中成功", icon='🌟')
    st.toast(":red[注意：]请确保你的 TAVILY_API_KEY 有效，TAVILY_TOOL工具可以配置，但似乎只会在运行中报错，",
             icon='🚫')
    st.session_state.tavily_toast_shown = True

# 侧边栏按钮触发对话框
with st.sidebar:
    if st.button("设置 LANGCHAIN_API_KEY 和 TAVILY_API_KEY"):
        st.session_state.show_dialog = True
        set_api_keys()

    if st.button("删除环境变量中的 LANGCHAIN_API_KEY"):
        delete_environment_variable("LANGCHAIN_API_KEY")
    if st.button("删除环境变量中的 TAVILY_API_KEY"):
        delete_environment_variable("TAVILY_API_KEY")

import sys
import io


class OutputCatcher(io.StringIO):
    def __init__(self):
        super().__init__()
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        self.initial_check_done = False

    def __enter__(self):
        sys.stdout = self
        sys.stderr = self
        return self

    def __exit__(self, *args):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.seek(0)
        output = self.read()
        if "Failed to batch ingest runs: LangSmithError" in output:
            st.toast("再次提醒，你的Langchain_Api_Key是错误的，你可以忽略这个提醒，但是如果你要监控流程，务必使用正确的KEY",
                     icon="⚠️")
        self.close()

    def check_output(self):
        self.seek(0)
        output = self.read()
        if "Failed to batch ingest runs: LangSmithError" in output:
            st.toast("注意，你的Langchain_Api_Key是错误的，LangSmith无法配置，但是仍然可以运行", icon="⚠️")
        self.truncate(0)
        self.seek(0)


if not (openai_api_key or zhipu_api_key):
    st.info("请输入OpenAI API Key或者智谱AI的API key")

if openai_api_key and zhipu_api_key:
    st.info("有两个api，请选择一个使用即可")

if (openai_api_key or zhipu_api_key) and (
        not os.environ.get("TAVILY_API_KEY") and not os.environ.get("LANGCHAIN_API_KEY")):
    st.info("环境变量缺失:red[LANGCHAIN_API_KEY]和:red[TAVILY_API_KEY]")


def _set_env(var: str):
    if not os.environ.get(var):
        print(f"{var} not set, setting...")
        os.environ[var] = getpass.getpass(f"{var} not set, please enter: ")


import os
import shutil
import sqlite3
import pandas as pd
import requests

db_url = "https://storage.googleapis.com/benchmarks-artifacts/travel-db/travel2.sqlite"
local_file = "travel2.sqlite"

backup_file = "travel2.backup.sqlite"

import re
import numpy as np
from langchain_core.tools import tool

model_name = 'bge-large-zh-v1.5'
embedding_model = HuggingFaceEmbeddings(model_name=model_name)

# 初始化 session_state
if "vector_store_initialized" not in st.session_state:
    st.session_state.vector_store_initialized = False

if "db" not in st.session_state:
    st.session_state.db = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if not st.session_state.vector_store_initialized:
    with st.spinner("创建数据库和构成向量库中..."):
        overwrite = False
        if overwrite or not os.path.exists(local_file):
            response = requests.get(db_url)
            response.raise_for_status()
            with open(local_file, "wb") as f:
                f.write(response.content)

            shutil.copy(local_file, backup_file)

        conn = sqlite3.connect(local_file)
        cursor = conn.cursor()

        tables = pd.read_sql(
            "SELECT name FROM sqlite_master WHERE type='table';", conn
        ).name.tolist()
        tdf = {}
        for t in tables:
            tdf[t] = pd.read_sql(f"SELECT * FROM {t}", conn)

        example_time = pd.to_datetime(
            tdf["flights"]["actual_departure"].replace("\\N", pd.NaT)
        ).max()

        current_time = pd.to_datetime("now").tz_localize(example_time.tz)
        time_diff = current_time - example_time

        tdf["bookings"]["book_date"] = (
                pd.to_datetime(tdf["bookings"]["book_date"].replace("\\N", pd.NaT), utc=True)
                + time_diff
        )

        datetime_columns = [
            "scheduled_departure",
            "scheduled_arrival",
            "actual_departure",
            "actual_arrival",
        ]

        for column in datetime_columns:
            tdf["flights"][column] = pd.to_datetime(tdf["flights"][column].replace("\\N", pd.NaT))
            tdf["flights"][column] = tdf["flights"][column] + time_diff

        for table_name, df in tdf.items():
            df.to_sql(table_name, conn, if_exists="replace", index=False)

        del df
        del tdf
        conn.commit()
        conn.close()

        st.session_state.db = local_file = local_file

        with open('swiss_faq.md', 'r', encoding='utf-8') as file:
            faq_text = file.read()

        docs = [{"page_content": txt} for txt in re.split(r"(?=\n##)", faq_text)]


        class VectorStoreRetriever:
            def __init__(self, docs: list, vectors: list):
                self._arr = np.array(vectors)
                self._docs = docs

            @classmethod
            def from_docs(cls, docs):
                doc_texts = [doc["page_content"] for doc in docs]
                embeddings = embedding_model.embed_documents(doc_texts)
                vectors = embeddings
                return cls(docs, vectors)

            def query(self, query: str, k: int = 5) -> list[dict]:
                embed = embedding_model.embed_documents([query])[0]  # 修正此行，直接使用embed_documents方法
                scores = np.array(embed) @ self._arr.T
                top_k_idx = np.argpartition(scores, -k)[-k:]
                top_k_idx_sorted = top_k_idx[np.argsort(-scores[top_k_idx])]
                return [
                    {**self._docs[idx], "similarity": scores[idx]} for idx in top_k_idx_sorted
                ]


        st.session_state.vector_store = VectorStoreRetriever.from_docs(docs)

        st.session_state.vector_store_initialized = True
        st.toast('向量库构建完成', icon='🌟')

db = st.session_state.db
retriever = st.session_state.vector_store


@tool
def lookup_policy(query: str) -> str:
    """Consult the company policies to check whether certain options are permitted.
     Use this before making any flight changes performing other 'write' events."""
    docs = retriever.query(query, k=2)
    return "\n\n".join([doc["page_content"] for doc in docs])


# ——————————————————flight——————————————————————————————————
from typing import Optional
from langchain_core.runnables import ensure_config
from datetime import date, datetime


@tool
def fetch_user_flight_information() -> list[dict]:
    """Fetch all tickets for the user along with corresponding flight information and seat assignments.

    Returns:
        A list of dictionaries where each dictionary contains the ticket details,
        associated flight details, and the seat assignments for each ticket belonging to the user.
    """
    config = ensure_config()
    configuration = config.get("configurable", {})
    passenger_id = configuration.get("passenger_id", None)
    if not passenger_id:
        raise ValueError("No passenger ID configured.")

    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    query = """
    SELECT 
        t.ticket_no, t.book_ref,
        f.flight_id, f.flight_no, f.departure_airport, f.arrival_airport, f.scheduled_departure, f.scheduled_arrival,
        bp.seat_no, tf.fare_conditions
    FROM 
        tickets t
        JOIN ticket_flights tf ON t.ticket_no = tf.ticket_no
        JOIN flights f ON tf.flight_id = f.flight_id
        JOIN boarding_passes bp ON bp.ticket_no = t.ticket_no AND bp.flight_id = f.flight_id
    WHERE 
        t.passenger_id = ?
    """
    cursor.execute(query, (passenger_id,))
    rows = cursor.fetchall()
    column_names = [column[0] for column in cursor.description]
    results = [dict(zip(column_names, row)) for row in rows]

    cursor.close()
    conn.close()

    return results


@tool
def search_flights(
        departure_airport: Optional[str] = None,
        arrival_airport: Optional[str] = None,
        start_time: Optional[date | datetime] = None,
        end_time: Optional[date | datetime] = None,
        limit: int = 20,
) -> list[dict]:
    """Search for flights based on departure airport, arrival airport, and departure time range."""
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    query = "SELECT * FROM flights WHERE 1 = 1"
    params = []

    if departure_airport:
        query += " AND departure_airport = ?"
        params.append(departure_airport)

    if arrival_airport:
        query += " AND arrival_airport = ?"
        params.append(arrival_airport)

    if start_time:
        query += " AND scheduled_departure >= ?"
        params.append(start_time)

    if end_time:
        query += " AND scheduled_departure <= ?"
        params.append(end_time)

    query += " LIMIT ?"
    params.append(limit)

    # print("Query:", query)  # 调试输出查询语句
    # print("Params:", params)  # 调试输出参数

    cursor.execute(query, params)
    rows = cursor.fetchall()
    column_names = [column[0] for column in cursor.description]
    results = [dict(zip(column_names, row)) for row in rows]

    cursor.close()
    conn.close()

    return results


@tool
def update_ticket_to_new_flight(ticket_no: str, new_flight_id: int) -> str:
    """Update the user's ticket to a new valid flight."""
    config = ensure_config()
    configuration = config.get("configurable", {})
    passenger_id = configuration.get("passenger_id", None)
    if not passenger_id:
        raise ValueError("No passenger ID configured.")

    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT departure_airport, arrival_airport, scheduled_departure FROM flights WHERE flight_id = ?",
        (new_flight_id,),

    )
    new_flight = cursor.fetchone()
    if not new_flight:
        cursor.close()
        conn.close()
        return "Invalid new flight ID provided."
    column_names = [column[0] for column in cursor.description]
    new_flight_dict = dict(zip(column_names, new_flight))
    timezone = pytz.timezone("Etc/GMT-3")
    current_time = datetime.now(tz=timezone)
    departure_time = datetime.strptime(
        new_flight_dict["scheduled_departure"], "%Y-%m-%d %H:%M:%S.%f%z"
    )
    time_until = (departure_time - current_time).total_seconds()
    if time_until < (3 * 60 * 60):
        return f"Not permitted to reschedule to a flight that is less than 3 hours from the current time. Selected flight is at {departure_time}."

    cursor.execute(
        "SELECT flight_id FROM ticket_flights WHERE ticket_no = ?", (ticket_no,)

    )

    current_flight = cursor.fetchone()
    if not current_flight:
        cursor.close()
        conn.close()
        return "No existing ticket found for the given ticket number."

    cursor.execute(
        "SELECT * FROM tickets WHERE ticket_no = ? AND passenger_id = ?", (ticket_no, passenger_id)

    )
    current_ticket = cursor.fetchone()
    if not current_ticket:
        cursor.close()
        conn.close()
        return f"Current signed-in passenger with ID {passenger_id} not the owner of ticket {ticket_no}"

    # In a real application, you'd likely add additional checks here to enforce business logic,
    # like "does the new departure airport match the current ticket", etc.
    # While it's best to try to be *proactive* in 'type-hinting' policies to the LLM
    # it's inevitably going to get things wrong, so you **also** need to ensure your
    # API enforces valid behavior

    cursor.execute(
        "UPDATE ticket_flights SET flight_id = ? WHERE ticket_no = ?", (new_flight_id, ticket_no)
    )

    conn.commit()
    cursor.close()
    conn.close()
    return "Ticket successfully updated to new flight."


@tool
def cancel_ticket(ticket_no: str) -> str:
    """Cancel the user's ticket and remove it from the database."""
    config = ensure_config()
    configuration = config.get("configurable", {})
    passenger_id = configuration.get("passenger_id", None)
    if not passenger_id:
        raise ValueError("No passenger ID configured.")
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT flight_id FROM ticket_flights WHERE ticket_no = ?", (ticket_no,)
    )
    existing_ticket = cursor.fetchone()
    if not existing_ticket:
        cursor.close()
        conn.close()
        return "No existing ticket found for the given ticket number."

    # Check the signed-in user actually has this ticket
    cursor.execute(
        "SELECT flight_id FROM tickets WHEREticket_no = ? AND passenger_id = ?", (ticket_no, passenger_id)
    )
    current_ticket = cursor.fetchone()
    if not current_ticket:
        cursor.close()
        conn.close()
        return f"Current signed-in passenger with ID {passenger_id} not the owner of ticket {ticket_no}"

    cursor.execute("DELETE FROM ticket_flights WHERE ticket_no = ?", (ticket_no,))
    conn.commit()
    cursor.close()
    conn.close()
    return "Ticket successfully cancelled."


# ————————————————————————————————————————————————————————————————————————————————————

# ————————————————————————————————————Car Rental——————————————————————————————————————————


from typing import Optional, Union


@tool
def search_car_rentals(
        location: Optional[str] = None,
        name: Optional[str] = None,
        price_tier: Optional[str] = None,
        start_date: Optional[Union[datetime, date]] = None,
        end_date: Optional[Union[datetime, date]] = None,
) -> list[dict]:
    """
    Search for car rentals based on location, name, price tier, start date, and end date.

    Args:
        location (Optional[str]): The location of the car rental. Defaults to None.
        name (Optional[str]): The name of the car rental company. Defaults to None.
        price_tier (Optional[str]): The price tier of the car rental. Defaults to None.
        start_date (Optional[Union[datetime, date]]): The start date of the car rental. Defaults to None.
        end_date (Optional[Union[datetime, date]]): The end date of the car rental. Defaults to None.

    Returns:
        list[dict]: A list of car rental dictionaries matching the search criteria.
    """

    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    query = "SELECT * FROM car_rentals WHERE 1= 1"
    params = []

    if location:
        query += " AND location LIKE ?"
        params.append(f"%{location}%")

    if name:
        query += " AND name LIKE ?"
        params.append(f"%{name}%")
        # For our tutorial, we will let you match on any dates and price tier.
        # (since our toy dataset doesn't have much data)
    cursor.execute(query, params)
    results = cursor.fetchall()

    conn.close()
    return [
        dict(zip([column[0] for column in cursor.description], row)) for row in results
    ]


@tool
def book_car_rental(rental_id: int) -> str:
    """
    Book a car rental by its ID.

    Args:
        rental_id (int): The ID of the car rental to book.

    Returns:
        str: A message indicating whether the car rental was successfully booked or not.
    """
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute("UPDATE car_rentals SET bookded = 1 WHERE id =?", (rental_id,))
    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        return f"Car rental with ID {rental_id} successfully booked."
    else:
        conn.close()
        return f"Car rental with ID {rental_id} not found."


@tool
def update_car_rental(
        rental_id: int,
        start_date: Optional[Union[datetime, date]] = None,
        end_date: Optional[Union[datetime, date]] = None,
) -> str:
    """
    Update a car rental's start and end dates by its ID.

    Args:
        rental_id (int): The ID of the car rental to update.
        start_date (Optional[Union[datetime, date]]): The new start date of the car rental. Defaults to None.
        end_date (Optional[Union[datetime, date]]): The new end date of the car rental. Defaults to None.

    Returns:
        str: A message indicating whether the car rental was successfully updated or not.
    """
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    if start_date:
        cursor.execute(
            "UPDATE car_rentals SET start_date = ? WHERE id = ?",
            (start_date, rental_id),
        )
    if end_date:
        cursor.execute(
            "UPDATE car_rentals SET end_date = ? WHERE id = ?", (end_date, rental_id),
        )

    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        return f"Car rental with ID {rental_id} successfully updated."
    else:
        conn.close()
        return f"Car rental with ID {rental_id} not found."


@tool
def cancel_car_rental(rental_id: int) -> str:
    """
    Cancel a car rental by its ID.

    Args:
        rental_id (int): The ID of the car rental to cancel.

    Returns:
        str: A message indicating whether the car rental was successfully cancelled or not.
    """
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute("UPDATE car_rentals SET booked = 0 WHERE id =?", (rental_id,))
    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        return f"Car rental with ID {rental_id} successfully cancelled."
    else:
        conn.close()
        return f"Car rental with ID {rental_id} not found."


@tool
def search_hotels(
        location: Optional[str] = None,
        name: Optional[str] = None,
        price_tier: Optional[str] = None,
        checkin_date: Optional[Union[datetime, date]] = None,
        checkout_date: Optional[Union[datetime, date]] = None,
) -> list[dict]:
    """
    Search for hotels based on location, name, price tier, check-in date, and check-out date.

    Args:
        location (Optional[str]): The location of the hotel. Defaults to None.
        name (Optional[str]): The name of the hotel. Defaults to None.
        price_tier (Optional[str]): The price tier of the hotel. Defaults to None. Examples: Midscale, Upper Midscale, Upscale, Luxury
        checkin_date (Optional[Union[datetime, date]]): The check-in date of the hotel. Defaults to None.
        checkout_date (Optional[Union[datetime, date]]): The check-out date of the hotel. Defaults to None.

    Returns:
        list[dict]: A list of hotel dictionaries matching the search criteria.
    """
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    query = "SELECT * FROM hotels WHERE 1=1"
    params = []

    if location:
        query += " AND location LIKE ?"
        params.append(f"%{location}%")
    if name:
        query += " AND name LIKE ?"
        params.append(f"%{name}%")
    # For the sake of this tutorial, we will let you match on any dates and price tier.
    cursor.execute(query, params)
    results = cursor.fetchall()

    conn.close()

    return [
        dict(zip([column[0] for column in cursor.description], row)) for row in results
    ]


@tool
def book_hotel(hotel_id: int) -> str:
    """
    Book a hotel by its ID.

    Args:
        hotel_id (int): The ID of the hotel to book.

    Returns:
        str: A message indicating whether the hotel was successfully booked or not.
    """
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute("UPDATE hotels SET booked = 1 WHERE id = ?", (hotel_id,))
    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        return f"Hotel {hotel_id} successfully booked."
    else:
        conn.close()
        return f"No hotel found with ID {hotel_id}."


@tool
def update_hotel(
        hotel_id: int,
        checkin_date: Optional[Union[datetime, date]] = None,
        checkout_date: Optional[Union[datetime, date]] = None,
) -> str:
    """
    Update a hotel's check-in and check-out dates by its ID.

    Args:
        hotel_id (int): The ID of the hotel to update.
        checkin_date (Optional[Union[datetime, date]]): The new check-in date of the hotel. Defaults to None.
        checkout_date (Optional[Union[datetime, date]]): The new check-out date of the hotel. Defaults to None.

    Returns:
        str: A message indicating whether the hotel was successfully updated or not.
    """
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    if checkin_date:
        cursor.execute(
            "UPDATE hotels SET checkin_date = ? WHERE id = ?", (checkin_date, hotel_id)
        )
    if checkout_date:
        cursor.execute(
            "UPDATE hotels SET checkout_date = ? WHERE id = ?",
            (checkout_date, hotel_id),
        )

    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        return f"Hotel {hotel_id} successfully updated."
    else:
        conn.close()
        return f"No hotel found with ID {hotel_id}."


@tool
def cancel_hotel(hotel_id: int) -> str:
    """
    Cancel a hotel by its ID.

    Args:
        hotel_id (int): The ID of the hotel to cancel.

    Returns:
        str: A message indicating whether the hotel was successfully cancelled or not.
    """
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute("UPDATE hotels SET booked = 0 WHERE id = ?", (hotel_id,))
    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        return f"Hotel {hotel_id} successfully cancelled."
    else:
        conn.close()
        return f"No hotel found with ID {hotel_id}."


@tool
def search_trip_recommendations(
        location: Optional[str] = None,
        name: Optional[str] = None,
        keywords: Optional[str] = None,
) -> list[dict]:
    """
    Search for trip recommendations based on location, name, and keywords.

    Args:
        location (Optional[str]): The location of the trip recommendation. Defaults to None.
        name (Optional[str]): The name of the trip recommendation. Defaults to None.
        keywords (Optional[str]): The keywords associated with the trip recommendation. Defaults to None.

    Returns:
        list[dict]: A list of trip recommendation dictionaries matching the search criteria.
    """
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    query = "SELECT * FROM trip_recommendations WHERE 1=1"
    params = []

    if location:
        query += " AND location LIKE ?"
        params.append(f"%{location}%")
    if name:
        query += " AND name LIKE ?"
        params.append(f"%{name}%")
    if keywords:
        keyword_list = keywords.split(",")
        keyword_conditions = " OR ".join(["keywords LIKE ?" for _ in keyword_list])
        query += f" AND ({keyword_conditions})"
        params.extend([f"%{keyword.strip()}%" for keyword in keyword_list])

    cursor.execute(query, params)
    results = cursor.fetchall()

    conn.close()

    return [
        dict(zip([column[0] for column in cursor.description], row)) for row in results
    ]


@tool
def book_excursion(recommendation_id: int) -> str:
    """
    Book a excursion by its recommendation ID.

    Args:
        recommendation_id (int): The ID of the trip recommendation to book.

    Returns:
        str: A message indicating whether the trip recommendation was successfully booked or not.
    """
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute(
        "UPDATE trip_recommendations SET booked = 1 WHERE id = ?", (recommendation_id,)
    )
    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        return f"Trip recommendation {recommendation_id} successfully booked."
    else:
        conn.close()
        return f"No trip recommendation found with ID {recommendation_id}."


@tool
def update_excursion(recommendation_id: int, details: str) -> str:
    """
    Update a trip recommendation's details by its ID.

    Args:
        recommendation_id (int): The ID of the trip recommendation to update.
        details (str): The new details of the trip recommendation.

    Returns:
        str: A message indicating whether the trip recommendation was successfully updated or not.
    """
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute(
        "UPDATE trip_recommendations SET details = ? WHERE id = ?",
        (details, recommendation_id),
    )
    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        return f"Trip recommendation {recommendation_id} successfully updated."
    else:
        conn.close()
        return f"No trip recommendation found with ID {recommendation_id}."


@tool
def cancel_excursion(recommendation_id: int) -> str:
    """
    Cancel a trip recommendation by its ID.

    Args:
        recommendation_id (int): The ID of the trip recommendation to cancel.

    Returns:
        str: A message indicating whether the trip recommendation was successfully cancelled or not.
    """
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute(
        "UPDATE trip_recommendations SET booked = 0 WHERE id = ?", (recommendation_id,)
    )
    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        return f"Trip recommendation {recommendation_id} successfully cancelled."
    else:
        conn.close()
        return f"No trip recommendation found with ID {recommendation_id}."


from langchain_core.messages import ToolMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import ToolNode


def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in:", current_state[-1])
    message = event.get("message")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + "...(truncated)"
            print(msg_repr)
            _printed.add(message.id)


#
tutorial_questions = [
    "Hi there, what time is my flight?",
    "Am i allowed to update my flight to something sooner? I want to leave later today.",
    "Update my flight to sometime next week then",
    "The next available option is great",
    "what about lodging and transportation?",
    "Yeah i think i'd like an affordable hotel for my week-long stay (7 days). And I'll want to rent a car.",
    "OK could you place a reservation for your recommended hotel? It sounds nice.",
    "yes go ahead and book anything that's moderate expense and has availability.",
    "Now for a car, what are my options?",
    "Awesome let's just get the cheapest option. Go ahead and book for 7 days",
    "Cool so now what recommendations do you have on excursions?",
    "Are they available while I'm there?",
    "interesting - i like the museums, what options are there? ",
    "OK great pick one and book it for my second day there.",
]

# ———————————————————————————————————————————————Part 3——————————————————————————————————————————————
# from typing import Annotated
#
# from langchain_anthropic import ChatAnthropic
# from langchain_community.tools.tavily_search import TavilySearchResults
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import Runnable, RunnableConfig
# from typing_extensions import TypedDict
#
# from langgraph.graph.message import AnyMessage, add_messages
#
#
# class State(TypedDict):
#     messages: Annotated[list[AnyMessage], add_messages]
#     user_info: str
#
#
# class Assistant:
#     def __init__(self, runnable: Runnable):
#         self.runnable = runnable
#
#     def __call__(self, state: State, config: RunnableConfig):
#         while True:
#             result = self.runnable.invoke(state)
#             # If the LLM happens to return an empty response, we will re-prompt it
#             # for an actual response.
#             if not result.tool_calls and (
#                     not result.content
#                     or isinstance(result.content, list)
#                     and not result.content[0].get("text")
#             ):
#                 messages = state["messages"] + [("user", "Respond with a real output.")]
#                 state = {**state, "messages": messages}
#                 messages = state["messages"] + [("user", "Respond with a real output.")]
#                 state = {**state, "messages": messages}
#             else:
#                 break
#         return {"messages": result}
#
#
# # Haiku is faster and cheaper, but less accurate
# # llm = ChatAnthropic(model="claude-3-haiku-20240307")
# # llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=1)
# # You can update the LLMs, though you may need to update the prompts
# # from langchain_openai import ChatOpenAI
#
# # llm = ChatOpenAI(model="gpt-4-turbo-preview")
#
#
# from typing import Literal
#
# from langgraph.checkpoint.sqlite import SqliteSaver
# from langgraph.graph import StateGraph
# from langgraph.prebuilt import tools_condition
#
# builder = StateGraph(State)
#
#
# def user_info(state: State):
#     return {"user_info": fetch_user_flight_information.invoke({})}
#
#
# if (
#         openai_api_key and os.environ.get("LANGCHAIN_API_KEY") and os.environ.get(
#     "TAVILY_API_KEY") and not zhipu_api_key) or (
#         zhipu_api_key and os.environ.get("LANGCHAIN_API_KEY") and os.environ.get(
#     "TAVILY_API_KEY") and not openai_api_key):
#
#     model_name = st.text_input("输入你要使用的模型:", placeholder="gpt-4")
#
#     start_button = st.button("开始")
#     if start_button:
#         if openai_api_key:
#             api_key = openai_api_key
#             api_base = openai_api_base
#
#         else:
#             api_key = zhipu_api_key
#             api_base = zhipu_api_base
#
#         with st.spinner("创建LLM实例和助手..."):
#             # llm = ChatOpenAI(model="gpt-4o",
#             #                  api_key="sk-usqU5KXzBCpOP2T881D9A21838Fe47Ed8f64Ce753e192aE3",
#             #                  base_url="https://api.gpts.vin/v1")
#
#             llm = ChatOpenAI(model=model_name,
#                              api_key=api_key,
#                              base_url=api_base)
#
#             assistant_prompt = ChatPromptTemplate.from_messages(
#                 [
#                     (
#                         "system",
#                         "You are a helpful customer support assistant for Swiss Airlines. "
#                         " Use the provided tools to search for flights, company policies, and other information to assist the user's queries. "
#                         " When searching, be persistent. Expand your query bounds if the first search returns no results. "
#                         " If a search comes up empty, expand your search before giving up."
#                         "\n\nCurrent user:\n\n{user_info}\n"
#                         "\nCurrent time: {time}.",
#                     ),
#                     ("placeholder", "{messages}"),
#                 ]
#             ).partial(time=datetime.now())
#
#             # "Read"-only tools (such as retrievers) don't need a user confirmation to use
#             part_3_safe_tools = [
#                 TavilySearchResults(max_results=1),
#                 fetch_user_flight_information,
#                 search_flights,
#                 lookup_policy,
#                 search_car_rentals,
#                 search_hotels,
#                 search_trip_recommendations,
#             ]
#
#             # These tools all change the user's reservations.
#             # The user has the right to control what decisions are made
#             part_3_sensitive_tools = [
#                 update_ticket_to_new_flight,
#                 cancel_ticket,
#                 book_car_rental,
#                 update_car_rental,
#                 cancel_car_rental,
#                 book_hotel,
#                 update_hotel,
#                 cancel_hotel,
#                 book_excursion,
#                 update_excursion,
#                 cancel_excursion,
#             ]
#             sensitive_tool_names = {t.name for t in part_3_sensitive_tools}
#             # Our LLM doesn't have to know which nodes it has to route to. In its 'mind', it's just invoking functions.
#             part_3_assistant_runnable = assistant_prompt | llm.bind_tools(
#                 part_3_safe_tools + part_3_sensitive_tools
#             )
#
#         with st.spinner("创建Graph中，请稍等..."):
#             # NEW: The fetch_user_info node runs first, meaning our assistant can see the user's flight information without
#             # having to take an action
#             builder.add_node("fetch_user_info", user_info)
#             builder.add_edge(START, "fetch_user_info")
#             builder.add_node("assistant", Assistant(part_3_assistant_runnable))
#             builder.add_node("safe_tools", create_tool_node_with_fallback(part_3_safe_tools))
#             builder.add_node(
#                 "sensitive_tools", create_tool_node_with_fallback(part_3_sensitive_tools)
#             )
#             # Define logic
#             builder.add_edge("fetch_user_info", "assistant")
#
#
#             def route_tools(state: State) -> Literal["safe_tools", "sensitive_tools", "__end__"]:
#                 next_node = tools_condition(state)
#                 # If no tools are invoked, return to the user
#                 if next_node == END:
#                     return END
#                 ai_message = state["messages"][-1]
#                 # This assumes single tool calls. To handle parallel tool calling, you'd want to
#                 # use an ANY condition
#                 first_tool_call = ai_message.tool_calls[0]
#                 if first_tool_call["name"] in sensitive_tool_names:
#                     return "sensitive_tools"
#                 return "safe_tools"
#
#
#             builder.add_conditional_edges(
#                 "assistant",
#                 route_tools,
#             )
#             builder.add_edge("safe_tools", "assistant")
#             builder.add_edge("sensitive_tools", "assistant")
#
#             memory = SqliteSaver.from_conn_string(":memory:")
#             part_3_graph = builder.compile(
#                 checkpointer=memory,
#                 # NEW: The graph will always halt before executing the "tools" node.
#                 # The user can approve or reject (or even alter the request) before
#                 # the assistant continues
#                 interrupt_before=["sensitive_tools"],
#             )
#
#         st.toast('创建Graph成功啦~', icon='🌟')
#
#         import shutil
#         import uuid
#
#         # Update with the backup file so we can restart from the original place in each section
#         shutil.copy(backup_file, db)
#         thread_id = str(uuid.uuid4())
#
#         config = {
#             "configurable": {
#                 # The passenger_id is used in our flight tools to
#                 # fetch the user's flight information
#                 "passenger_id": "3442 587242",
#                 # Checkpoints are accessed by thread_id
#                 "thread_id": thread_id,
#             }
#         }
#
#         tutorial_questions = [
#             "Hi there, what time is my flight?",
#             "Am i allowed to update my flight to something sooner? I want to leave later today.",
#             "Update my flight to sometime next week then",
#             "The next available option is great",
#             "what about lodging and transportation?",
#             "Yeah i think i'd like an affordable hotel for my week-long stay (7 days). And I'll want to rent a car.",
#             "OK could you place a reservation for your recommended hotel? It sounds nice.",
#             "yes go ahead and book anything that's moderate expense and has availability.",
#             "Now for a car, what are my options?",
#             "Awesome let's just get the cheapest option. Go ahead and book for 7 days",
#             "Cool so now what recommendations do you have on excursions?",
#             "Are they available while I'm there?",
#             "interesting - i like the museums, what options are there? ",
#             "OK great pick one and book it for my second day there.",
#         ]
#
#         _printed = set()
#
#
#         # Function to display AI and Tool messages
#         # Function to display AI and Tool messages
#         def display_messages(event, _printed):
#             for s in event['messages']:
#                 # st.write(event)
#                 if s.id not in _printed:
#                     if isinstance(s, HumanMessage):
#                         st.chat_message("human").write(s.content)
#                     elif isinstance(s, AIMessage):
#                         if not s.content and 'tool_calls' in s.additional_kwargs:
#                             try:
#                                 tool_calls = s.additional_kwargs['tool_calls']
#                                 tool_names = [tool_call['function']['name'] for tool_call in tool_calls if
#                                               'function' in tool_call and 'name' in tool_call['function']]
#                                 st.chat_message("ai").write(f"I will use the following tools: {', '.join(tool_names)}")
#                             except Exception as e:
#                                 st.error(f"Error processing AIMessage tool calls: {str(e)}")
#                         else:
#                             st.chat_message("ai").write(s.content)
#                     elif isinstance(s, ToolMessage):
#                         try:
#                             if 'HTTPError' in s.content:
#                                 st.toast(f"Tool调用失败：{s.content}，请检查你的API key是否正确")
#                                 st.error(f"Tool调用失败：{s.content}，请检查你的API key是否正确")
#                                 time.sleep(2)
#                                 st.stop()
#                             else:
#                                 tool_info = {
#                                     "Tool": s.name,
#                                     "Content": s.content
#                                 }
#                                 st.chat_message("tool", avatar="tools.png").write(tool_info)
#                         except Exception as e:
#                             st.error(f"处理 ToolMessage 时出错: {str(e)}")
#                     _printed.add(s.id)
#
#
#         if "input_state" not in st.session_state:
#             st.session_state.input_state = None
#             st.session_state.input_approved = False
#
#         try:
#             with OutputCatcher() as output_catcher:
#
#                 # We can reuse the tutorial questions from part 1 to see how it does.
#                 for question in tutorial_questions:
#
#                     events = part_3_graph.stream(
#                         {"messages": ("user", question)}, config, stream_mode="values"
#                     )
#                     for event in events:
#
#                         display_messages(event, _printed)
#
#                     snapshot = part_3_graph.get_state(config)
#                     while snapshot.next:
#                         # We have an interrupt! The agent is trying to use a tool, and the user can approve or deny it
#
#                         # 检查是否已经获取了用户输入
#                         if not st.session_state.input_approved:
#                             user_input = st.chat_input(
#                                 "Do you approve of the above actions? Type 'y' to continue; otherwise, explain your requested changes.\n\n",
#                                 key="user_input"
#                             )
#
#                             # 如果用户输入了内容，更新状态
#                             if user_input:
#                                 st.session_state.input_state = user_input
#                                 st.session_state.input_approved = True
#
#                         # 如果用户已经输入，处理输入并继续循环
#                         if st.session_state.input_approved:
#                             user_input = st.session_state.input_state
#                             if user_input.strip() == "y":
#                                 result = part_3_graph.invoke(None, config)
#                             else:
#                                 result = part_3_graph.invoke(
#                                     {
#                                         "messages": [
#                                             ToolMessage(
#                                                 tool_call_id=event["messages"][-1].tool_calls[0]["id"],
#                                                 content=f"API call denied by user. Reasoning: '{user_input}'. Continue assisting, accounting for the user's input.",
#                                             )
#                                         ]
#                                     },
#                                     config,
#                                 )
#                             # 重置输入状态以便下一个循环使用
#                             st.session_state.input_approved = False
#                             st.session_state.input_state = None
#                             snapshot = part_3_graph.get_state(config)
#
#         except Exception as e:
#             st.toast(f"运行出错: {str(e)}", icon='🚫')
#

#————————————————————————Part2————————————-
from typing import Annotated

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from typing_extensions import TypedDict

from langgraph.graph.message import AnyMessage, add_messages


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_info: str


class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            result = self.runnable.invoke(state)
            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}


# Haiku is faster and cheaper, but less accurate
# llm = ChatAnthropic(model="claude-3-haiku-20240307")

# You could also use OpenAI or another model, though you will likely have
# to adapt the prompts
# from langchain_openai import ChatOpenAI

# llm = ChatOpenAI(model="gpt-4-turbo-preview")
if (
        openai_api_key and os.environ.get("LANGCHAIN_API_KEY") and os.environ.get(
    "TAVILY_API_KEY") and not zhipu_api_key) or (
        zhipu_api_key and os.environ.get("LANGCHAIN_API_KEY") and os.environ.get(
    "TAVILY_API_KEY") and not openai_api_key):

    model_name = st.text_input("输入你要使用的模型:","gpt-4o", placeholder="gpt-4")

    start_button = st.button("开始")
    if start_button:
        if openai_api_key:
            api_key = openai_api_key
            api_base = openai_api_base

        else:
            api_key = zhipu_api_key
            api_base = zhipu_api_base

        with st.spinner("创建LLM实例和助手..."):
            # llm = ChatOpenAI(model="gpt-4o",
            #                  api_key="sk-usqU5KXzBCpOP2T881D9A21838Fe47Ed8f64Ce753e192aE3",
            #                  base_url="https://api.gpts.vin/v1")

            llm = ChatOpenAI(model=model_name,
                             api_key=api_key,
                             base_url=api_base)
        assistant_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful customer support assistant for Swiss Airlines. "
                    " Use the provided tools to search for flights, company policies, and other information to assist the user's queries. "
                    " When searching, be persistent. Expand your query bounds if the first search returns no results. "
                    " If a search comes up empty, expand your search before giving up."
                    "\n\nCurrent user:\n\n{user_info}\n"
                    "\nCurrent time: {time}.",
                ),
                ("placeholder", "{messages}"),
            ]
        ).partial(time=datetime.now())

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


        from langgraph.checkpoint.sqlite import SqliteSaver
        from langgraph.graph import StateGraph
        from langgraph.prebuilt import tools_condition

        builder = StateGraph(State)


        def user_info(state: State):
            return {"user_info": fetch_user_flight_information.invoke({})}


        with st.spinner("创建Graph中，请稍等..."):
            # NEW: The fetch_user_info node runs first, meaning our assistant can see the user's flight information without
            # having to take an action
            builder.add_node("fetch_user_info", user_info)
            builder.add_edge(START, "fetch_user_info")
            builder.add_node("assistant", Assistant(part_2_assistant_runnable))
            builder.add_node("tools", create_tool_node_with_fallback(part_2_tools))
            builder.add_edge("fetch_user_info", "assistant")
            builder.add_conditional_edges(
                "assistant",
                tools_condition,
            )
            builder.add_edge("tools", "assistant")

            memory = SqliteSaver.from_conn_string(":memory:")
            part_2_graph = builder.compile(
                checkpointer=memory,
                # NEW: The graph will always halt before executing the "tools" node.
                # The user can approve or reject (or even alter the request) before
                # the assistant continues
                interrupt_before=["tools"],
            )
        st.toast('创建Graph成功啦~', icon='🌟')


        import shutil
        import uuid

        # Update with the backup file so we can restart from the original place in each section
        shutil.copy(backup_file, db)
        thread_id = str(uuid.uuid4())

        def set_state_if_absent(key, value):
            if key not in st.session_state:
                st.session_state[key] = value

        set_state_if_absent(key="thread_id", value=str(uuid.uuid4()))
        config = {
            "configurable": {
                # The passenger_id is used in our flight tools to
                # fetch the user's flight information
                "passenger_id": "3442 587242",
                # Checkpoints are accessed by thread_id
                "thread_id": st.session_state.thread_id,
            }
        }




        _printed = set()
        def display_messages(event, _printed):
            for s in event['messages']:
                # st.write(event)
                if s.id not in _printed:
                    if isinstance(s, HumanMessage):
                        st.chat_message("human").write(s.content)
                    elif isinstance(s, AIMessage):
                        if not s.content and 'tool_calls' in s.additional_kwargs:
                            try:
                                tool_calls = s.additional_kwargs['tool_calls']
                                tool_names = [tool_call['function']['name'] for tool_call in tool_calls if
                                              'function' in tool_call and 'name' in tool_call['function']]
                                st.chat_message("ai").write(f"I will use the following tools: {', '.join(tool_names)}")
                            except Exception as e:
                                st.error(f"Error processing AIMessage tool calls: {str(e)}")
                        else:
                            st.chat_message("ai").write(s.content)
                    elif isinstance(s, ToolMessage):
                        try:
                            if 'HTTPError' in s.content:
                                st.toast(f"Tool调用失败：{s.content}，请检查你的API key是否正确")
                                st.error(f"Tool调用失败：{s.content}，请检查你的API key是否正确")
                                time.sleep(2)
                                st.stop()
                            else:
                                tool_info = {
                                    "Tool": s.name,
                                    "Content": s.content
                                }
                                st.chat_message("tool", avatar="tools.png").write(tool_info)
                        except Exception as e:
                            st.error(f"处理 ToolMessage 时出错: {str(e)}")
                    _printed.add(s.id)


        # We can reuse the tutorial questions from part 1 to see how it does.
        try:
            with OutputCatcher() as output_catcher:

                for question in tutorial_questions:
                    events = part_2_graph.stream(
                        {"messages": ("user", question)}, config, stream_mode="values"
                    )
                    for event in events:
                        display_messages(event, _printed)
                    snapshot = part_2_graph.get_state(config)
                    while snapshot.next:
                        # We have an interrupt! The agent is trying to use a tool, and the user can approve or deny it
                        # Note: This code is all outside of your graph. Typically, you would stream the output to a UI.
                        # Then, you would have the frontend trigger a new run via an API call when the user has provided input.
                        st.write("plesae input")
                        user_input = st.chat_input(
                            "Do you approve of the above actions? Type 'y' to continue; otherwise, explain your requested changed.",key=uuid.uuid4()
                        )

                        if user_input is not None and user_input.strip() == "y":
                            # Just continue
                            result = part_2_graph.invoke(
                                None,
                                config,
                            )
                        elif user_input is not None:
                            # Satisfy the tool invocation by
                            # providing instructions on the requested changes / change of mind
                            result = part_2_graph.invoke(
                                {
                                    "messages": [
                                        ToolMessage(
                                            tool_call_id=event["messages"][-1].tool_calls[0]["id"],
                                            content=f"API call denied by user. Reasoning: '{user_input}'. Continue assisting, accounting for the user's input.",
                                        )
                                    ]
                                },
                                config,
                            )
                        snapshot = part_2_graph.get_state(config)

        except Exception as e:
            st.info(f"运行出错：{str(e)}")
            st.toast(f"运行出错: {str(e)}", icon='🚫')

            # llm = ChatOpenAI(model="gpt-4o",
            # api_key="sk-usqU5KXzBCpOP2T881D9A21838Fe47Ed8f64Ce753e192aE3",
            # base_url="https://api.gpts.vin/v1")