import getpass
import os
import time
import uuid

import pytz
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.utils.math import cosine_similarity
from langchain_openai import ChatOpenAI
from langgraph.constants import END, START

import gradio as gr
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

# 可不可以参考之前的各种api_key的输入呢？

#To fix this, please make sure that the key argument is unique for each widget you create.



# ————————————————————————————————————————————————————————————————————————————————————————————————————————————————————#


import os
import shutil
import sqlite3
import pandas as pd
import requests
import re
import numpy as np
import gradio as gr
from langchain_core.tools import tool

db_url = "https://storage.googleapis.com/benchmarks-artifacts/travel-db/travel2.sqlite"
local_file = "travel2.sqlite"
backup_file = "travel2.backup.sqlite"
model_name = 'bge-large-zh-v1.5'
embedding_model = HuggingFaceEmbeddings(model_name=model_name)

# 全局变量用于存储数据库和向量库状态
db = None
retriever = None
vector_store_initialized = False

session_state = {
    "langchain_api_key_set": False,
    "tavily_api_key_set": False,
    "langchain_api_key": "",
    "tavily_api_key": "",
    "openai_api_key": "",
    "zhipu_api_key": ""
}

def set_api_keys(langchain_key, tavily_key, openai_key, zhipu_key):
    """
    设置api_key。
    """
    session_state["langchain_api_key"] = langchain_key
    session_state["tavily_api_key"] = tavily_key
    session_state["openai_api_key"] = openai_key
    session_state["zhipu_api_key"] = zhipu_key

    if langchain_key:
        os.environ["LANGCHAIN_API_KEY"] = langchain_key
        session_state["langchain_api_key_set"] = True

    if tavily_key:
        os.environ["TAVILY_API_KEY"] = tavily_key
        session_state["tavily_api_key_set"] = True

    # Add any necessary initialization for TavilySearchResults here

    return f"LANGCHAIN_API_KEY 设置为: {langchain_key}", f"TAVILY_API_KEY 设置为: {tavily_key}", f"OpenAI API Key 设置为: {openai_key}", f"智谱AI API Key 设置为: {zhipu_key}"

def initialize_vector_store():
    """
    初始化向量库
    """
    global db, retriever, vector_store_initialized

    if vector_store_initialized:
        return "向量库已构建", gr.update(visible=True)

    # 下载并处理数据库
    if not os.path.exists(local_file):
        response = requests.get(db_url)
        response.raise_for_status()
        with open(local_file, "wb") as f:
            f.write(response.content)
        shutil.copy(local_file, backup_file)

    conn = sqlite3.connect(local_file)
    cursor = conn.cursor()

    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn).name.tolist()
    tdf = {t: pd.read_sql(f"SELECT * FROM {t}", conn) for t in tables}

    example_time = pd.to_datetime(tdf["flights"]["actual_departure"].replace("\\N", pd.NaT)).max()
    current_time = pd.to_datetime("now").tz_localize(example_time.tz)
    time_diff = current_time - example_time

    tdf["bookings"]["book_date"] = pd.to_datetime(tdf["bookings"]["book_date"].replace("\\N", pd.NaT), utc=True) + time_diff

    datetime_columns = ["scheduled_departure", "scheduled_arrival", "actual_departure", "actual_arrival"]
    for column in datetime_columns:
        tdf["flights"][column] = pd.to_datetime(tdf["flights"][column].replace("\\N", pd.NaT)) + time_diff

    for table_name, df in tdf.items():
        df.to_sql(table_name, conn, if_exists="replace", index=False)

    conn.commit()
    conn.close()

    db = local_file

    with open('swiss_faq.md', 'r', encoding='utf-8') as file:
        faq_text = file.read()

    docs = [{"page_content": txt} for txt in re.split(r"(?=\n##)", faq_text)]

    class VectorStoreRetriever:
        def __init__(self, docs, vectors):
            self._arr = np.array(vectors)
            self._docs = docs

        @classmethod
        def from_docs(cls, docs):
            doc_texts = [doc["page_content"] for doc in docs]
            embeddings = embedding_model.embed_documents(doc_texts)
            vectors = embeddings
            return cls(docs, vectors)

        def query(self, query: str, k: int = 5) -> list[dict]:
            embed = embedding_model.embed_documents([query])[0]
            scores = np.array(embed) @ self._arr.T
            top_k_idx = np.argpartition(scores, -k)[-k:]
            top_k_idx_sorted = top_k_idx[np.argsort(-scores[top_k_idx])]
            return [{**self._docs[idx], "similarity": scores[idx]} for idx in top_k_idx_sorted]

    retriever = VectorStoreRetriever.from_docs(docs)
    vector_store_initialized = True

    return "向量库构建完成", gr.update(visible=True)

@tool
def lookup_policy(query:str) -> str:
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


#————————————————————————Part2————————————-
from typing import Annotated

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from typing_extensions import TypedDict
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph
from langgraph.prebuilt import tools_condition
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

def fetch_user_info(state: State):
    """
    获取用户信息的函数。
    """
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
                        tool_info = f"Tool: {s.name}, Content: {s.content}"
                        chat.append(("tool", tool_info))
                except Exception as e:
                    chat.append(("error", f"处理 ToolMessage 时出错: {str(e)}"))
            _printed.add(s.id)
            yield chat

def run_chat(model_name, api_key, api_base, tutorial_questions, user_input_handler):
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

    config = {
        "configurable": {
            "passenger_id": "3442 587242",
            "thread_id": f"{uuid.uuid4()}",
        }
    }
    _printed = set()
    chat = []
    try:
        for question in tutorial_questions:
            events = part_2_graph.stream({"messages": ("user", question)}, config, stream_mode="values")
            for event in events:
                for chat_update in display_messages(event, _printed, chat):
                    yield chat_update
            snapshot = part_2_graph.get_state(config)
            while snapshot.next:
                chat.append(("ai", "Do you approve of the above actions? Type 'y' to continue; otherwise, explain your requested change.\n\n"))
                yield chat

                user_input = yield "input"
                chat.append(("human", user_input))
                yield chat

                if user_input.strip() == "y":
                    result = part_2_graph.invoke(None, config)
                else:
                    result = part_2_graph.invoke(
                        {"messages": [ToolMessage(tool_call_id=event["messages"][-1].tool_calls[0]["id"], content=f"API call denied by user. Reasoning: '{user_input}'. Continue assisting, accounting for the user's input.")]}
                    , config)
                snapshot = part_2_graph.get_state(config)
    except Exception as e:
        chat.append(("error", f"运行出错：{str(e)}"))
        yield chat

def main_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Customer Chat bot (Including Flight, Car, Hotel, Etc.)")

        with gr.Row():
            build_button = gr.Button("构建向量库")
            status_text = gr.Textbox(label="状态", interactive=False)

        with gr.Column(visible=False) as api_keys_section:
            with gr.Tab("OpenAI"):
                openai_api_key = gr.Textbox(label="OpenAI API Key", type="password")
                openai_api_base = gr.Textbox(label="OpenAI API Base", value="https://api.gpts.vin/v1")
                gr.Markdown("[获取OpenAI API key](https://platform.openai.com/account/api-keys)")
                gr.Markdown("[OpenAI API文档](https://platform.openai.com/docs/api-reference/introduction)")
                gr.Markdown("要用直连原版的API话，要开VPN，端口设置为7890。用中转的不用开VPN，已测试过中转的跟直连的效果一样。")

            with gr.Tab("智谱"):
                zhipu_api_key = gr.Textbox(label="智谱AI的API Key", type="password")
                zhipu_api_base = gr.Textbox(label="智谱AI的API Base")
                gr.Markdown("[获取智谱AI的API key](https://www.zhipuai.cn/)")
                gr.Markdown("国产的LLM模型基本上无法完成任务，但是可能可以通过修改prompt完成任务")

            with gr.Row():
                langchain_api_key = gr.Textbox(label="Langchain API Key", type="password")
                tavily_api_key = gr.Textbox(label="Tavily API Key", type="password")

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

        model_name = gr.Textbox(label="输入你要使用的模型:", value="gpt-4o", placeholder="gpt-4o")
        start_button = gr.Button("开始")
        chat_output = gr.Chatbot(label="Chat")
        user_input_box = gr.Textbox(label="Your Input")

        def on_start(openai_api_key, zhipu_api_key, openai_api_base, zhipu_api_base, model_name, user_input):
            api_key = openai_api_key if openai_api_key else zhipu_api_key
            api_base = openai_api_base if openai_api_base else zhipu_api_base

            chat_generator = run_chat(model_name, api_key, api_base, tutorial_questions, user_input_handler)
            for chat in chat_generator:
                if chat == "input":
                    return chat  # 继续显示输入框
                yield chat

        def user_input_handler(user_input):
            return user_input

        start_button.click(on_start, inputs=[openai_api_key, zhipu_api_key, openai_api_base, zhipu_api_base, model_name, user_input_box], outputs=chat_output)

        user_input_box.submit(lambda user_input: user_input_handler(user_input), inputs=[user_input_box], outputs=chat_output)


    return demo


if __name__ == "__main__":
    demo = main_interface()
    demo.launch(share=True)

# https://tavily.com/
# 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
# https://smith.langchain.com/settings
# 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

# llm = ChatOpenAI(model="gpt-4o",
# api_key="sk-usqU5KXzBCpOP2T881D9A21838Fe47Ed8f64Ce753e192aE3",
# base_url="https://api.gpts.vin/v1")

#运行到最后错误的时候，才会在chatbox中展示
#user_input的textbox似乎会出错，到时候验证一下？



#编号1的代码：
#
# def display_messages(event, _printed, chat):
#     for s in event['messages']:
#         if s.id not in _printed:
#             if isinstance(s, HumanMessage):
#                 chat.append(("human", s.content))
#             elif isinstance(s, AIMessage):
#                 if not s.content and 'tool_calls' in s.additional_kwargs:
#                     try:
#                         tool_calls = s.additional_kwargs['tool_calls']
#                         tool_names = [tool_call['function']['name'] for tool_call in tool_calls if 'function' in tool_call and 'name' in tool_call['function']]
#                         chat.append(("ai", f"I will use the following tools: {', '.join(tool_names)}"))
#                     except Exception as e:
#                         chat.append(("error", f"Error processing AIMessage tool calls: {str(e)}"))
#                 else:
#                     chat.append(("ai", s.content))
#             elif isinstance(s, ToolMessage):
#                 try:
#                     if 'HTTPError' in s.content:
#                         chat.append(("error", f"Tool调用失败：{s.content}，请检查你的API key是否正确"))
#                     else:
#                         tool_info = f"Tool: {s.name}, Content: {s.content}"
#                         chat.append(("tool", tool_info))
#                 except Exception as e:
#                     chat.append(("error", f"处理 ToolMessage 时出错: {str(e)}"))
#             _printed.add(s.id)
#             yield chat
# from langgraph.checkpoint.sqlite import SqliteSaver
# from langgraph.graph import StateGraph
# from langgraph.prebuilt import tools_condition
#
# def run_chat(model_name, api_key, api_base, tutorial_questions, user_input_handler):
#     llm = ChatOpenAI(model=model_name, api_key=api_key, base_url=api_base)
#     assistant_prompt = ChatPromptTemplate.from_messages([
#         ("system", "You are a helpful customer support assistant for Swiss Airlines. Use the provided tools to search for flights, company policies, and other information to assist the user's queries. When searching, be persistent. Expand your query bounds if the first search returns no results. If a search comes up empty, expand your search before giving up.\n\nCurrent user:\n\n{user_info}\n\nCurrent time: {time}."),
#         ("placeholder", "{messages}"),
#     ]).partial(time=datetime.now())
#
#     part_2_tools = [
#         TavilySearchResults(max_results=1),
#         fetch_user_flight_information,
#         search_flights,
#         lookup_policy,
#         update_ticket_to_new_flight,
#         cancel_ticket,
#         search_car_rentals,
#         book_car_rental,
#         update_car_rental,
#         cancel_car_rental,
#         search_hotels,
#         book_hotel,
#         update_hotel,
#         cancel_hotel,
#         search_trip_recommendations,
#         book_excursion,
#         update_excursion,
#         cancel_excursion,
#     ]
#     part_2_assistant_runnable = assistant_prompt | llm.bind_tools(part_2_tools)
#
#     builder = StateGraph(State)
#     builder.add_node("fetch_user_info", fetch_user_info)
#     builder.add_edge(START, "fetch_user_info")
#     builder.add_node("assistant", Assistant(part_2_assistant_runnable))
#     builder.add_node("tools", create_tool_node_with_fallback(part_2_tools))
#     builder.add_edge("fetch_user_info", "assistant")
#     builder.add_conditional_edges("assistant", tools_condition)
#     builder.add_edge("tools", "assistant")
#
#     memory = SqliteSaver.from_conn_string(":memory:")
#     part_2_graph = builder.compile(checkpointer=memory, interrupt_before=["tools"])
#
#     config = {
#         "configurable": {
#             "passenger_id": "3442 587242",
#             "thread_id": f"{uuid.uuid4()}",
#         }
#     }
#     _printed = set()
#     chat = []
#     try:
#         for question in tutorial_questions:
#             events = part_2_graph.stream({"messages": ("user", question)}, config, stream_mode="values")
#             for event in events:
#                 for chat_update in display_messages(event, _printed, chat):
#                     yield chat_update
#             snapshot = part_2_graph.get_state(config)
#             while snapshot.next:
#                 chat.append(("ai", "Do you approve of the above actions? Type 'y' to continue; otherwise, explain your requested change.\n\n"))
#                 yield chat
#
#                 user_input = yield "input"
#                 chat.append(("human", user_input))
#                 yield chat
#
#                 if user_input.strip() == "y":
#                     result = part_2_graph.invoke(None, config)
#                 else:
#                     result = part_2_graph.invoke(
#                         {"messages": [ToolMessage(tool_call_id=event["messages"][-1].tool_calls[0]["id"], content=f"API call denied by user. Reasoning: '{user_input}'. Continue assisting, accounting for the user's input.")]}
#                     , config)
#                 snapshot = part_2_graph.get_state(config)
#     except Exception as e:
#         chat.append(("error", f"运行出错：{str(e)}"))
#         yield chat
#
# def main_interface():
#     with gr.Blocks() as demo:
#         gr.Markdown("# Customer Chat bot (Including Flight, Car, Hotel, Etc.)")
#
#         with gr.Row():
#             build_button = gr.Button("构建向量库")
#             status_text = gr.Textbox(label="状态", interactive=False)
#
#         with gr.Column(visible=False) as api_keys_section:
#             with gr.Tab("OpenAI"):
#                 openai_api_key = gr.Textbox(label="OpenAI API Key", type="password")
#                 openai_api_base = gr.Textbox(label="OpenAI API Base", value="https://api.gpts.vin/v1")
#                 gr.Markdown("[获取OpenAI API key](https://platform.openai.com/account/api-keys)")
#                 gr.Markdown("[OpenAI API文档](https://platform.openai.com/docs/api-reference/introduction)")
#                 gr.Markdown("要用直连原版的API话，要开VPN，端口设置为7890。用中转的不用开VPN，已测试过中转的跟直连的效果一样。")
#
#             with gr.Tab("智谱"):
#                 zhipu_api_key = gr.Textbox(label="智谱AI的API Key", type="password")
#                 zhipu_api_base = gr.Textbox(label="智谱AI的API Base")
#                 gr.Markdown("[获取智谱AI的API key](https://www.zhipuai.cn/)")
#                 gr.Markdown("国产的LLM模型基本上无法完成任务，但是可能可以通过修改prompt完成任务")
#
#             with gr.Row():
#                 langchain_api_key = gr.Textbox(label="Langchain API Key", type="password")
#                 tavily_api_key = gr.Textbox(label="Tavily API Key", type="password")
#
#             gr.Button("设置 API 密钥").click(
#                 set_api_keys,
#                 inputs=[langchain_api_key, tavily_api_key, openai_api_key, zhipu_api_key],
#                 outputs=[gr.Text(), gr.Text(), gr.Text(), gr.Text()]
#             )
#
#         build_button.click(
#             initialize_vector_store,
#             inputs=[],
#             outputs=[status_text, api_keys_section]
#         )
#
#         model_name = gr.Textbox(label="输入你要使用的模型:", value="gpt-4o", placeholder="gpt-4o")
#         start_button = gr.Button("开始")
#         chat_output = gr.Chatbot(label="Chat")
#         user_input_box = gr.Textbox(label="Your Input")
#
#         def on_start(openai_api_key, zhipu_api_key, openai_api_base, zhipu_api_base, model_name, user_input):
#             api_key = openai_api_key if openai_api_key else zhipu_api_key
#             api_base = openai_api_base if openai_api_base else zhipu_api_base
#
#             chat_generator = run_chat(model_name, api_key, api_base, tutorial_questions, user_input_handler)
#             for chat in chat_generator:
#                 if chat == "input":
#                     return chat  # 继续显示输入框
#                 yield chat
#
#         def user_input_handler(user_input):
#             return user_input
#
#         start_button.click(on_start, inputs=[openai_api_key, zhipu_api_key, openai_api_base, zhipu_api_base, model_name, user_input_box], outputs=chat_output)
#
#     return demo
#
# if __name__ == "__main__":
#     demo = main_interface()
#     demo.launch(share=True)
#
#
#
# #编号2的代码：
# def display_messages(event, _printed, chat):
#     for s in event['messages']:
#         if s.id not in _printed:
#             if isinstance(s, HumanMessage):
#                 chat.append(("human", s.content))
#             elif isinstance(s, AIMessage):
#                 if not s.content and 'tool_calls' in s.additional_kwargs:
#                     try:
#                         tool_calls = s.additional_kwargs['tool_calls']
#                         tool_names = [tool_call['function']['name'] for tool_call in tool_calls if 'function' in tool_call and 'name' in tool_call['function']]
#                         chat.append(("ai", f"I will use the following tools: {', '.join(tool_names)}"))
#                     except Exception as e:
#                         chat.append(("error", f"Error processing AIMessage tool calls: {str(e)}"))
#                 else:
#                     chat.append(("ai", s.content))
#             elif isinstance(s, ToolMessage):
#                 try:
#                     if 'HTTPError' in s.content:
#                         chat.append(("error", f"Tool调用失败：{s.content}，请检查你的API key是否正确"))
#                     else:
#                         tool_info = f"Tool: {s.name}, Content: {s.content}"
#                         chat.append(("tool", tool_info))
#                 except Exception as e:
#                     chat.append(("error", f"处理 ToolMessage 时出错: {str(e)}"))
#             _printed.add(s.id)
#             yield chat
#
# def run_chat(model_name, api_key, api_base, tutorial_questions, user_input_handler):
#     llm = ChatOpenAI(model=model_name, api_key=api_key, base_url=api_base)
#     assistant_prompt = ChatPromptTemplate.from_messages([
#         ("system", "You are a helpful customer support assistant for Swiss Airlines. Use the provided tools to search for flights, company policies, and other information to assist the user's queries. When searching, be persistent. Expand your query bounds if the first search returns no results. If a search comes up empty, expand your search before giving up.\n\nCurrent user:\n\n{user_info}\n\nCurrent time: {time}."),
#         ("placeholder", "{messages}"),
#     ]).partial(time=datetime.now())
#
#     part_2_tools = [
#         TavilySearchResults(max_results=1),
#         fetch_user_flight_information,
#         search_flights,
#         lookup_policy,
#         update_ticket_to_new_flight,
#         cancel_ticket,
#         search_car_rentals,
#         book_car_rental,
#         update_car_rental,
#         cancel_car_rental,
#         search_hotels,
#         book_hotel,
#         update_hotel,
#         cancel_hotel,
#         search_trip_recommendations,
#         book_excursion,
#         update_excursion,
#         cancel_excursion,
#     ]
#     part_2_assistant_runnable = assistant_prompt | llm.bind_tools(part_2_tools)
#
#     builder = StateGraph(State)
#     builder.add_node("fetch_user_info", fetch_user_info)
#     builder.add_edge(START, "fetch_user_info")
#     builder.add_node("assistant", Assistant(part_2_assistant_runnable))
#     builder.add_node("tools", create_tool_node_with_fallback(part_2_tools))
#     builder.add_edge("fetch_user_info", "assistant")
#     builder.add_conditional_edges("assistant", tools_condition)
#     builder.add_edge("tools", "assistant")
#
#     memory = SqliteSaver.from_conn_string(":memory:")
#     part_2_graph = builder.compile(checkpointer=memory, interrupt_before=["tools"])
#
#     config = {
#         "configurable": {
#             "passenger_id": "3442 587242",
#             "thread_id": f"{uuid.uuid4()}",
#         }
#     }
#     _printed = set()
#     chat = []
#     try:
#         for question in tutorial_questions:
#             events = part_2_graph.stream({"messages": ("user", question)}, config, stream_mode="values")
#             for event in events:
#                 print("——————————————————————————")
#                 print(f"Event:{event}")
#                 for chat_update in display_messages(event, _printed, chat):
#                     yield chat_update
#             snapshot = part_2_graph.get_state(config)
#             while snapshot.next:
#                 chat.append(("ai", "Do you approve of the above actions? Type 'y' to continue; otherwise, explain your requested change.\n\n"))
#                 yield chat
#
#                 user_input = yield "input"
#                 chat.append(("human", user_input))
#                 yield chat
#
#                 if user_input.strip() == "y":
#                     result = part_2_graph.invoke(None, config)
#                 else:
#                     result = part_2_graph.invoke(
#                         {"messages": [ToolMessage(tool_call_id=event["messages"][-1].tool_calls[0]["id"], content=f"API call denied by user. Reasoning: '{user_input}'. Continue assisting, accounting for the user's input.")]}
#                     , config)
#                 snapshot = part_2_graph.get_state(config)
#     except Exception as e:
#         chat.append(("error", f"运行出错：{str(e)}"))
#         yield chat
#
# def main_interface():
#     with gr.Blocks() as demo:
#         gr.Markdown("# Customer Chat bot (Including Flight, Car, Hotel, Etc.)")
#
#         with gr.Row():
#             build_button = gr.Button("构建向量库")
#             status_text = gr.Textbox(label="状态", interactive=False)
#
#         with gr.Column(visible=False) as api_keys_section:
#             with gr.Tab("OpenAI"):
#                 openai_api_key = gr.Textbox(label="OpenAI API Key", type="password")
#                 openai_api_base = gr.Textbox(label="OpenAI API Base", value="https://api.gpts.vin/v1")
#                 gr.Markdown("[获取OpenAI API key](https://platform.openai.com/account/api-keys)")
#                 gr.Markdown("[OpenAI API文档](https://platform.openai.com/docs/api-reference/introduction)")
#                 gr.Markdown("要用直连原版的API话，要开VPN，端口设置为7890。用中转的不用开VPN，已测试过中转的跟直连的效果一样。")
#
#             with gr.Tab("智谱"):
#                 zhipu_api_key = gr.Textbox(label="智谱AI的API Key", type="password")
#                 zhipu_api_base = gr.Textbox(label="智谱AI的API Base")
#                 gr.Markdown("[获取智谱AI的API key](https://www.zhipuai.cn/)")
#                 gr.Markdown("国产的LLM模型基本上无法完成任务，但是可能可以通过修改prompt完成任务")
#
#             with gr.Row():
#                 langchain_api_key = gr.Textbox(label="Langchain API Key", type="password")
#                 tavily_api_key = gr.Textbox(label="Tavily API Key", type="password")
#
#             gr.Button("设置 API 密钥").click(
#                 set_api_keys,
#                 inputs=[langchain_api_key, tavily_api_key, openai_api_key, zhipu_api_key],
#                 outputs=[gr.Text(), gr.Text(), gr.Text(), gr.Text()]
#             )
#
#         build_button.click(
#             initialize_vector_store,
#             inputs=[],
#             outputs=[status_text, api_keys_section]
#         )
#
#         model_name = gr.Textbox(label="输入你要使用的模型:", value="gpt-4o", placeholder="gpt-4o")
#         start_button = gr.Button("开始")
#         chat_output = gr.Chatbot(label="Chat")
#         user_input_box = gr.Textbox(label="Your Input", visible=False)
#         submit_button = gr.Button("提交", visible=False)
#
#         def on_start(openai_api_key, zhipu_api_key, openai_api_base, zhipu_api_base, model_name):
#             api_key = openai_api_key if openai_api_key else zhipu_api_key
#             api_base = openai_api_base if openai_api_base else zhipu_api_base
#
#             chat_generator = run_chat(model_name, api_key, api_base, tutorial_questions, user_input_handler)
#             for chat in chat_generator:
#                 if chat == "input":
#                     user_input_box.visible = True
#                     submit_button.visible = True
#                     yield chat_output, user_input_box, submit_button
#                 else:
#                     yield chat_output, user_input_box, submit_button
#
#         def user_input_handler(user_input):
#             return user_input
#
#         start_button.click(on_start, inputs=[openai_api_key, zhipu_api_key, openai_api_base, zhipu_api_base, model_name], outputs=[chat_output, user_input_box, submit_button])
#
#         def submit_user_input(user_input):
#             return user_input
#
#         submit_button.click(submit_user_input, inputs=[user_input_box], outputs=chat_output)
#         user_input_box.submit(submit_user_input, inputs=[user_input_box], outputs=chat_output)
#
#     return demo
#
# if __name__ == "__main__":
#     demo = main_interface()
#     demo.launch(share=True)


# #请你参考编号1和编号2，编号1的代码可以正常运行并出现一个textbox，但是他的问题是：很好，你很好地完成了任务。但是目前还是有瑕疵，我向你说明一下目前遇到的问题和瑕疵。
#
# 1.Human的“Your Input”的文本框的出现比AI的询问要快。正常的流程的话，应该是AI的询问先出现，然后才是Human的文本框出现。
# 2.Human的文本框，无法提交，我想应该可以设置一个“回车键”或者“提交按钮”进行提交。
#
# 编号2 的问题是：我使用print的时候发现有效果，但是实际上，chatbot却不会出现任何显示，我不确定这种方法能不能解决编号1的问题，这只是一种思路供你参考
# 我希望你以编号1为基础，集中解决编号1的问题，尤其是：提交问题