import getpass
import os

import pytz
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.utils.math import cosine_similarity
from langchain_openai import ChatOpenAI
from langgraph.constants import END, START


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

def _set_env(var: str):
    if not os.environ.get(var):
        print(f"{var} not set, setting...")
        os.environ[var] = getpass.getpass(f"{var} not set, please enter: ")


os.environ["TAVILY_API_KEY"] = "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"
os.environ["LANGCHAIN_API_KEY"] = "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"
print(os.environ["TAVILY_API_KEY"])
print(os.environ["LANGCHAIN_API_KEY"])
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Customer Support Bot"
import os
import shutil
import sqlite3
import pandas as pd
import requests

db_url = "https://storage.googleapis.com/benchmarks-artifacts/travel-db/travel2.sqlite"
local_file = "travel2.sqlite"

backup_file = "travel2.backup.sqlite"
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

db = local_file

import re
import numpy as np
from langchain_core.tools import tool

model_name = 'bge-large-zh-v1.5'
embedding_model = HuggingFaceEmbeddings(model_name=model_name)

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


retriever = VectorStoreRetriever.from_docs(docs)


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


from langchain_core.messages import ToolMessage
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


# ——————————————————————————————————————Part1————————————————————————————————————————
#
# from typing import Annotated
# from typing_extensions import TypedDict
# from langgraph.graph.message import AnyMessage, add_messages
#
#
# class State(TypedDict):
#     messages: Annotated[list[AnyMessage], add_messages]
#
#
# # Agent
#
# from langchain_anthropic import ChatAnthropic
# from langchain_community.tools.tavily_search import TavilySearchResults
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import Runnable, RunnableConfig
#
#
# class Assistant:
#     def __init__(self, runnable: Runnable):
#         self.runnable = runnable
#
#     def __call__(self, state: State, config: RunnableConfig):
#         while True:
#             configuration = config.get("configurable", {})
#             passenger_id = configuration.get("passenger_id", None)
#             state = {**state, "user_info": passenger_id}
#             result = self.runnable.invoke(state)
#
#             if not result.tool_calls and (
#                     not result.content
#                     or isinstance(result.content, list)
#                     and not result.content[0].get("text")
#
#             ):
#                 messages = state["messages"] + [("user", "Respond with a real output.")]
#                 state = {**state, "messages": messages}
#             else:
#                 break
#
#         return {"messages": result}
#
#
# from langchain_openai import ChatOpenAI
#
# llm = ChatOpenAI(model="gpt-4-1106-preview",
#                  api_key="sk-9oYJRePIyAbz7wNj955dBbC98f0c44F8B91bF7779d38B131",
#                  base_url="https://gtapi.xiaoerchaoren.com:8932/v1")
#
# primary_assistant_prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "You are a helpful customer support assistant for Swiss Airlines. "
#             " Use the provided tools to search for flights, company policies, and other information to assist the user's queries. "
#             " When searching, be persistent. Expand your query bounds if the first search returns no results. "
#             " If a search comes up empty, expand your search before giving up."
#             "\n\nCurrent user:\n\n{user_info}\n"
#             "\nCurrent time: {time}.",
#         ),
#         ("placeholder", "{messages}"),
#     ]
# ).partial(time=datetime.now())
#
# part_1_tools = [
#     TavilySearchResults(max_results=1),
#     fetch_user_flight_information,
#     search_flights,
#     lookup_policy,
#     update_ticket_to_new_flight,
#     cancel_ticket,
#     search_car_rentals,
#     book_car_rental,
#     update_car_rental,
#     cancel_car_rental,
#     search_hotels,
#     book_hotel,
#     update_hotel,
#     cancel_hotel,
#     search_trip_recommendations,
#     book_excursion,
#     update_excursion,
#     cancel_excursion,
#
# ]
# part_1_assistant_runnable = primary_assistant_prompt | llm.bind_tools(part_1_tools)
#
# # Define Graph
#
# from langgraph.checkpoint.sqlite import SqliteSaver
# from langgraph.graph import END, StateGraph, START
# from langgraph.prebuilt import tools_condition
#
# builder = StateGraph(State)
#
# # Define nodes: these do the work
# builder.add_node("assistant", Assistant(part_1_assistant_runnable))
# builder.add_node("tools", create_tool_node_with_fallback(part_1_tools))
# # Define edges: these determine how the control flow moves
# builder.add_edge(START, "assistant")
# builder.add_conditional_edges(
#     "assistant",
#     tools_condition,
# )
# builder.add_edge("tools", "assistant")
#
# # The checkpointer lets the graph persist its state
# # this is a complete memory for the entire graph.
# memory = SqliteSaver.from_conn_string(":memory:")
# part_1_graph = builder.compile(checkpointer=memory)
#
# from IPython.display import Image, display
#
# try:
#     # 获取图形的 PNG 数据
#     png_data = part_1_graph.get_graph(xray=True).draw_mermaid_png()
#
#     # 指定文件名
#     file_name = "graph.png"
#
#     # 将 PNG 数据写入文件
#     with open(file_name, "wb") as file:
#         file.write(png_data)
#
#     print(f"Graph saved as {file_name}")
# except Exception as e:
#     # 处理异常，例如输出错误信息或采取其他恰当的措施
#     print(f"Failed to display graph: {e}")
#
# import shutil
# import uuid
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
#
# shutil.copy(backup_file, db)
# print(f"Backup file {backup_file} copied to {db}")
# thread_id = str(uuid.uuid4())
# print(f"Generated thread_id: {thread_id}")
#
# config = {
#     "configurable": {
#         # The passener_id is used in our flight tools to fetch the user's flight information
#         "passenger_id": "3442 587242",
#
#         # Checkpoints are accessed by thread_id
#         "thread_id": thread_id,
#     }
# }
# print(f"Configuration: {config}")
# #
# # _printed = set()
# # for question in tutorial_questions:
# #     print(f"Processing question: {question}")
# #     events = part_1_graph.stream(
# #         {"messages": ("user", question)}, config, stream_mode="values"
# #     )
# #     for event in events:
# #         _print_event(event, _printed)
# #         print(f"Printed event: {event}")
#

# ————————————————————————————————Part 2: Add Confirmation——————————————————————————————————————
# ''''When an assistant takes actions on behalf of the user, the user should (almost) always have the final say on whether to follow through with the actions. Otherwise, any small mistake the assistant makes (or any prompt injection it succombs to) can cause real damage to the user.
#
# In this section, we will use interrupt_before to pause the graph and return control to the user before executing any of the tools.
#
# Your graph will look something like the following:'''
#
# # State & Assistant
#
# from typing import Annotated
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
#
#             result = self.runnable.invoke(state)
#
#             if not result.tool_calls and (
#                     not result.content
#                     or isinstance(result.content, list)
#                     and not result.content[0].get("text")
#
#             ):
#                 messages = state["messages"] + [("user", "Respond with a real output.")]
#                 state = {**state, "messages": messages}
#             else:
#                 break
#
#         return {"messages": result}
#
#
# # llm = ChatAnthropic(model="claude-3-sonnet-20240229",
# #                     api_key="sk-boFgxLeJNEtzKNgo1b0c6b9f35684cFc90Ed3bDaDe970a74",
# #                     base_url="https://api.claude-Plus.top")
# #
# # llm = ChatOpenAI(model = "claude-3-5-sonnet-20240620",
# #                    api_key = "sk-boFgxLeJNEtzKNgo1b0c6b9f35684cFc90Ed3bDaDe970a74",
# #                    base_url = "https://api.claude-plus.top/v1")
# # llm = ChatOpenAI(model="gpt-4-turbo-preview",
# #                  api_key="sk-v73ENyjuJjl9UW2X0fC79c668f89467f91A26911Bf6cAa81",
# #                  base_url="https://gtapi.xiaoerchaoren.com:8932/v1")
#
# assistant_prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "You are a helpful customer support assistant for Swiss Airlines. "
#             " Use the provided tools to search for flights, company policies, and other information to assist the user's queries. "
#             " When searching, be persistent. Expand your query bounds if the first search returns no results. "
#             " If a search comes up empty, expand your search before giving up."
#             "\n\nCurrent user:\n\n{user_info}\n"
#             "\nCurrent time: {time}.",
#         ),
#         ("placeholder", "{messages}"),
#     ]
# ).partial(time=datetime.now())
#
# part_2_tools = [
#     TavilySearchResults(max_results=1),
#     fetch_user_flight_information,
#     search_flights,
#     lookup_policy,
#     update_ticket_to_new_flight,
#     cancel_ticket,
#     search_car_rentals,
#     book_car_rental,
#     update_car_rental,
#     cancel_car_rental,
#     search_hotels,
#     book_hotel,
#     update_hotel,
#     cancel_hotel,
#     search_trip_recommendations,
#     book_excursion,
#     update_excursion,
#     cancel_excursion,
#
# ]
# part_2_assistant_runnable = assistant_prompt | llm.bind_tools(part_2_tools)
#
# from langgraph.checkpoint.sqlite import SqliteSaver
# from langgraph.graph import StateGraph
# from langgraph.prebuilt import tools_condition
#
# from langgraph.graph import END, StateGraph, START
#
# builder = StateGraph(State)
#
#
# def user_info(state: State):
#     return {"user_info": fetch_user_flight_information.invoke({})}
#
#
# builder.add_node("fetch_user_info", user_info)
# builder.add_edge(START, "fetch_user_info")
# builder.add_node("assistant", Assistant(part_2_assistant_runnable))
# builder.add_node("tools", create_tool_node_with_fallback(part_2_tools))
# builder.add_edge("fetch_user_info", "assistant")
# builder.add_conditional_edges(
#     "assistant",
#     tools_condition,
# )
# builder.add_edge("tools", "assistant")
#
# memory = SqliteSaver.from_conn_string(":memory:")
# part_2_graph = builder.compile(
#     checkpointer=memory,
#     interrupt_before=["tools"],
# )
#
# try:
#     # 获取图形的 PNG 数据
#     png_data = part_2_graph.get_graph(xray=True).draw_mermaid_png()
#
#     # 指定文件名
#     file_name = "graph.png"
#
#     # 将 PNG 数据写入文件
#     with open(file_name, "wb") as file:
#         file.write(png_data)
#
#     print(f"Graph saved as {file_name}")
# except Exception as e:
#     # 处理异常，例如输出错误信息或采取其他恰当的措施
#     print(f"Failed to display graph: {e}")
#
# import shutil
# import uuid
#
# shutil.copy(backup_file, db)
# thread_id = str(uuid.uuid4())
#
# config = {
#     "configurable": {
#         "passenger_id": "3442 587242",
#         "thread_id": thread_id,
#     }
# }
#
# _printed = set()
#
# for question in tutorial_questions:
#     events = part_2_graph.stream(
#         {"messages": ("user", question)}, config, stream_mode="values"
#     )
#     for event in events:
#         print(f"Pat2_Event: {event}")  # 打印事件信息
#         print("————————————————————————————————————————————————")
#         _print_event(event, _printed)
#     snapshot = part_2_graph.get_state(config)
#     while snapshot.next:
#         print("Snapshot state before user input:", snapshot)
#         user_input = input(
#             "Do you approve of the above actions? Type 'y' to continue;"
#             " otherwise, explain your requested changed.\n\n"
#         )
#         if user_input.strip() == "y":
#             result = part_2_graph.invoke(
#                 None,
#                 config,
#             )
#         else:
#             result = part_2_graph.invoke(
#                 {"messages": [
#                     ToolMessage(
#                         tool_call_id=event["messages"][-1].tool_calls[0]["id"],
#                         content=f"API call denied by user. Reasoning:'{user_input}'. Continue assisting, accounting for the user's input.",
#                     )
#                 ]},
#                 config,
#             )
#
#         snapshot = part_2_graph.get_state(config)

# ————————————————————————————————————————————————————Part 3——————————————————————————————————————————————

# '''In this section, we'll refine our interrupt strategy by categorizing tools as safe (read-only) or sensitive (data-modifying). We'll apply interrupts to the sensitive tools only, allowing the bot to handle simple queries autonomously.
#
# This balances user control and conversational flow, but as we add more tools, our single graph may grow too complex for this "flat" structure. We'll address that in the next section.
#
# Your graph for Part 3 will look something like the following diagram.'''
#
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
#                 not result.content
#                 or isinstance(result.content, list)
#                 and not result.content[0].get("text")
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
# from langchain_openai import ChatOpenAI
#
# llm = ChatOpenAI(model="gpt-4o",
#                  api_key="sk-v73ENyjuJjl9UW2X0fC79c668f89467f91A26911Bf6cAa81",
#                  base_url="https://gtapi.xiaoerchaoren.com:8932/v1")
#
#
# # llm = ChatOpenAI(model = "claude-3-sonnet-20240229",
# #                    openai_api_key = "sk-boFgxLeJNEtzKNgo1b0c6b9f35684cFc90Ed3bDaDe970a74",
# #                    openai_api_base = "https://api.claude-plus.top/v1")
#
# assistant_prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "You are a helpful customer support assistant for Swiss Airlines. "
#             " Use the provided tools to search for flights, company policies, and other information to assist the user's queries. "
#             " When searching, be persistent. Expand your query bounds if the first search returns no results. "
#             " If a search comes up empty, expand your search before giving up."
#             "\n\nCurrent user:\n\n{user_info}\n"
#             "\nCurrent time: {time}.",
#         ),
#         ("placeholder", "{messages}"),
#     ]
# ).partial(time=datetime.now())
#
#
# # "Read"-only tools (such as retrievers) don't need a user confirmation to use
# part_3_safe_tools = [
#     TavilySearchResults(max_results=1),
#
#     search_flights,
#     lookup_policy,
#     search_car_rentals,
#     search_hotels,
#     search_trip_recommendations,
# ]
#
# # These tools all change the user's reservations.
# # The user has the right to control what decisions are made
# part_3_sensitive_tools = [
#     fetch_user_flight_information,
#     update_ticket_to_new_flight,
#     cancel_ticket,
#     book_car_rental,
#     update_car_rental,
#     cancel_car_rental,
#     book_hotel,
#     update_hotel,
#     cancel_hotel,
#     book_excursion,
#     update_excursion,
#     cancel_excursion,
# ]
# sensitive_tool_names = {t.name for t in part_3_sensitive_tools}
# # Our LLM doesn't have to know which nodes it has to route to. In its 'mind', it's just invoking functions.
# part_3_assistant_runnable = assistant_prompt | llm.bind_tools(
#     part_3_safe_tools + part_3_sensitive_tools
# )
# # Define Graph
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
# # NEW: The fetch_user_info node runs first, meaning our assistant can see the user's flight information without
# # having to take an action
# builder.add_node("fetch_user_info", user_info)
# builder.add_edge(START, "fetch_user_info")
# builder.add_node("assistant", Assistant(part_3_assistant_runnable))
# builder.add_node("safe_tools", create_tool_node_with_fallback(part_3_safe_tools))
# builder.add_node(
#     "sensitive_tools", create_tool_node_with_fallback(part_3_sensitive_tools)
# )
# # Define logic
# builder.add_edge("fetch_user_info", "assistant")
#
#
# def route_tools(state: State) -> Literal["safe_tools", "sensitive_tools", "__end__"]:
#     next_node = tools_condition(state)
#     # If no tools are invoked, return to the user
#     if next_node == END:
#         return END
#     ai_message = state["messages"][-1]
#     # This assumes single tool calls. To handle parallel tool calling, you'd want to
#     # use an ANY condition
#     first_tool_call = ai_message.tool_calls[0]
#     if first_tool_call["name"] in sensitive_tool_names:
#         return "sensitive_tools"
#     return "safe_tools"
#
#
# builder.add_conditional_edges(
#     "assistant",
#     route_tools,
# )
# builder.add_edge("safe_tools", "assistant")
# builder.add_edge("sensitive_tools", "assistant")
#
# memory = SqliteSaver.from_conn_string(":memory:")
# part_3_graph = builder.compile(
#     checkpointer=memory,
#     # NEW: The graph will always halt before executing the "tools" node.
#     # The user can approve or reject (or even alter the request) before
#     # the assistant continues
#     interrupt_before=["sensitive_tools"],
# )
#
# try:
#     # 获取图形的 PNG 数据
#     png_data = part_3_graph.get_graph(xray=True).draw_mermaid_png()
#
#     # 指定文件名
#     file_name = "graph.png"
#
#     # 将 PNG 数据写入文件
#     with open(file_name, "wb") as file:
#         file.write(png_data)
#
#     print(f"Graph saved as {file_name}")
# except Exception as e:
#     # 处理异常，例如输出错误信息或采取其他恰当的措施
#     print(f"Failed to display graph: {e}")
#
# # Example Conversation
# import shutil
# import uuid
#
# # Update with the backup file so we can restart from the original place in each section
# shutil.copy(backup_file, db)
# thread_id = str(uuid.uuid4())
#
# config = {
#     "configurable": {
#         # The passenger_id is used in our flight tools to
#         # fetch the user's flight information
#         "passenger_id": "3442 587242",
#         # Checkpoints are accessed by thread_id
#         "thread_id": thread_id,
#     }
# }
#
# tutorial_questions = [
#     "Hi there, what time is my flight?",
#     "Am i allowed to update my flight to something sooner? I want to leave later today.",
#     "Update my flight to sometime next week then",
#     "The next available option is great",
#     "what about lodging and transportation?",
#     "Yeah i think i'd like an affordable hotel for my week-long stay (7 days). And I'll want to rent a car.",
#     "OK could you place a reservation for your recommended hotel? It sounds nice.",
#     "yes go ahead and book anything that's moderate expense and has availability.",
#     "Now for a car, what are my options?",
#
# ]
#
#
# _printed = set()
# # We can reuse the tutorial questions from part 1 to see how it does.
# for question in tutorial_questions:
#     events = part_3_graph.stream(
#         {"messages": ("user", question)}, config, stream_mode="values"
#     )
#     for event in events:
#         print(f"Event: {event}")
#         print("————————————————————————————")
#         _print_event(event, _printed)
#     snapshot = part_3_graph.get_state(config)
#     while snapshot.next:
#         # We have an interrupt! The agent is trying to use a tool, and the user can approve or deny it
#         # Note: This code is all outside of your graph. Typically, you would stream the output to a UI.
#         # Then, you would have the frontend trigger a new run via an API call when the user has provided input.
#         user_input = input(
#             "Do you approve of the above actions? Type 'y' to continue;"
#             " otherwise, explain your requested changed.\n\n"
#         )
#         if user_input.strip() == "y":
#             # Just continue
#             result = part_3_graph.invoke(
#                 None,
#                 config,
#             )
#         else:
#             # Satisfy the tool invocation by
#             # providing instructions on the requested changes / change of mind
#             result = part_3_graph.invoke(
#                 {
#                     "messages": [
#                         ToolMessage(
#                             tool_call_id=event["messages"][-1].tool_calls[0]["id"],
#                             content=f"API call denied by user. Reasoning: '{user_input}'. Continue assisting, accounting for the user's input.",
#                         )
#                     ]
#                 },
#                 config,
#             )
#         snapshot = part_3_graph.get_state(config)


# ——————————————————————————Part 4——————————————————————————————————————————————————————————————————————————

# State
from typing import Annotated, Literal, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages


def update_dialog_stack(left: list[str], right: Optional[str]) -> list[str]:
    if right is None:
        return left
    if right == "pop":
        return left[:-1]
    return left + [right]


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_info: str
    dialog_state: Annotated[
        list[
            Literal[
                "assistant",
                "update_flight",
                "book_car_rental",
                "book_hotel",
                "book_excursion",
            ]
        ],
        update_dialog_stack,
    ]
    tool_calls_stack: list[str]  # 添加这一行

# Assitant
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import Runnable, RunnableConfig


class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            result = self.runnable.invoke(state)

            if not result.tool_calls and (
                    not result.content
                    or isinstance(result.content, list)
                    and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}

            else:
                break
        return {"messages": result}


class CompleteOrEscalate(BaseModel):
    """A tool to mark the current task as completed and/or to escalate control of the dialog to the main assistant,
    who can re-route the dialog based on the user's needs."""

    cancel: bool = True
    reason: str

    class Config:
        schema_extra = {
            "example": {
                "cancel": True,
                "reason": "User changed their mind about the current task.",

            },
            "example 2": {
                "cancel": True,
                "reason": "I have fully completed the task.",
            },
            "example 3": {
                "cancel": False,
                "reason": "I need to search the user's emails or calendar for more information.",
            },

        }


# Flight booking assistant

flight_booking_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a specialized assistant for handling flight updates. "
            " The primary assistant delegates work to you whenever the user needs help updating their bookings. "
            "Confirm the updated flight details with the customer and inform them of any additional fees. "
            " When searching, be persistent. Expand your query bounds if the first search returns no results. "
            "If you need more information or the customer changes their mind, escalate the task back to the main assistant."
            " Remember that a booking isn't completed until after the relevant tool has successfully been used."
            "\n\nCurrent user flight information:\n\n{user_info}\n"
            "\nCurrent time: {time}."
            "\n\nIf the user needs help, and none of your tools are appropriate for it, then"
            ' "CompleteOrEscalate" the dialog to the host assistant. Do not waste the user\'s time. Do not make up invalid tools or functions.',

        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())

llm = ChatOpenAI(model="gpt-4o",
                 api_key="sk-GfWFJo25weXH3eUS0f08C3A842A241F29cE8BfA171643a00",
                 base_url="https://gtapi.xiaoerchaoren.com:8932/v1")

update_flight_safe_tools = [search_flights]
update_flight_sensitive_tools = [update_ticket_to_new_flight, cancel_ticket]
update_flight_tools = update_flight_safe_tools + update_flight_sensitive_tools
update_flight_runnable = flight_booking_prompt | llm.bind_tools(update_flight_tools + [CompleteOrEscalate])

# Hotel Booking Assistant
book_hotel_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are a specialized assistant for handling hotel bookings. "
         "The primary assistant delegates work to you whenever the user needs help booking a hotel. "
         "Search for available hotels based on the user's preferences and confirm the booking details with the customer. "
         " When searching, be persistent. Expand your query bounds if the first search returns no results. "
         "If you need more information or the customer changes their mind, escalate the task back to the main assistant."
         " Remember that a booking isn't completed until after the relevant tool has successfully been used."
         "\nCurrent time: {time}."
         '\n\nIf the user needs help, and none of your tools are appropriate for it, then "CompleteOrEscalate" the dialog to the host assistant.'
         " Do not waste the user's time. Do not make up invalid tools or functions."
         "\n\nSome examples for which you should CompleteOrEscalate:\n"
         " - 'what's the weather like this time of year?'\n"
         " - 'nevermind i think I'll book separately'\n"
         " - 'i need to figure out transportation while i'm there'\n"
         " - 'Oh wait i haven't booked my flight yet i'll do that first'\n"
         " - 'Hotel booking confirmed'",
         ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())

book_hotel_safe_tools = [search_hotels]
book_hotel_sensitive_tools = [book_hotel, update_hotel, cancel_hotel]
book_hotel_tools = book_hotel_safe_tools + book_hotel_sensitive_tools
book_hotel_runnable = book_hotel_prompt | llm.bind_tools(
    book_hotel_tools + [CompleteOrEscalate]
)

# Car Rental Assistant
book_car_rental_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are a specialized assistant for handling car rental bookings. "
         "The primary assistant delegates work to you whenever the user needs help booking a car rental. "
         "Search for available car rentals based on the user's preferences and confirm the booking details with the customer. "
         " When searching, be persistent. Expand your query bounds if the first search returns no results. "
         "If you need more information or the customer changes their mind, escalate the task back to the main assistant."
         " Remember that a booking isn't completed until after the relevant tool has successfully been used."
         "\nCurrent time: {time}."
         "\n\nIf the user needs help, and none of your tools are appropriate for it, then "
         '"CompleteOrEscalate" the dialog to the host assistant. Do not waste the user\'s time. Do not make up invalid tools or functions.'
         "\n\nSome examples for which you should CompleteOrEscalate:\n"
         " - 'what's the weather like this time of year?'\n"
         " - 'What flights are available?'\n"
         " - 'nevermind i think I'll book separately'\n"
         " - 'Oh wait i haven't booked my flight yet i'll do that first'\n"
         " - 'Car rental booking confirmed'",
         ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())

book_car_rental_safe_tools = [search_car_rentals]
book_car_rental_sensitive_tools = [
    book_car_rental,
    update_car_rental,
    cancel_car_rental,
]
book_car_rental_tools = book_car_rental_safe_tools + book_car_rental_sensitive_tools
book_car_rental_runnable = book_car_rental_prompt | llm.bind_tools(
    book_car_rental_tools + [CompleteOrEscalate]
)

# Excursion Assistant

book_excursion_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a specialized assistant for handling trip recommendations. "
            "The primary assistant delegates work to you whenever the user needs help booking a recommended trip. "
            "Search for available trip recommendations based on the user's preferences and confirm the booking details with the customer. "
            "If you need more information or the customer changes their mind, escalate the task back to the main assistant."
            " When searching, be persistent. Expand your query bounds if the first search returns no results. "
            " Remember that a booking isn't completed until after the relevant tool has successfully been used."
            "\nCurrent time: {time}."
            '\n\nIf the user needs help, and none of your tools are appropriate for it, then "CompleteOrEscalate" the dialog to the host assistant. Do not waste the user\'s time. Do not make up invalid tools or functions.'
            "\n\nSome examples for which you should CompleteOrEscalate:\n"
            " - 'nevermind i think I'll book separately'\n"
            " - 'i need to figure out transportation while i'm there'\n"
            " - 'Oh wait i haven't booked my flight yet i'll do that first'\n"
            " - 'Excursion booking confirmed!'",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())

book_excursion_safe_tools = [search_trip_recommendations]
book_excursion_sensitive_tools = [book_excursion, update_excursion, cancel_excursion]
book_excursion_tools = book_excursion_sensitive_tools + book_excursion_safe_tools
book_excursion_runnable = book_excursion_prompt | llm.bind_tools(book_excursion_tools + [CompleteOrEscalate])


# Primary Assistant
class ToFlightBookingAssistant(BaseModel):
    """Transfers work to a specialized assistant to handle flight updates and cancellations."""

    request: str = Field(
        description="Any necessary followup questions the update flight assistant should clarify before proceeding."
    )


class ToBookCarRental(BaseModel):
    """Transfers work to a specialized assistant to handle car rental bookings."""

    location: str = Field(
        description="The location where the user wants to rent a car."
    )
    start_date: str = Field(description="The start date of the car rental.")
    end_date: str = Field(description="the end date of the car rental.")
    requests: str = Field(
        description="Any additional information or requests from the user regarding the car rental"
    )

    class Config:
        schema_extra = {
            "example": {
                "location": "Basel",
                "start_date": "2023-07-01",
                "end_date": "2023-07-05",
                "request": "I need a compact car with automatic transmission.",
            }
        }


class ToHotelBookingAssistant(BaseModel):
    """Transfer work to a specialized assistant to handle hotel bookings."""

    location: str = Field(
        description=
        "The location where the user wants to book a hotel.")

    checkin_date: str = Field(description="The check-in date for the hotel.")
    checkout_date: str = Field(description="The check-out date for the hotel.")
    requests: str = Field(
        description="Any additional information or requests from the user regarding the hoel booking."
    )

    class Config:
        schema_extra = {
            "example": {
                "location": "Zurich",
                "checkin_date": "2023-08-15",
                "checkout_date": "2023-08-20",
                "request": "I prefer a hotel near the city center with a room that has a view.",
            }
        }


class ToBookExcursion(BaseModel):
    """Transfers work to a specialized assistant to handle trip recommendation and other excursion bookings."""

    location: str = Field(
        description="The location where the user wants to book a recommended trip."
    )

    request: str = Field(
        description="Any additional information or requests from the user regarding the trip recommendation."
    )

    class Config:
        schema_extra = {
            "example": {
                "location": "Lucerne",
                "request": "The user is interested in outdoor activities and scenic views.",
            }
        }


primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are a helpful customer support assistant for Swiss Airlines. "
         "Your primary role is to search for flight information and company policies to answer customer queries. "
         "If a customer requests to update or cancel a flight, book a car rental, book a hotel, or get trip recommendations, "
         "delegate the task to the appropriate specialized assistant by invoking the corresponding tool. You are not able to make these types of changes yourself."
         " Only the specialized assistants are given permission to do this for the user."
         "The user is not aware of the different specialized assistants, so do not mention them; just quietly delegate through function calls. "
         "Provide detailed information to the customer, and always double-check the database before concluding that information is unavailable. "
         " When searching, be persistent. Expand your query bounds if the first search returns no results. "
         " If a search comes up empty, expand your search before giving up."
         "\n\nCurrent user flight information:\n\n{user_info}\n"
         "\nCurrent time: {time}.",),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())

primary_assistant_tools = [
    TavilySearchResults(max_results=1),
    search_flights,
    lookup_policy,
]
assistant_runnable = primary_assistant_prompt | llm.bind_tools(
    primary_assistant_tools +
    [
        ToFlightBookingAssistant,
        ToBookCarRental,
        ToHotelBookingAssistant,
        ToBookExcursion,
    ]
)

# Create Assistants
'''We're about ready to create the graph. In the previous section, we made the design decision to have a shared messages state between all the nodes. This is powerful in that each delegated assistant can see the entire user journey and have a shared context. This, however, means that weaker LLMs can easily get mixed up about there specific scope. To mark the "handoff" between the primary assistant and one of the delegated workflows (and complete the tool call from the router), we will add a ToolMessage to the state.
'''

# Utility
'''Create a function to make an "entry" node for each workflow, stating "the current assistant ix assistant_name".'''
from typing import Callable
from langchain_core.messages import ToolMessage


def create_entry_node(assistant_name: str, new_dialog_state: str) -> Callable:
    def entry_node(state: State) -> dict:
        tool_call_id = state["messages"][-1].tool_calls[0]["id"]
        return {
            "messages": [
                ToolMessage(
                    content=f"The assistant is now the {assistant_name}. Reflect on the above conversation between the host assistant and the user."
                            f" The user's intent is unsatisfied. Use the provided tools to assist the user. Remember, you are {assistant_name},"
                            " and the booking, update, other other action is not complete until after you have successfully invoked the appropriate tool."
                            " If the user changes their mind or needs help for other tasks, call the CompleteOrEscalate function to let the primary host assistant take control."
                            " Do not mention who you are - just act as the proxy for the assistant.",

                    tool_call_id=tool_call_id,
                )
            ],
            "dialog_state": new_dialog_state,
        }

    return entry_node


# Denfine Graph
from typing import Literal
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph
from langgraph.prebuilt import tools_condition

builder = StateGraph(State)


def user_info(state: State):
    return {"user_info": fetch_user_flight_information.invoke({})}


builder.add_node("fetch_user_info", user_info)
builder.add_edge(START, "fetch_user_info")

# Flight booking assistant
builder.add_node(
    "enter_update_flight",
    create_entry_node("Flight Updates & Booking Assistant", "update_flight"),

)
builder.add_node("update_flight", Assistant(update_flight_runnable))
builder.add_edge("enter_update_flight", "update_flight")
builder.add_node("update_flight_sensitive_tools",
                 create_tool_node_with_fallback(update_flight_sensitive_tools),
                 )
builder.add_node("update_flight_safe_tools",
                 create_tool_node_with_fallback(update_flight_safe_tools),
                 )


def route_update_flight(
        state: State,
) -> Literal[
    "update_flight_sensitive_tools",
    "update_flight_safe_tools",
    "leave_skill",
    "__end__",
]:
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
    safe_toolnames = [t.name for t in update_flight_safe_tools]
    if all(tc["name"] in safe_toolnames for tc in tool_calls):
        return "update_flight_safe_tools"
    return "update_flight_sensitive_tools"


builder.add_edge("update_flight_sensitive_tools", "update_flight")
builder.add_edge("update_flight_safe_tools", "update_flight")
builder.add_conditional_edges("update_flight", route_update_flight)



def tool_looper(state: State) -> dict:
    """Loop through all tool calls from the primary assistant."""

    # 获取当前的工具调用堆栈，添加默认值以防止 KeyError
    tool_calls_stack = state.get("tool_calls_stack", [])

    messages = []
    if tool_calls_stack:
        # 处理下一个工具调用
        next_tool_call_id = tool_calls_stack.pop(0)
        messages.append(
            ToolMessage(
                content="Processing tool call...",
                tool_call_id=next_tool_call_id,
            )
        )

    return {
        "dialog_state": "tool_looper",
        "messages": messages,
        "tool_calls_stack": tool_calls_stack,
    }




# This node will be shared for exiting all specialized assistants
def pop_dialog_state(state: State) -> dict:
    """Pop the dialog stack and return to the main assistant."""

    messages = []
    if state["messages"][-1].tool_calls:
        tool_call_id = state["messages"][-1].tool_calls[0]["id"]
        # 获取并更新 tool_calls_stack
        tool_calls_stack = state.get("tool_calls_stack", [])
        tool_calls_stack.append(tool_call_id)

        messages.append(
            ToolMessage(
                content="Resuming dialog with the host assistant. Please reflect on the past conversation and assist the user as needed.",
                tool_call_id=tool_call_id,
            )
        )
    else:
        tool_calls_stack = state.get("tool_calls_stack", [])

    return {
        "dialog_state": "pop",
        "messages": messages,
        "tool_calls_stack": tool_calls_stack,
    }


builder.add_node("leave_skill", pop_dialog_state)
builder.add_edge("leave_skill", "primary_assistant")

# Car rental assistant
builder.add_node(
    "enter_book_car_rental",
    create_entry_node("Car Rental Assistant", "book_car_rental"),
)
builder.add_node("book_car_rental", Assistant(book_car_rental_runnable))
builder.add_edge("enter_book_car_rental", "book_car_rental")
builder.add_node(
    "book_car_rental_safe_tools",
    create_tool_node_with_fallback(book_car_rental_safe_tools),
)
builder.add_node(
    "book_car_rental_sensitive_tools",
    create_tool_node_with_fallback(book_car_rental_sensitive_tools),
)


def route_book_car_rental(
        state: State,
) -> Literal[
    "book_car_rental_safe_tools",
    "book_car_rental_sensitive_tools",
    "leave_skill",
    "__end__",
]:
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
    safe_toolnames = [t.name for t in book_car_rental_safe_tools]
    if all(tc["name"] in safe_toolnames for tc in tool_calls):
        return "book_car_rental_safe_tools"
    return "book_car_rental_sensitive_tools"


builder.add_edge("book_car_rental_sensitive_tools", "book_car_rental")
builder.add_edge("book_car_rental_safe_tools", "book_car_rental")
builder.add_conditional_edges("book_car_rental", route_book_car_rental)

# Hotel booking assistant
builder.add_node(
    "enter_book_hotel", create_entry_node("Hotel Booking Assistant", "book_hotel")
)
builder.add_node("book_hotel", Assistant(book_hotel_runnable))
builder.add_edge("enter_book_hotel", "book_hotel")
builder.add_node(
    "book_hotel_safe_tools",
    create_tool_node_with_fallback(book_hotel_safe_tools),
)
builder.add_node(
    "book_hotel_sensitive_tools",
    create_tool_node_with_fallback(book_hotel_sensitive_tools),
)


def route_book_hotel(
        state: State,
) -> Literal[
    "leave_skill", "book_hotel_safe_tools", "book_hotel_sensitive_tools", "__end__"
]:
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
    tool_names = [t.name for t in book_hotel_safe_tools]
    if all(tc["name"] in tool_names for tc in tool_calls):
        return "book_hotel_safe_tools"
    return "book_hotel_sensitive_tools"


builder.add_edge("book_hotel_sensitive_tools", "book_hotel")
builder.add_edge("book_hotel_safe_tools", "book_hotel")
builder.add_conditional_edges("book_hotel", route_book_hotel)

# Excursion assistant
builder.add_node(
    "enter_book_excursion",
    create_entry_node("Trip Recommendation Assistant", "book_excursion"),
)
builder.add_node("book_excursion", Assistant(book_excursion_runnable))
builder.add_edge("enter_book_excursion", "book_excursion")
builder.add_node(
    "book_excursion_safe_tools",
    create_tool_node_with_fallback(book_excursion_safe_tools),
)
builder.add_node(
    "book_excursion_sensitive_tools",
    create_tool_node_with_fallback(book_excursion_sensitive_tools),
)


def route_book_excursion(
        state: State,
) -> Literal[
    "book_excursion_safe_tools",
    "book_excursion_sensitive_tools",
    "leave_skill",
    "__end__",
]:
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
    tool_names = [t.name for t in book_excursion_safe_tools]
    if all(tc["name"] in tool_names for tc in tool_calls):
        return "book_excursion_safe_tools"
    return "book_excursion_sensitive_tools"


builder.add_edge("book_excursion_sensitive_tools", "book_excursion")
builder.add_edge("book_excursion_safe_tools", "book_excursion")
builder.add_conditional_edges("book_excursion", route_book_excursion)

# Primary assistant
builder.add_node("primary_assistant", Assistant(assistant_runnable))
builder.add_node(
    "primary_assistant_tools", create_tool_node_with_fallback(primary_assistant_tools)
)
builder.add_node("tool_looper", tool_looper)  # 添加 tool_looper 节点


def route_primary_assistant(
        state: State,
) -> Literal[
    "primary_assistant_tools",
    "enter_update_flight",
    "enter_book_hotel",
    "enter_book_excursion",
    "enter_book_car_rental",
    "__end__",
    "tool_looper"
]:
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    if tool_calls:
        # 如果有多个工具调用，将它们推送到工具调用栈中
        if len(tool_calls) > 1:
            tool_calls_stack = state.get("tool_calls_stack", [])
            for tool_call in tool_calls:
                tool_calls_stack.append(tool_call["id"])
            state["tool_calls_stack"] = tool_calls_stack
            return "tool_looper"

        # 处理单个工具调用的情况
        if tool_calls[0]["name"] == ToFlightBookingAssistant.__name__:
            return "enter_update_flight"
        elif tool_calls[0]["name"] == ToBookCarRental.__name__:
            return "enter_book_car_rental"
        elif tool_calls[0]["name"] == ToHotelBookingAssistant.__name__:
            return "enter_book_hotel"
        elif tool_calls[0]["name"] == ToBookExcursion.__name__:
            return "enter_book_excursion"
        return "primary_assistant_tools"
    raise ValueError("Invalid route")


# The assistant can route to one of the delegated assistants,
# directly use a tool, or directly respond to the user
builder.add_conditional_edges(
    "primary_assistant",
    route_primary_assistant,
    {
        "enter_update_flight": "enter_update_flight",
        "enter_book_car_rental": "enter_book_car_rental",
        "enter_book_hotel": "enter_book_hotel",
        "enter_book_excursion": "enter_book_excursion",
        "primary_assistant_tools": "primary_assistant_tools",
        "tool_looper": "tool_looper",  # 添加 tool_looper 边缘
        END: END,
    }
)
builder.add_edge("primary_assistant_tools", "primary_assistant")
builder.add_edge("tool_looper", "primary_assistant")  # 确保从 tool_looper 回到主助手


# Each delegated workflow can directly respond to the user
# When the user responds, we want to return to the currently active workflow
def route_to_workflow(
        state: State,
) -> Literal[
    "primary_assistant",
    "update_flight",
    "book_car_rental",
    "book_hotel",
    "book_excursion",
]:
    """If we are in a delegated state, route directly to the appropriate assistant."""
    dialog_state = state.get("dialog_state")
    if not dialog_state:
        return "primary_assistant"
    return dialog_state[-1]


builder.add_conditional_edges("fetch_user_info", route_to_workflow)

# Compile graph
memory = SqliteSaver.from_conn_string(":memory:")
part_4_graph = builder.compile(
    checkpointer=memory,
    interrupt_before=[
        "update_flight_sensitive_tools",
        "book_car_rental_sensitive_tools",
        "book_hotel_sensitive_tools",
        "book_excursion_sensitive_tools",

    ],
)

try:
    # 获取图形的 PNG 数据
    png_data = part_4_graph.get_graph(xray=True).draw_mermaid_png()

    # 指定文件名
    file_name = "graph.png"

    # 将 PNG 数据写入文件
    with open(file_name, "wb") as file:
        file.write(png_data)

    print(f"Graph saved as {file_name}")
except Exception as e:
    # 处理异常，例如输出错误信息或采取其他恰当的措施
    print(f"Failed to display graph: {e}")

import shutil
import uuid

shutil.copy(backup_file, db)
thread_id = str(uuid.uuid4())

config = {
    "configurable": {
        "passenger_id": "3442 587242",
        "thread_id": thread_id,
    }
}

_printed = set()

for question in tutorial_questions:
    events = part_4_graph.stream(
        {"messages": ("user", question)}, config, stream_mode="values"
    )
    for event in events:
        print("————————————————")
        print(f"Event:{event}")
        _print_event(event, _printed)

    snapshot = part_4_graph.get_state(config)
    while snapshot.next:

        user_input = input(
            "Do you approve of the above actions? Type 'y' to continue;"
            " otherwise, explain your requested changed.\n\n"
        )
        if user_input.strip() == "y":
            result = part_4_graph.invoke(
                None,
                config,
            )
        else:
            result = part_4_graph.invoke(
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
        snapshot = part_4_graph.get_state(config)
