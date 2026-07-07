import streamlit as st
import requests
import json
from dotenv import load_dotenv

from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_ollama import ChatOllama

load_dotenv()



st.set_page_config(
    page_title="QuickBite AI",
    page_icon="🍔",
    layout="centered"
)

st.title("🍔 QuickBite AI")
st.caption("AI Powered Food Delivery Assistant")



with st.sidebar:

    st.header("User Details")

    st.text_input(
        "Delivery Address",
        key="address"
    )

    st.selectbox(
        "Dietary Preference",
        ["Veg", "Non-Veg", "Any"],
        key="dietary_preference"
    )

    st.number_input(
        "Distance (KM)",
        min_value=1.0,
        value=5.0,
        key="distance_km"
    )

    st.number_input(
        "Items",
        min_value=1,
        value=2,
        key="item_count"
    )

    st.selectbox(
        "Weather",
        ["No Rain", "Rain"],
        key="rain_flag"
    )

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()


if "messages" not in st.session_state:
    st.session_state.messages = []



def load_menu():
    with open("menu.json", "r", encoding="utf-8") as f:
        return json.load(f)


def search_menu(query, preference):

    menu = load_menu()

    query = query.lower()

    if preference == "Veg":
        menu = [m for m in menu if m["veg"]]

    elif preference == "Non-Veg":
        menu = [m for m in menu if not m["veg"]]

    cuisines = [
        "indian",
        "italian",
        "chinese",
        "south indian",
        "american"
    ]

    for c in cuisines:
        if c in query:
            menu = [
                m for m in menu
                if m["cuisine"].lower() == c
            ]

    return menu[:3]


@tool
def get_delivery_estimate(query: str) -> str:
    """
    Use this tool whenever user asks:

    - delivery time
    - ETA
    - order status
    - when food will arrive
    - how long delivery takes
    """

    payload = {
        "distance_km": st.session_state["distance_km"],
        "item_count": st.session_state["item_count"],
        "rain_flag": 1 if st.session_state["rain_flag"] == "Rain" else 0
    }

    try:

        response = requests.post(
            "http://127.0.0.1:5000/predict",
            json=payload,
            timeout=5
        )

        result = response.json()

        return f"Estimated delivery time is {result['delivery_time_min']} minutes."

    except Exception:

        return "Delivery prediction server is unavailable."


llm = ChatOllama(
    model="mistral",
    temperature=0
)

agent = create_agent(
    model=llm,
    tools=[get_delivery_estimate]
)

def build_prompt(menu):

    if menu:

        menu_text = "\n".join(

            [
                f"- {m['name']} | ₹{m['price']} | {m['cuisine']} | {'Veg' if m['veg'] else 'Non-Veg'}"
                for m in menu
            ]

        )

    else:

        menu_text = "No exact menu found."

    return f"""
You are QuickBite AI.

Customer Details

Address:
{st.session_state.get("address")}

Preference:
{st.session_state.get("dietary_preference")}

Menu

{menu_text}

Rules

1. Recommend menu items.
2. Keep answers short.
3. Be friendly.
4. If user asks about delivery time, ETA,
arrival time or order status,
ALWAYS use get_delivery_estimate tool.
5. Never guess delivery time yourself.
"""


for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


user_input = st.chat_input("Ask QuickBite AI...")

if user_input:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    menu = search_menu(
        user_input,
        st.session_state["dietary_preference"]
    )

    system_prompt = build_prompt(menu)

    messages = [
        SystemMessage(content=system_prompt)
    ]

    for msg in st.session_state.messages:

        if msg["role"] == "user":
            messages.append(
                HumanMessage(content=msg["content"])
            )

        else:
            messages.append(
                AIMessage(content=msg["content"])
            )

    response = agent.invoke(
        {
            "messages": messages
        }
    )

    answer = response["messages"][-1].content

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    st.rerun()