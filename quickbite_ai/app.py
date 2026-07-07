import streamlit as st
import requests
import json
from langchain_community.llms import Ollama
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv()

st.set_page_config(page_title="QuickBite AI", layout="centered")
st.title("🍔 QuickBite AI")
st.caption("Intelligent Food Delivery Assistant")

with st.sidebar:
    st.header("User Details")
    address = st.text_input("Delivery Address", key="address")
    dietary_preference = st.selectbox("Dietary Preference", ["Veg", "Non-Veg", "Any"], key="dietary_preference")
    distance_km = st.number_input("Distance in KM", min_value=1.0, value=5.0, key="distance_km")
    item_count = st.number_input("Item Count", min_value=1, value=2, key="item_count")
    rain_flag = st.selectbox("Weather", ["No Rain", "Rain"], key="rain_flag")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

def load_menu():
    with open("menu.json", "r", encoding="utf-8") as file:
        return json.load(file)

def search_menu(user_query, preference):
    menu = load_menu()
    query = user_query.lower()

    if preference == "Veg":
        menu = [item for item in menu if item["veg"] is True]
    elif preference == "Non-Veg":
        menu = [item for item in menu if item["veg"] is False]

    cuisine_keywords = ["indian", "italian", "chinese", "south indian", "american"]
    for cuisine in cuisine_keywords:
        if cuisine in query:
            menu = [item for item in menu if item["cuisine"].lower() == cuisine]

    return menu[:3]

@tool
def get_delivery_estimate(query: str) -> str:
    """
    Use this tool when user asks delivery time, order arrival time,
    or how long the food will take.
    """
    payload = {
        "distance_km": st.session_state.get("distance_km", 5.0),
        "item_count": st.session_state.get("item_count", 2),
        "rain_flag": 1 if st.session_state.get("rain_flag", "No Rain") == "Rain" else 0
    }

    try:
        response = requests.post("http://127.0.0.1:5000/predict", json=payload, timeout=5)
        response.raise_for_status()
        result = response.json()
        minutes = result["delivery_time_min"]
        return f"Estimated delivery time is around {minutes} minutes."
    except Exception as e:
        return f"Sorry, delivery estimate service is not available. Error: {e}"

llm = ChatOllama(
    model="mistral",
    temperature=0
)
agent = create_agent(model=llm, tools=[get_delivery_estimate])

def build_system_prompt(menu_results):
    if menu_results:
        menu_text = "\n".join([
            f"- {item['name']} | {item['cuisine']} | ₹{item['price']} | {'Veg' if item['veg'] else 'Non-Veg'}"
            for item in menu_results
        ])
    else:
        menu_text = "No exact menu match found. Suggest nearby popular items politely."

    return f"""
You are QuickBite AI, a friendly food delivery assistant.

User context:
Delivery address: {st.session_state.get('address', '')}
Dietary preference: {st.session_state.get('dietary_preference', 'Any')}

Top matching menu items:
{menu_text}

Rules:
- Keep answers short and helpful.
- Recommend food based on user preference.
- If user asks delivery time, use get_delivery_estimate tool.
- Never ask address again if already given.
- Use friendly Indian food delivery tone.

Few-shot examples:

Example 1:
User: I want something spicy Indian.
Assistant: Sure! Based on your preference, I recommend Paneer Butter Masala or Veg Biryani. Both are tasty Indian options.

Example 2:
User: How long will my order take?
Assistant: Let me check your delivery estimate. Your order should arrive soon with an estimated time from our delivery model.
"""

for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(f"<div style='text-align:right'>{msg['content']}</div>", unsafe_allow_html=True)
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])

user_input = st.chat_input("Ask QuickBite AI...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    menu_results = search_menu(user_input, st.session_state.get("dietary_preference", "Any"))
    system_prompt = build_system_prompt(menu_results)

    langchain_messages = [SystemMessage(content=system_prompt)]
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            langchain_messages.append(HumanMessage(content=msg["content"]))
        else:
            langchain_messages.append(AIMessage(content=msg["content"]))

    response = agent.invoke({"messages": langchain_messages})
    assistant_reply = response["messages"][-1].content

    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
    st.rerun()
