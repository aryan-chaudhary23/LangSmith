import os
import requests

from dotenv import load_dotenv

from langchain_huggingface import (
    HuggingFaceEndpoint,
    ChatHuggingFace
)

from langchain_core.tools import tool

from langchain_community.tools import DuckDuckGoSearchRun

from langchain.agents import create_agent


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not HF_TOKEN:
    raise ValueError(
        "HUGGINGFACEHUB_API_TOKEN is not set."
    )


# ============================================================
# 1. SEARCH TOOL
# ============================================================

search_tool = DuckDuckGoSearchRun()


# ============================================================
# 2. WEATHER TOOL
# ============================================================

@tool
def get_weather_data(city: str) -> str:
    """
    Fetch the current weather data for a given city.
    """

    url = (
        "https://api.weatherstack.com/current"
        f"?access_key=f07d9636974c4120025fadf60678771b"
        f"&query={city}"
    )

    response = requests.get(url)

    response.raise_for_status()

    return response.text


# ============================================================
# 3. HUGGING FACE LLM
# ============================================================

llm_backend = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    huggingfacehub_api_token=HF_TOKEN,
    temperature=0.2,
    max_new_tokens=1024,
    task="conversational"
)

llm = ChatHuggingFace(
    llm=llm_backend
)


# ============================================================
# 4. TOOLS
# ============================================================

tools = [
    search_tool,
    get_weather_data
]


# ============================================================
# 5. CREATE AGENT
# ============================================================

agent = create_agent(
    model=llm,
    tools=tools,

    system_prompt="""
    You are a helpful assistant.

    You have access to two tools:

    1. DuckDuckGo search:
       Use it when you need current or web-based information.

    2. get_weather_data:
       Use it when the user asks for current weather or temperature
       of a city.

    Decide yourself which tool is appropriate.
    """
)


# ============================================================
# 6. RUN AGENT
# ============================================================

response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What is the current temperature of Gurgaon?"
            }
        ]
    }
)


# ============================================================
# 7. PRINT RESPONSE
# ============================================================

print("\nFull response:")
print(response)


print("\nFinal answer:")
print(response["messages"][-1].content)