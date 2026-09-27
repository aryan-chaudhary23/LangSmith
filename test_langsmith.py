import os
from dotenv import load_dotenv

load_dotenv(override=True)

from langchain_core.runnables import RunnableLambda
from langchain_core.tracers import LangChainTracer


def hello(x):
    return f"Hello {x}"


chain = RunnableLambda(hello)

tracer = LangChainTracer(
    project_name="langchain-demo"
)

result = chain.invoke(
    "Aryan",
    config={
        "callbacks": [tracer]
    }
)

print(result)