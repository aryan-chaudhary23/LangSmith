import os
from dotenv import load_dotenv

load_dotenv(override=True)

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tracers import LangChainTracer


# -------------------------
# Hugging Face model
# -------------------------

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="conversational",
    temperature=0.3,
    max_new_tokens=512,
)

model = ChatHuggingFace(llm=llm)


# -------------------------
# Prompt + parser
# -------------------------

prompt = PromptTemplate.from_template("{question}")

parser = StrOutputParser()

chain = prompt | model | parser


# -------------------------
# LangSmith tracer
# -------------------------

tracer = LangChainTracer(
    project_name="langchain-demo"
)


# -------------------------
# Run chain
# -------------------------

result = chain.invoke(
    {
        "question": "What is the capital of India?"
    },
    config={
        "callbacks": [tracer]
    }
)

print(result)