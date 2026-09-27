from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.tracers import LangChainTracer

load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="conversational",
    temperature=0.3,
    max_new_tokens=512,
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

tracer = LangChainTracer(
    project_name="langchain-demo-1"
)

config = {
    "callbacks": [tracer],
    "tags": ["sequential-llm", "test", "langchain", "langsmith"],
    "metadata": {
        "description": "This is a test run for sequential LLM calls using LangChain and LangSmith.",
        "author": "Aryan",
        "model": "openai/gpt-oss-20b",
        "task": "conversational"
    },
    "run_name": "sequential-llm-test"
}

result = chain.invoke({'topic': 'Unemployment in US'}, config=config)

print(result)