import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_huggingface import (
    HuggingFaceEndpoint,
    ChatHuggingFace,
    HuggingFaceEndpointEmbeddings
)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tracers import LangChainTracer


load_dotenv()


# ============================================================
# 1. EMBEDDING MODEL
# ============================================================

embeddings = HuggingFaceEndpointEmbeddings(
    model="BAAI/bge-small-en-v1.5",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    task="feature-extraction"
)


# ============================================================
# 2. LLM
# ============================================================

llm_backend = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    temperature=0.2,
    max_new_tokens=1024,
    task="conversational"
)

llm = ChatHuggingFace(llm=llm_backend)


# ============================================================
# 3. LOAD PDF
# ============================================================

PDF_PATH = "islr.pdf"

loader = PyPDFLoader(PDF_PATH)

docs = loader.load()

print(f"Loaded {len(docs)} pages")


# ============================================================
# 4. SPLIT DOCUMENT
# ============================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

splits = splitter.split_documents(docs)

print(f"Created {len(splits)} chunks")


# ============================================================
# 5. CREATE VECTOR STORE
# ============================================================

vector_store = FAISS.from_documents(
    splits,
    embeddings
)

tracer = LangChainTracer(
    project_name="RAG-demo"
)
counter = 1
config = {
    "callbacks": [tracer],
    "tags": ["RAG", "test", "langchain", "langsmith"],
    "metadata": {
        "description": "This is a test run for RAG calls using LangChain and LangSmith.",
        "author": "Aryan",
        "model": "openai/gpt-oss-20b",
        "task": "conversational"
    },
    "run_name": f"RAG-test-{counter}"
}

print("Vector store created")


# ============================================================
# 6. RETRIEVER
# ============================================================

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)


# ============================================================
# 7. PROMPT
# ============================================================

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        Answer ONLY from the provided context.

        If the answer cannot be found in the context,
        say that you don't know.

        Do not use outside knowledge.
        """
    ),
    (
        "human",
        """
        Question:
        {question}

        Context:
        {context}
        """
    )
])


# ============================================================
# 8. FORMAT DOCUMENTS
# ============================================================

def format_docs(docs):
    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


# ============================================================
# 9. RAG CHAIN
# ============================================================

parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

chain = (
    parallel
    | prompt
    | llm
    | StrOutputParser()
)


# ============================================================
# 10. ASK QUESTIONS
# ============================================================

print("\nPDF RAG ready.")
print("Type 'exit' to quit.")

while True:

    question = input("\nQ: ").strip()

    if question.lower() == "exit":
        break

    answer = chain.invoke(question, config=config)
    counter += 1

    print("\nA:", answer)