# pip install -U langgraph langchain-huggingface pydantic python-dotenv langsmith

import operator
import os

from typing import TypedDict, Annotated, List

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langsmith import traceable

from langchain_huggingface import (
    HuggingFaceEndpoint,
    ChatHuggingFace
)
from langchain_core.tracers import LangChainTracer

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from langgraph.graph import StateGraph, START, END


# ============================================================
# Setup
# ============================================================

load_dotenv()

print("TRACING =", os.getenv("LANGSMITH_TRACING"))
print("PROJECT =", os.getenv("LANGSMITH_PROJECT"))
print("ENDPOINT =", os.getenv("LANGSMITH_ENDPOINT"))
print("API KEY =", "SET" if os.getenv("LANGSMITH_API_KEY") else "NOT SET")


# ============================================================
# Hugging Face model
# ============================================================

llm_backend = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    temperature=0.2,
    max_new_tokens=1024,
    task="conversational"
)

model = ChatHuggingFace(llm=llm_backend)


# ============================================================
# Structured schema & parser
# ============================================================

class EvaluationSchema(BaseModel):

    feedback: str = Field(
        description="Detailed feedback for the essay"
    )

    score: int = Field(
        description="Score out of 10",
        ge=0,
        le=10
    )


# IMPORTANT:
# Do NOT use model.with_structured_output()
# PydanticOutputParser is used instead.

parser = PydanticOutputParser(
    pydantic_object=EvaluationSchema
)


# ============================================================
# Sample essay
# ============================================================

essay2 = """India and AI Time

Now world change very fast because new tech call Artificial Intel… something (AI). India also want become big in this AI thing. If work hard, India can go top. But if no careful, India go back.

India have many good. We have smart student, many engine-ear, and good IT peoples. Big company like TCS, Infosys, Wipro already use AI. Government also do program “AI for All”. It want AI in farm, doctor place, school and transport.

In farm, AI help farmer know when to put seed, when rain come, how stop bug. In health, AI help doctor see sick early. In school, AI help student learn good. Government office use AI to find bad people and work fast.

But problem come also. First is many villager no have phone or internet. So AI not help them. Second, many people lose job because AI and machine do work. Poor people get more bad.

One more big problem is privacy. AI need big big data. Who take care? India still make data rule. If no strong rule, AI do bad.

India must all people together – govern, school, company and normal people. We teach AI and make sure AI not bad. Also talk to other country and learn from them.

If India use AI good way, we become strong, help poor and make better life. But if only rich use AI, and poor no get, then big bad thing happen.

So, in short, AI time in India have many hope and many danger. We must go right road. AI must help all people, not only some. Then India grow big and world say "good job India".
"""


# ============================================================
# LangGraph state
# ============================================================

class UPSCState(TypedDict, total=False):

    essay: str

    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str
    overall_feedback: str

    # operator.add combines scores returned
    # by the three parallel branches
    individual_scores: Annotated[
        List[int],
        operator.add
    ]

    avg_score: float


# ============================================================
# Language evaluation
# ============================================================

@traceable(
    name="evaluate_language",
    tags=["dimension:language"],
    metadata={
        "dimension": "language"
    }
)
def evaluate_language(state: UPSCState):

    prompt = PromptTemplate(
        template="""
Evaluate the language quality of the following essay.

Focus on:
- Grammar
- Vocabulary
- Sentence construction
- Word choice
- Overall language quality

Provide detailed feedback and assign a score out of 10.

{format_instructions}

Essay:
{essay}
""",
        input_variables=["essay"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        },
    )

    formatted_prompt = prompt.format(
        essay=state["essay"]
    )

    response = model.invoke(formatted_prompt)

    output = parser.parse(response.content)

    return {
        "language_feedback": output.feedback,
        "individual_scores": [output.score]
    }


# ============================================================
# Analysis evaluation
# ============================================================

@traceable(
    name="evaluate_analysis",
    tags=["dimension:analysis"],
    metadata={
        "dimension": "analysis"
    }
)
def evaluate_analysis(state: UPSCState):

    prompt = PromptTemplate(
        template="""
Evaluate the depth of analysis in the following essay.

Focus on:
- Quality of arguments
- Depth of reasoning
- Examples
- Cause and effect
- Balance of arguments
- Understanding of the topic

Provide detailed feedback and assign a score out of 10.

{format_instructions}

Essay:
{essay}
""",
        input_variables=["essay"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        },
    )

    formatted_prompt = prompt.format(
        essay=state["essay"]
    )

    response = model.invoke(formatted_prompt)

    output = parser.parse(response.content)

    return {
        "analysis_feedback": output.feedback,
        "individual_scores": [output.score]
    }


# ============================================================
# Clarity of thought evaluation
# ============================================================

@traceable(
    name="evaluate_thought",
    tags=["dimension:clarity"],
    metadata={
        "dimension": "clarity_of_thought"
    }
)
def evaluate_thought(state: UPSCState):

    prompt = PromptTemplate(
        template="""
Evaluate the clarity of thought in the following essay.

Focus on:
- Logical flow
- Organization
- Coherence
- Clarity of ideas
- Connection between arguments
- Quality of conclusion

Provide detailed feedback and assign a score out of 10.

{format_instructions}

Essay:
{essay}
""",
        input_variables=["essay"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        },
    )

    formatted_prompt = prompt.format(
        essay=state["essay"]
    )

    response = model.invoke(formatted_prompt)

    output = parser.parse(response.content)

    return {
        "clarity_feedback": output.feedback,
        "individual_scores": [output.score]
    }


# ============================================================
# Final evaluation
# ============================================================

@traceable(
    name="final_evaluation-1",
    tags=["aggregate"],
    metadata={
        "stage": "final_evaluation"
    }
)
def final_evaluation(state: UPSCState):

    prompt = PromptTemplate(
        template="""
Based on the following three evaluations of the essay,
create a summarized overall feedback.

Language feedback:
{language_feedback}

Depth of analysis feedback:
{analysis_feedback}

Clarity of thought feedback:
{clarity_feedback}

Provide:
1. Overall strengths
2. Major weaknesses
3. Most important improvements

{format_instructions}
""",
        input_variables=[
            "language_feedback",
            "analysis_feedback",
            "clarity_feedback"
        ],
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        },
    )

    formatted_prompt = prompt.format(
        language_feedback=state.get(
            "language_feedback",
            ""
        ),
        analysis_feedback=state.get(
            "analysis_feedback",
            ""
        ),
        clarity_feedback=state.get(
            "clarity_feedback",
            ""
        )
    )

    response = model.invoke(formatted_prompt)

    output = parser.parse(response.content)

    scores = state.get(
        "individual_scores",
        []
    ) or []

    avg_score = (
        sum(scores) / len(scores)
        if scores
        else 0.0
    )

    return {
        "overall_feedback": output.feedback,
        "avg_score": avg_score
    }


# ============================================================
# Build LangGraph
# ============================================================

graph = StateGraph(UPSCState)

graph.add_node(
    "evaluate_language",
    evaluate_language
)

graph.add_node(
    "evaluate_analysis",
    evaluate_analysis
)

graph.add_node(
    "evaluate_thought",
    evaluate_thought
)

graph.add_node(
    "final_evaluation",
    final_evaluation
)


# ============================================================
# Fan-out
# ============================================================

graph.add_edge(
    START,
    "evaluate_language"
)

graph.add_edge(
    START,
    "evaluate_analysis"
)

graph.add_edge(
    START,
    "evaluate_thought"
)


# ============================================================
# Fan-in
# ============================================================

graph.add_edge(
    "evaluate_language",
    "final_evaluation"
)

graph.add_edge(
    "evaluate_analysis",
    "final_evaluation"
)

graph.add_edge(
    "evaluate_thought",
    "final_evaluation"
)

graph.add_edge(
    "final_evaluation",
    END
)


# ============================================================
# Compile
# ============================================================

workflow = graph.compile()


# ============================================================
# Run
# ============================================================

tracer = LangChainTracer(
    project_name="langgraph-demo-1"
)

if __name__ == "__main__":

    result = workflow.invoke(
        {"essay": essay2},

        config={
    "run_name": "evaluate_upsc_essay",
    "tags": [
        "essay",
        "langgraph",
        "evaluation"
    ],
    "metadata": {
        "essay_length": len(essay2),
        "model": "openai/gpt-oss-20b",
        "dimensions": [
            "language",
            "analysis",
            "clarity"
        ]
    },
    "callbacks": [tracer]
}
    )

    print("\n========================================")
    print("         EVALUATION RESULTS")
    print("========================================")

    print("\nLanguage feedback:")
    print(
        result.get(
            "language_feedback",
            ""
        )
    )

    print("\nAnalysis feedback:")
    print(
        result.get(
            "analysis_feedback",
            ""
        )
    )

    print("\nClarity feedback:")
    print(
        result.get(
            "clarity_feedback",
            ""
        )
    )

    print("\nOverall feedback:")
    print(
        result.get(
            "overall_feedback",
            ""
        )
    )

    print("\nIndividual scores:")
    print(
        result.get(
            "individual_scores",
            []
        )
    )

    print("\nAverage score:")
    print(
        result.get(
            "avg_score",
            0.0
        )
    )

    print("\n========================================")
    print("========================================")