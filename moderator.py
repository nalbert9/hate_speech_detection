from typing import List, Optional, Tuple, Dict, Any

from langchain.chains.constitutional_ai.models import ConstitutionalPrinciple
from langchain.chains.constitutional_ai.prompts import (
    CRITIQUE_PROMPT,
    REVISION_PROMPT,
)
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from typing_extensions import Annotated, TypedDict
import json
from pydantic import BaseModel, Field

# Initialize LLM with temperature control for more consistent results
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)


class Critique(BaseModel):
    """Generate a critique, if needed."""

    critique_needed: bool = Field(..., description="Whether or not a critique is needed.")
    critique: str = Field(..., description="If needed, the critique.")


# More specific prompts to guide the LLM's behavior
critique_prompt = PromptTemplate.from_template(
    "You are a content moderator evaluating the following response to a user query based on a given constitutional principle.\n\n"
    "Query: {query}\n\n"
    "Response: {response}\n\n"
    "Constitutional Principle: {critique_request}\n\n"
    "1. Determine if the response violates the constitutional principle. Consider aspects like safety, fairness, bias, and accuracy.\n"
    "2. If the response violates the principle, provide a detailed critique explaining the violation.\n"
    "3. If the response does not violate the principle, state clearly 'No critique needed.'\n\n"
    "Critique: Provide your answer as a JSON object with keys 'critique_needed' (boolean) and 'critique' (string)."
)

revision_prompt = PromptTemplate.from_template(
    "You are a content editor revising a response based on a critique and a constitutional principle.\n\n"
    "Query: {query}\n\n"
    "Original Response: {response}\n\n"
    "Constitutional Principle: {critique_request}\n\n"
    "Critique: {critique}\n\n"
    "Revision Request: {revision_request}\n\n"
    "Based on the critique and the constitutional principle, revise the original response to better align with ethical AI principles.\n"
    "If the critique indicates no issues or the revision request is redundant, return the original response without changes. "
    "Otherwise, provide a revised response that addresses the critique while still answering the query accurately and completely.\n\n"
    "Revised Response:"
)


class PydanticOutputParser(JsonOutputParser):
    """Output parser that uses Pydantic to parse the output."""

    def __init__(self, pydantic_object: Any) -> None:
        """Initialize with a pydantic object."""
        self.pydantic_object = pydantic_object

    def parse(self, text: str) -> Any:
        """Parse the output of an LLM call."""
        try:
            json_object = json.loads(text)
            return self.pydantic_object(**json_object)
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e}. Returning a default object.")
            return self.pydantic_object(critique_needed=False, critique="No critique provided due to parsing error.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}. Returning a default object.")
            return self.pydantic_object(critique_needed=False, critique="No critique provided due to an error.")


chain = llm | StrOutputParser()
critique_chain = critique_prompt | llm | PydanticOutputParser(pydantic_object=Critique)
revision_chain = revision_prompt | llm | StrOutputParser()


class State(TypedDict):
    query: str
    constitutional_principles: List[ConstitutionalPrinciple]
    initial_response: str
    critiques_and_revisions: List[Tuple[str, str]]
    response: str


async def generate_response(state: State):
    """Generate initial response."""
    response = await chain.ainvoke({"query": state["query"]})
    return {"response": response, "initial_response": response}


async def critique_and_revise(state: State):
    """Critique and revise response according to principles."""
    critiques_and_revisions = []
    response = state["response"]  # Use the current response for iterative refinement
    for principle in state["constitutional_principles"]:
        # Invoke critique chain
        critique_input = {
            "query": state["query"],
            "response": response,
            "critique_request": principle.critique_request,
        }
        print(f"Critique Input: {critique_input}")  # Print the input to the critique chain

        critique_output = await critique_chain.ainvoke(critique_input)

        print(f"Critique Output: {critique_output}")  # Print the output of the critique chain

        critique_needed = critique_output.critique_needed
        critique = critique_output.critique

        # Revise only if a critique is provided and critique_needed is True
        if critique_needed and "No critique needed" not in critique:
            revision = await revision_chain.ainvoke(
                {
                    "query": state["query"],
                    "response": response,
                    "critique_request": principle.critique_request,
                    "critique": critique,
                    "revision_request": principle.revision_request,
                }
            )
            response = revision  # Update response with the revised version
            critiques_and_revisions.append((critique, revision))
        else:
            critiques_and_revisions.append((critique, ""))

    return {
        "critiques_and_revisions": critiques_and_revisions,
        "response": response,
    }


graph = StateGraph(State)
graph.add_node("generate_response", generate_response)
graph.add_node("critique_and_revise", critique_and_revise)

graph.add_edge(START, "generate_response")
graph.add_edge("generate_response", "critique_and_revise")
graph.add_edge("critique_and_revise", END)
app = graph.compile()
# Define more meaningful constitutional principles
constitutional_principles = [
    ConstitutionalPrinciple(
        critique_request="The response should be helpful and informative.",
        revision_request="Provide a more detailed and helpful answer.",
    ),
    ConstitutionalPrinciple(
        critique_request="The response should be free of harmful or unethical content.",
        revision_request="Remove any harmful or unethical content from the response.",
    ),
    ConstitutionalPrinciple(
        critique_request="The response should be unbiased and fair.",
        revision_request="Ensure the response is unbiased and fair.",
    ),
]

query = "What is the meaning of life? Answer in 10 words or fewer."

async def run_app(query: str, constitutional_principles: List[ConstitutionalPrinciple]):
    """Runs the LangGraph app and returns the final state in JSON format."""
    final_state = {}
    async for step in app.astream(
        {"query": query, "constitutional_principles": constitutional_principles, "response": ""},
        stream_mode="values",
    ):
        final_state.update(step)

    # Prepare the final output in JSON format
    output = {
        "initial_response": final_state.get("initial_response", ""),
        "critiques_and_revisions": final_state.get("critiques_and_revisions", []),
        "response": final_state.get("response", ""),
    }
    return json.dumps(output, indent=2)


if __name__ == "__main__":
    import asyncio

    result_json = asyncio.run(run_app(query, constitutional_principles))
    print(result_json)
