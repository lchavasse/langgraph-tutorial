from typing import Literal
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

from langgraph.graph import StateGraph, START, END

from langchain_anthropic import ChatAnthropic

from langchain.prompts import PromptTemplate
from langgraph.types import Command

from dotenv import load_dotenv
load_dotenv()

# Import the graph for the basic RAG agent - this can be invoked as the "rag" agent node.
from basic_rag_agent import rag_graph

memory = MemorySaver()

class State(TypedDict):
    user_messages: Annotated[list, add_messages] # The conversation between the user and the assistant
    ai_messages: Annotated[list, add_messages] # The conversation between the assistant and the worker
    query: str # The query to the worker
    task_completed: bool = False # Whether the worker has completed the task

# Create the LLM
llm = ChatAnthropic(model="claude-3-5-haiku-latest")

# Define the workers accessible to the supervisor agent.
members = ["rag"]
# Include additional termination options.
options = members + ["ANSWER", "FINISH"]

# Define the structure of the supervisor agent's output.
class SupervisorRouter(TypedDict):
    user_message: str
    query: str
    next: Literal[*options]

# Define the supervisor agent.
def supervisor_agent(state: State) -> Command[Literal[*members, "ANSWER", "__end__"]]:
    user_messages = state["user_messages"]
    # Preprocess user_messages to create a string of all message contents
    user_messages_content = "\n".join([message.content for message in user_messages])

    ai_messages = state["ai_messages"]
    agent_response = "No response yet."
    # If the task is completed, get the agent response.
    task_completed = state.get("task_completed", False) # not quite sure why we need the .get here...
    if task_completed:
        agent_response = ai_messages[-1].content

    # Define the system prompt for the supervisor agent.
    sys_prompt = PromptTemplate(
        template="""You are a helpful assistant, who receives a user query.
        You should have a fluid conversation with the user. You should attempt to answer the user query yourself first. If it is vague, you should ask for more information.
        Once you assemble a full understanding of a required task that you feel you cannot answer, you should delegate the task to a worker.
        You also have a team of worker agents who can help you.
        You have the following workers: {members} \n\n
        You should delegate the task to a worker if you cannot answer the user query yourself or if you are asked for more information.
        You do not need to use a worker if you can answer the user query yourself.
        
        If you answer the query yourself:
        - Respond to the user as "user_message".
        - Pass "FINISH" to the "next" field.

        If you wish to use a worker to:
        - Announce what you are doing to the user as "user_message".
        - Pass a refined query to the worker as "query".
        - Specify which worker to use with the "next" field. \n\n
        The worker will return their results to you in order to communicate with the user.
        
        If the task is completed: {task_completed}, then the worker response is: \n\n {agent_response}
        You should decide if you are happy with the response. \n\n

        If you are happy with the response, you should do the following:
        - Give a one sentence summary of the response to the user as "user_message".
        - Respond with "ANSWER" to the "next" field. (this will pass the conversation to the writer to respond to the user in full)

        If you are not happy with the response, you should amend your query to the agent(s) and call them again.

        The usermessages are as follows:
        {user_messages}
    """,
        input_variables=["members", "user_messages", "task_completed", "agent_response"],
    )

    chain = sys_prompt | llm.with_structured_output(SupervisorRouter)

    response = chain.invoke({"members": members, "user_messages": user_messages_content, "task_completed": task_completed, "agent_response": agent_response})

    goto = response["next"]

    if goto == "FINISH":
        goto = "__end__" # end the conversation

    return Command(
        update={
            "user_messages": [HumanMessage(content = response["user_message"], name = "supervisor")],
            "query": response["query"],
            "task_completed": False
        },
        goto=goto,
    )

# Define the node to invoke the RAG agent.
def rag_agent(state: State) -> Command[Literal["supervisor"]]:
    query = state["query"]
    response = rag_graph.invoke({"messages": [HumanMessage(content=query)]})

    return Command(
        update={
            "ai_messages": [HumanMessage(content=response["messages"][-1].content, name="rag")],
            "task_completed": True
        },
        goto="supervisor",
    )

# Define the node to give a full response to the user.
def final_answer(state: State) -> Command[Literal["__end__"]]:
    user_messages = state["user_messages"]
    # Preprocess user_messages to create a string of all message contents
    user_messages_content = "\n".join([message.content for message in user_messages])

    ai_messages = state["ai_messages"]
    ai_messages_content = "\n".join([message.content for message in ai_messages])
    prompt = PromptTemplate(
        template="""You are a helpful assistant. You should give a final answer to the user following the conversation:
        {user_messages}
        and the agent response(s):
        {ai_messages}
        """,
        input_variables=["user_messages", "ai_messages"],
    )
    chain = prompt | llm
    response = chain.invoke({"user_messages": user_messages_content, "ai_messages": ai_messages_content})
    print(response)
    return Command(
        update={"user_messages": [HumanMessage(content=response.content, name="final_answer")]},
        goto=END,
    )

# Create the graph.
builder = StateGraph(State)

builder.add_node("supervisor", supervisor_agent)
builder.add_node("rag", rag_agent)
builder.add_node("ANSWER", final_answer)

builder.add_edge(START, "supervisor")

multi_agent_graph = builder.compile(checkpointer=memory)

# Stream the graph updates.
def stream_graph_updates(user_input: str):
    for event in multi_agent_graph.stream(
        {"user_messages": [("user", user_input)]},
        {"configurable": {"thread_id": "1"}},
    ):
        for value in event.values():
            print(value)  # Keep debug print
            if "user_messages" in value and value["user_messages"]:
                print("User Assistant:", value["user_messages"][-1].content)
            elif "ai_messages" in value and value["ai_messages"]:
                print("Other Assistant:", value["ai_messages"][-1].content)

while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        break