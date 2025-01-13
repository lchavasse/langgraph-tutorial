import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.tools.retriever import create_retriever_tool
from langgraph.prebuilt import ToolNode, tools_condition

from typing import Literal
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from langgraph.graph import StateGraph, MessagesState, START, END

from langchain_anthropic import ChatAnthropic

from langchain_community.tools.tavily_search import TavilySearchResults

from dotenv import load_dotenv
load_dotenv()

memory = MemorySaver()

class State(TypedDict):
    messages: Annotated[list, add_messages]

search_tool = TavilySearchResults(max_results=2)

documents = []
for filename in os.listdir("./tut-docs"):
    if filename.endswith(".txt"):
        with open(os.path.join("./tut-docs", filename), "r") as file:
            text = file.read()
            documents.append({"content": text, "filename": filename})

source_docs = [
    Document(page_content=doc["content"], metadata={"filename": doc["filename"]})
    for doc in documents
]

# Split the documents into chunks for vector store

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=50
)

doc_splits = text_splitter.split_documents(source_docs)

embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

vectorstore = InMemoryVectorStore.from_documents(documents=doc_splits,
                                    embedding=embed_model)

retriever = vectorstore.as_retriever(search_kwargs={"k":2})

retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_california_stats",
    "Search and return information about California.",
)

# Create tools array for agent
tools = [retriever_tool, search_tool]

# Create tools node for graph
tools_node = ToolNode(tools)

llm = ChatAnthropic(model="claude-3-5-haiku-latest")

llm_with_tools = llm.bind_tools(tools)


def agent(state: State):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    print(response)
    return {"messages": [response]}

builder = StateGraph(State)

builder.add_node("agent", agent)
builder.add_node("tools", tools_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition, ["tools", END])
builder.add_edge("tools", "agent")

graph = builder.compile(checkpointer=memory)

def stream_graph_updates(user_input: str):
    for event in graph.stream(
        {"messages": [("user", user_input)]},
        {"configurable": {"thread_id": "1"}},
    ):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


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