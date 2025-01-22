
###########################
# LangGraph Agent
###########################

from __future__ import annotations

import time
import os
import asyncio

from livekit.agents.llm.llm import LLM, LLMStream, ChatContext, ChatChunk, Choice, ChoiceDelta, LLMCapabilities, ToolChoice
from livekit.agents.llm import function_context
from livekit.agents.utils import aio
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

from typing import Optional, Union, Literal, AsyncIterator

# Import the graph for the multi agent example
from multi_agent import multi_agent_graph as graph

from dotenv import load_dotenv
load_dotenv()

###########################
# Export Classes
###########################

class LangGraphLLM(LLM):
    def __init__(self, graph):
        super().__init__()
        self._graph = graph
        self._capabilities = LLMCapabilities(supports_choices_on_int=True)

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        fnc_ctx: Optional[function_context.FunctionContext] = None,
        temperature: Optional[float] = None,
        n: Optional[int] = None,
        parallel_tool_calls: Optional[bool] = None,
        tool_choice: Union["ToolChoice", Literal["auto", "required", "none"]] = None,
    ) -> "LangGraphLLMStream":
        return LangGraphLLMStream(
            self,
            chat_ctx=chat_ctx,
            fnc_ctx=fnc_ctx,
            conn_options=conn_options,
            graph=self._graph
        )

    async def aclose(self) -> None:
        pass

class LangGraphLLMStream(LLMStream):
    def __init__(
        self,
        llm: "LangGraphLLM",
        *,
        chat_ctx: ChatContext,
        fnc_ctx: Optional[function_context.FunctionContext],
        conn_options: APIConnectOptions,
        graph,
    ):
        super().__init__(llm, chat_ctx=chat_ctx, fnc_ctx=fnc_ctx, conn_options=conn_options)
        self._graph = graph
        self._response_queue = asyncio.Queue()
        print("LangGraphLLMStream initialized")  # Debug

    async def _run(self) -> None:
        """Run the agent and put responses into the queue"""
        print("LangGraphLLMStream _run")  # Debug
        try:
            # Extract the latest user message from the chat context
            user_message = None
            for message in self._chat_ctx.messages:
                if message.role == "user":
                    user_message = message.content
                    print("User message: ", user_message)  # Debug
            if not user_message:
                raise ValueError("No user message found in chat context")

            # Track what we've sent to avoid duplicates
            sent_responses = set()

            # Stream responses from the LangGraph agent
            async for response in self._graph.astream(
                {"user_messages": [("user", user_message)]},
                {"configurable": {"thread_id": "1"}},
            ):
                print("LangGraph response: ", response)  # Debug
                
                if not response.get("supervisor"): # see below for terms...
                    continue

                # Get the AIMessage from the response
                ai_message = response["supervisor"]["user_messages"][0] # these fields are set by the agent workflow - check with your code.
                
                # Handle the content which could be a string or a list of message parts - COULD IT!?!?!?
                if isinstance(ai_message.content, str):
                    print("AI message content is a string")
                    text = ai_message.content
                    if text and text not in sent_responses:
                        chunk = ChatChunk(
                            request_id="langgraph",
                            choices=[Choice(delta=ChoiceDelta(role="assistant", content=text))]
                        )
                        await self._response_queue.put(chunk)
                        print("LangGraphLLMStream chunk: ", chunk)  # Debug
                        sent_responses.add(text)
                elif isinstance(ai_message.content, list):
                    print("AI message content is a list")
                    for part in ai_message.content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text = part.get("text")
                            if text and text not in sent_responses:
                                chunk = ChatChunk(
                                    request_id="langgraph",
                                    choices=[Choice(delta=ChoiceDelta(role="assistant", content=text))]
                                )
                                await self._response_queue.put(chunk)
                                print("LangGraphLLMStream chunk: ", chunk)  # Debug
                                sent_responses.add(text)

        except Exception as e:
            print(f"Error in _run: {e}")  # Debug
            raise
        finally:
            # Signal that we're done
            await self._response_queue.put(None)
            print("LangGraphLLMStream _run finally")  # Debug

    # This is the iterator for the LangGraphLLMStream
    def __aiter__(self) -> AsyncIterator[ChatChunk]:
        """Start the processing and return self as the iterator"""
        # self._task = asyncio.create_task(self._run()) - THIS WAS TROUBLESOME (double running the agent)
        return self

    # Return the next chunk from the queue
    async def __anext__(self) -> ChatChunk:
        """Get the next chunk from the queue"""
        chunk = await self._response_queue.get()
        if chunk is None:
            raise StopAsyncIteration
        return chunk

    # Clean up resources
    async def aclose(self) -> None:
        """Clean up resources"""
        if hasattr(self, '_task'):
            await aio.gracefully_cancel(self._task)
        await super().aclose()
