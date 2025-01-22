###########################
# LiveKit Pipeline Agent
###########################

import logging

from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
)

from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import deepgram, silero, openai

from langgraph_agent import LangGraphLLM, graph

from dotenv import load_dotenv
load_dotenv(".env.local")

logger = logging.getLogger("voice-agent")

# Define the prewarm function to speed up response time
def prewarm(proc: JobProcess):
    """
    Initialize resources before the worker starts processing jobs.
    """
    # Load the VAD model
    proc.userdata["vad"] = silero.VAD.load()

    # Initialize and store the LangGraph agent
    proc.userdata["graph"] = graph  # Assuming `graph` is defined elsewhere

async def entrypoint(ctx: JobContext):
    # This doesn't get passed to the agent currently.
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            "You are a voice assistant. Your interface with users will be voice. "
            "You should use short and concise responses, and avoid usage of unpronounceable punctuation. "
            "You have access to a research tool that can gather additional information. "
        ),
    )

    # Connect to the LiveKit room
    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Wait for the first participant to connect
    participant = await ctx.wait_for_participant()
    logger.info(f"starting voice assistant for participant {participant.identity}")

    # Retrieve the prewarmed resources
    vad = ctx.proc.userdata["vad"]
    graph = ctx.proc.userdata["graph"]

    # Wrap your LangGraph agent in the LangGraphLLM class
    langgraph_llm = LangGraphLLM(graph)

    # Replace OpenAI LLM with your LangGraph agent
    assistant = VoicePipelineAgent(
        vad=vad,
        stt=deepgram.STT(),
        llm=langgraph_llm,
        tts=openai.TTS(),
        chat_ctx=initial_ctx,
    )

    # Start the voice assistant
    logger.info(f"starting voice assistant for participant {participant.identity}")
    assistant.start(ctx.room, participant)

    # Greet the user
    await assistant.say("Hey, how can I help you today?", allow_interruptions=True)

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )