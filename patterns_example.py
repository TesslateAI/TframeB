import asyncio
import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from tframex import (
    TFrameXApp, OpenAIChatLLM, InMemoryMemoryStore, Message,
    Flow, SequentialPattern, ParallelPattern, RouterPattern, DiscussionPattern,
    TFrameXRuntimeContext # Ensure this is imported if type hinting
)

# --- Environment and Logging Setup ---
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - [%(funcName)s] - %(message)s'
)
logging.getLogger("tframex.flows").setLevel(logging.INFO)
logging.getLogger("tframex.patterns").setLevel(logging.INFO)
# logging.getLogger("tframex.agents.llm_agent").setLevel(logging.DEBUG) # For tool usage logs from LLMAgent

# --- LLM Configuration ---
llm_config = OpenAIChatLLM(
    model_name=os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo"),
    api_base_url=os.getenv("OPENAI_API_BASE", "http://localhost:11434"), # Common for Ollama
    api_key=os.getenv("OPENAI_API_KEY", "ollama")
)
if not llm_config.api_base_url:
    print("Error: OPENAI_API_BASE not set.")
    exit(1)

# --- Initialize TFrameX Application ---
app = TFrameXApp(default_llm=llm_config)


# --- Tool Definitions (Unchanged) ---
@app.tool()
async def get_current_weather(location: str, unit: str = "celsius") -> str:
    """Gets the current weather for a location."""
    logging.info(f"TOOL EXECUTED: get_current_weather(location='{location}', unit='{unit}')")
    if "tokyo" in location.lower():
        return f"The current weather in Tokyo is 25°{unit.upper()[0]} and sunny."
    if "paris" in location.lower():
        return f"The current weather in Paris is 18°{unit.upper()[0]} and cloudy."
    return f"Weather data for {location} is currently unavailable."

@app.tool()
async def get_city_info(city_name: str, info_type: str = "population") -> str:
    """Gets general information about a city, like population or main attractions."""
    logging.info(f"TOOL EXECUTED: get_city_info(city_name='{city_name}', info_type='{info_type}')")
    if "paris" in city_name.lower():
        if info_type == "population": return "Population of Paris is approximately 2.1 million."
        if info_type == "attractions": return "Main attractions in Paris: Eiffel Tower, Louvre Museum, Notre Dame."
    if "tokyo" in city_name.lower():
        if info_type == "population": return "Population of Tokyo is approximately 14 million."
        if info_type == "attractions": return "Main attractions in Tokyo: Tokyo Skytree, Senso-ji Temple, Shibuya Crossing."
    return f"Information of type '{info_type}' for {city_name} not found."

# --- Agent Definitions (Prompts Improved) ---

@app.agent(name="EchoAgent", system_prompt="You are an Echo agent. Your task is to repeat the user's message verbatim. Do not add any other text or explanation.")
async def echo_agent_placeholder(): pass

@app.agent(name="UpperCaseAgent", system_prompt="You are an UpperCase agent. Your task is to convert the entire user's message to uppercase. Respond with ONLY the uppercased text.")
async def uppercase_agent_placeholder(): pass

@app.agent(name="ReverseAgent", system_prompt="You are a Reverse agent. Your task is to reverse the text of the user's message. Respond with ONLY the reversed text.")
async def reverse_agent_placeholder(): pass

@app.agent(
    name="WeatherAgent",
    system_prompt=(
        "You are a helpful Weather Assistant. Your primary function is to provide weather information. "
        "Use the 'get_current_weather' tool to find the weather for the specified location. "
        "If the user asks for weather, extract the location and call the tool. "
        "If the user's input is not about weather, you can politely state that you only handle weather requests."
    ),
    tools=["get_current_weather"]
)
async def weather_agent_placeholder(): pass

@app.agent(
    name="CityInfoAgent",
    system_prompt=(
        "You are a knowledgeable City Information Provider. Your role is to give details about cities. "
        "Use the 'get_city_info' tool. You can infer the 'info_type' (e.g., 'population', 'attractions') "
        "from the user's query if not explicitly stated, otherwise default to 'attractions'. "
        "If the query is not about city information, you can state your purpose."
    ),
    tools=["get_city_info"]
)
async def city_info_agent_placeholder(): pass

@app.agent(
    name="SummarizerAgent",
    system_prompt=(
        "You are a Summarization Bot. Your task is to take the input text and provide a concise summary, "
        "capturing the main points. Aim for 1-2 sentences for shorter inputs, and a short paragraph for longer ones."
    )
)
async def summarizer_agent_placeholder(): pass

# Agents for Router Pattern
@app.agent(
    name="TaskRouterAgent",
    system_prompt=(
        "You are a Task Router. Your job is to analyze the user's query and decide which specialist can best handle it. "
        "Available specialists and their corresponding route keys are: "
        "- For weather-related queries: respond with 'weather' "
        "- For city information queries: respond with 'city_info' "
        "- For any other type of query: respond with 'echo' "
        "Your response MUST BE EXACTLY ONE of these route keys: 'weather', 'city_info', or 'echo'. Do not add any other words or punctuation."
    )
)
async def task_router_placeholder(): pass

# Agents for Discussion Pattern
@app.agent(
    name="OptimistAgent",
    system_prompt=(
        "You are participating in a discussion. You are the Optimist. "
        "Always find the positive aspects or potential benefits of the topic presented in the user's message. "
        "Keep your response to 1-2 sentences. Start your response with 'As an optimist, I see that...'"
    )
)
async def optimist_placeholder(): pass

@app.agent(
    name="PessimistAgent",
    system_prompt=(
        "You are participating in a discussion. You are the Pessimist. "
        "Always point out the potential downsides, risks, or challenges of the topic presented in the user's message. "
        "Keep your response to 1-2 sentences. Start your response with 'However, from a pessimistic view, ...'"
    )
)
async def pessimist_placeholder(): pass

@app.agent(
    name="RealistAgent",
    system_prompt=(
        "You are participating in a discussion. You are the Realist. "
        "Provide a balanced perspective on the topic from the user's message, considering both pros and cons, or offering a neutral observation. "
        "Keep your response to 1-2 sentences. Start your response with 'Realistically speaking, ...'"
    )
)
async def realist_placeholder(): pass

@app.agent(
    name="DiscussionModeratorAgent",
    system_prompt=(
        "You are the Discussion Moderator. You will receive a summary of a discussion round. "
        "Your task is to: "
        "1. Briefly summarize the key points made by each participant in the round. "
        "2. Identify any common themes, agreements, or disagreements. "
        "3. If appropriate, pose a follow-up question or topic refinement for the next round based on the discussion. "
        "Keep your overall response concise. Structure your summary clearly."
    )
)
async def discussion_moderator_placeholder(): pass


# --- Flow Definitions (Unchanged from your provided code) ---

# 1. Sequential Flow
sequential_flow = Flow(flow_name="SequentialEchoAndUpper", description="Echoes, then uppercases, then reverses input.")
sequential_flow.add_step("EchoAgent")
sequential_flow.add_step("UpperCaseAgent")
sequential_flow.add_step("ReverseAgent")
app.register_flow(sequential_flow)

# 2. Parallel Flow
parallel_tasks_flow = Flow(flow_name="ParallelWeatherAndCityInfo", description="Gets weather and city info in parallel, then summarizes.")
parallel_tasks_flow.add_step(
    ParallelPattern(
        pattern_name="GetInfoInParallel",
        tasks=["WeatherAgent", "CityInfoAgent"] 
    )
)
parallel_tasks_flow.add_step("SummarizerAgent")
app.register_flow(parallel_tasks_flow)

# 3. Router Flow
router_flow = Flow(flow_name="SmartTaskRouterFlow", description="Routes task to Weather, CityInfo, or Echo agent.")
router_flow.add_step(
    RouterPattern(
        pattern_name="MainTaskRouter",
        router_agent_name="TaskRouterAgent",
        routes={
            "weather": "WeatherAgent",
            "city_info": "CityInfoAgent",
            "echo": "EchoAgent"
        },
        default_route="EchoAgent"
    )
)
app.register_flow(router_flow)

# 4. Discussion Flow
discussion_flow = Flow(flow_name="ExpertDiscussionFlow", description="Optimist, Pessimist, and Realist discuss a topic, moderated.")
discussion_flow.add_step(
    DiscussionPattern(
        pattern_name="TeamDebate",
        participant_agent_names=["OptimistAgent", "PessimistAgent", "RealistAgent"],
        discussion_rounds=2,
        moderator_agent_name="DiscussionModeratorAgent",
        stop_phrase="end discussion" # Example: if an agent says "I think we should end discussion here."
    )
)
app.register_flow(discussion_flow)

# 5. Simple Orchestration (Sequential specialists)
simple_orchestrator_flow = Flow(flow_name="SimpleOrchestration", description="Gets weather, then uses weather output to get city info (less ideal, just demo).")
# This setup is a bit contrived for CityInfoAgent, as WeatherAgent's output might not be a good city name.
# A better orchestrator would manage inputs more explicitly.
simple_orchestrator_flow.add_step("WeatherAgent") 
simple_orchestrator_flow.add_step("CityInfoAgent") 
app.register_flow(simple_orchestrator_flow)


# --- Main Application CLI (Using interactive_chat from TFrameXRuntimeContext) ---
async def main():
    async with app.run_context() as rt:
        # You can specify a default flow name to start with for easier testing,
        # or leave it None to be prompted to choose from all registered flows.
        # await rt.interactive_chat(default_flow_name="SmartTaskRouterFlow")
        # await rt.interactive_chat(default_flow_name="ExpertDiscussionFlow")
        await rt.interactive_chat() # Will list available flows and ask

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_BASE"):
        print("FATAL: OPENAI_API_BASE environment variable is not set.")
    else:
        asyncio.run(main())