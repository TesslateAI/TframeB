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
logging.getLogger("tframex").setLevel(logging.INFO) # General tframex logs
# logging.getLogger("tframex.agents.llm_agent").setLevel(logging.DEBUG) # For tool/agent call logs from LLMAgent
# logging.getLogger("tframex.agents.base").setLevel(logging.DEBUG) # For agent init logs
# logging.getLogger("tframex.app").setLevel(logging.DEBUG) # For app/context logs
# logging.getLogger("tframex.agent_internal_debug").setLevel(logging.DEBUG) # Deep debug

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


# --- Tool Definitions ---
@app.tool(description="Gets the current weather for a specific location. Requires location and optionally unit (celsius/fahrenheit).")
async def get_current_weather(location: str, unit: str = "celsius") -> str:
    """Gets the current weather for a location."""
    logging.info(f"TOOL EXECUTED: get_current_weather(location='{location}', unit='{unit}')")
    if "tokyo" in location.lower():
        return f"The current weather in Tokyo is 25°{unit.upper()[0]} and sunny."
    if "paris" in location.lower():
        return f"The current weather in Paris is 18°{unit.upper()[0]} and cloudy."
    return f"Weather data for {location} is currently unavailable."

@app.tool(description="Retrieves general information about a city, such as population or main attractions. Requires city name and optionally info_type ('population', 'attractions').")
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

# --- Agent Definitions ---

# Basic Agents
@app.agent(
    name="EchoAgent",
    description="A simple agent that repeats the user's message verbatim.",
    system_prompt="You are an Echo agent. Your task is to repeat the user's message verbatim. Do not add any other text or explanation."
)
async def echo_agent_placeholder(): pass

@app.agent(
    name="UpperCaseAgent",
    description="Converts the entire user's message to uppercase.",
    system_prompt="You are an UpperCase agent. Your task is to convert the entire user's message to uppercase. Respond with ONLY the uppercased text."
)
async def uppercase_agent_placeholder(): pass

@app.agent(
    name="ReverseAgent",
    description="Reverses the text of the user's message.",
    system_prompt="You are a Reverse agent. Your task is to reverse the text of the user's message. Respond with ONLY the reversed text."
)
async def reverse_agent_placeholder(): pass

# Tool-using Agents (for old RouterPattern and new SmartQueryAgent)
@app.agent(
    name="WeatherAgent",
    description="Provides weather information for a given city. Uses the 'get_current_weather' tool.",
    system_prompt=(
        "You are a helpful Weather Assistant. Your primary function is to provide weather information. "
        "Use the 'get_current_weather' tool to find the weather for the specified location. "
        "If the user asks for weather, extract the location and call the tool. "
        "If the user's input is not about weather, politely state that you only handle weather requests."
    ),
    tools=["get_current_weather"]
)
async def weather_agent_placeholder(): pass

@app.agent(
    name="CityInfoAgent",
    description="Provides details about cities, like population or attractions. Uses the 'get_city_info' tool.",
    system_prompt=(
        "You are a knowledgeable City Information Provider. Your role is to give details about cities. "
        "Use the 'get_city_info' tool. You can infer the 'info_type' (e.g., 'population', 'attractions') "
        "from the user's query if not explicitly stated, otherwise default to 'attractions'. "
        "If the query is not about city information, state your purpose."
    ),
    tools=["get_city_info"]
)
async def city_info_agent_placeholder(): pass

# Summarizer Agent (for ParallelPattern)
@app.agent(
    name="SummarizerAgent",
    description="Summarizes the input text concisely.",
    system_prompt=(
        "You are a Summarization Bot. Your task is to take the input text and provide a concise summary, "
        "capturing the main points. Aim for 1-2 sentences for shorter inputs, and a short paragraph for longer ones."
    )
)
async def summarizer_agent_placeholder(): pass

# Agent for Router Pattern (Old way of routing)
@app.agent(
    name="TaskRouterAgent",
    description="Classifies a user query and outputs a route key ('weather', 'city_info', or 'echo'). This agent is used by the RouterPattern.",
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

# NEW: Supervisor Agent using "Agent as Tool"
@app.agent(
    name="SmartQueryDelegateAgent",
    description="A smart supervisor agent that analyzes a user's query and delegates it to either WeatherAgent or CityInfoAgent by calling them as tools. It then returns their response.",
    system_prompt=(
        "You are a Smart Query Supervisor. Your task is to understand the user's request and delegate it to the appropriate specialist agent. "
        "You have the following specialist agents available to call as functions (tools):\n"
        "{available_agents_descriptions}\n\n" # Placeholder for descriptions of callable agents
        "Based on the user's query, decide which agent is best suited. "
        "Then, call that agent using its name as the function name and provide the user's original query (or a relevant part of it) as the 'input_message' argument for that agent. "
        "After receiving the response from the specialist agent, present that information clearly to the user. "
        "If the query is ambiguous or cannot be handled by the available agents, politely state that."
    ),
    callable_agents=["WeatherAgent", "CityInfoAgent"] # This agent can call WeatherAgent and CityInfoAgent
)
async def smart_query_delegate_placeholder(): pass


# Agents for Discussion Pattern
@app.agent(
    name="OptimistAgent",
    description="Participates in discussions with an optimistic viewpoint.",
    system_prompt=(
        "You are participating in a discussion. You are the Optimist. "
        "Always find the positive aspects or potential benefits of the topic presented in the user's message. "
        "Keep your response to 1-2 sentences. Start your response with 'As an optimist, I see that...'"
    )
)
async def optimist_placeholder(): pass

@app.agent(
    name="PessimistAgent",
    description="Participates in discussions with a pessimistic viewpoint.",
    system_prompt=(
        "You are participating in a discussion. You are the Pessimist. "
        "Always point out the potential downsides, risks, or challenges of the topic presented in the user's message. "
        "Keep your response to 1-2 sentences. Start your response with 'However, from a pessimistic view, ...'"
    )
)
async def pessimist_placeholder(): pass

@app.agent(
    name="RealistAgent",
    description="Participates in discussions with a balanced, realistic viewpoint.",
    system_prompt=(
        "You are participating in a discussion. You are the Realist. "
        "Provide a balanced perspective on the topic from the user's message, considering both pros and cons, or offering a neutral observation. "
        "Keep your response to 1-2 sentences. Start your response with 'Realistically speaking, ...'"
    )
)
async def realist_placeholder(): pass

@app.agent(
    name="DiscussionModeratorAgent",
    description="Moderates a discussion by summarizing rounds and posing follow-up questions.",
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


# --- Flow Definitions ---

# 1. Sequential Flow
sequential_flow = Flow(flow_name="SequentialEchoUpperReverse", description="Echoes, then uppercases, then reverses input.")
sequential_flow.add_step("EchoAgent")
sequential_flow.add_step("UpperCaseAgent")
sequential_flow.add_step("ReverseAgent")
app.register_flow(sequential_flow)

# 2. Parallel Flow
parallel_tasks_flow = Flow(flow_name="ParallelWeatherAndCityInfoThenSummarize", description="Gets weather and city info in parallel, then summarizes.")
parallel_tasks_flow.add_step(
    ParallelPattern(
        pattern_name="GetInfoInParallel",
        tasks=["WeatherAgent", "CityInfoAgent"]
    )
)
parallel_tasks_flow.add_step("SummarizerAgent") # Summarizer gets the combined output of the parallel tasks
app.register_flow(parallel_tasks_flow)

# 3. Router Flow (Old method using a dedicated Router Agent)
router_flow_old_method = Flow(flow_name="SmartTaskRouterFlow_Old", description="Routes task to Weather, CityInfo, or Echo agent using TaskRouterAgent.")
router_flow_old_method.add_step(
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
app.register_flow(router_flow_old_method)

# 4. Discussion Flow
discussion_flow = Flow(flow_name="ExpertTeamDebate", description="Optimist, Pessimist, and Realist discuss a topic, moderated.")
discussion_flow.add_step(
    DiscussionPattern(
        pattern_name="TeamDebateOnTopic",
        participant_agent_names=["OptimistAgent", "PessimistAgent", "RealistAgent"],
        discussion_rounds=2,
        moderator_agent_name="DiscussionModeratorAgent",
        stop_phrase="end discussion now"
    )
)
app.register_flow(discussion_flow)

# 5. Simple Orchestration (Sequential specialists - less ideal for this specific scenario)
simple_orchestrator_flow = Flow(flow_name="ContrivedSequentialOrchestration", description="Gets weather, then (less ideally) uses weather output to get city info.")
simple_orchestrator_flow.add_step("WeatherAgent")
simple_orchestrator_flow.add_step("CityInfoAgent") # Output of WeatherAgent might not be a good input for CityInfoAgent here
app.register_flow(simple_orchestrator_flow)

# 6. NEW: Flow using Agent-as-Tool (Supervisor Agent)
smart_delegation_flow = Flow(flow_name="SmartQueryDelegationFlow_New", description="Uses SmartQueryDelegateAgent to handle weather or city info requests.")
smart_delegation_flow.add_step("SmartQueryDelegateAgent")
app.register_flow(smart_delegation_flow)

# 7. Flow demonstrating template variables for system prompts
templated_flow = Flow(flow_name="TemplatedGreetingFlow", description="Greets a user whose name is provided via template variable.")
# Define an agent that uses a template variable
@app.agent(
    name="GreetingAgent",
    description="Greets a user by name.",
    system_prompt="You are a friendly assistant. The user's name is {user_name}. Greet them warmly and ask how you can help with their query: '{user_query}'.",
)
async def greeting_agent_placeholder(): pass
templated_flow.add_step("GreetingAgent")
app.register_flow(templated_flow)


# --- Main Application CLI ---
async def main():
    async with app.run_context() as rt:
        # Example of running a specific flow with template variables
        # initial_msg = Message(role="user", content="I'd like to plan a trip.")
        # context = await rt.run_flow(
        #     "TemplatedGreetingFlow",
        #     initial_msg,
        #     flow_template_vars={"user_name": "Dr. Smith", "user_query": initial_msg.content}
        # )
        # print("Output from TemplatedGreetingFlow:", context.current_message.content)

        await rt.interactive_chat() # Will list available flows and ask

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_BASE"):
        print("FATAL: OPENAI_API_BASE environment variable is not set.")
    else:
        asyncio.run(main())