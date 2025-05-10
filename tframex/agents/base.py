import logging
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Union

from tframex.llms import BaseLLMWrapper
from tframex.tools import Tool
from tframex.memory import BaseMemoryStore, InMemoryMemoryStore
from tframex.primitives import Message

logger = logging.getLogger(__name__)

# --- Dedicated Deep Debug Logger ---
# This logger will always output, regardless of the main application's logging level for "tframex"
# This is for developers of/with the library to trace internal agent flow.
# Users can choose to see these by configuring a handler for "tframex.agent_internal_debug"
agent_internal_debug_logger = logging.getLogger("tframex.agent_internal_debug")
# To ensure it always outputs if a handler is attached to it or its parents,
# set its level to DEBUG and ensure it propagates. Or, add a specific handler here.
# For simplicity, we'll assume users will configure a handler if they want to see these.
# If no handler is configured for this specific logger or its parents (up to root),
# and the root logger's level is INFO, these messages won't show by default, which is fine.
# If a developer wants to see them, they can do:
# logging.basicConfig(level=logging.DEBUG) # or
# logging.getLogger("tframex.agent_internal_debug").setLevel(logging.DEBUG)
# logging.getLogger("tframex.agent_internal_debug").addHandler(logging.StreamHandler())
agent_internal_debug_logger.setLevel(logging.DEBUG) # Process messages at DEBUG level
# agent_internal_debug_logger.propagate = True # Default is True


class BaseAgent(ABC):
    def __init__(self, 
                 agent_id: str, # Unique ID for this instance of the agent
                 llm: Optional[BaseLLMWrapper] = None,
                 tools: Optional[List[Tool]] = None,
                 memory: Optional[BaseMemoryStore] = None,
                 system_prompt_template: Optional[str] = None,
                 **config: Any): # For other agent-specific configurations
        self.agent_id = agent_id
        self.llm = llm
        self.tools: Dict[str, Tool] = {tool.name: tool for tool in tools} if tools else {}
        self.memory: BaseMemoryStore = memory or InMemoryMemoryStore()
        self.system_prompt_template = system_prompt_template
        self.config = config
        
        agent_internal_debug_logger.debug(f"[{self.agent_id}] BaseAgent.__init__ called. LLM: {llm.model_id if llm else 'None'}. Tools: {list(self.tools.keys())}. System Prompt: {bool(system_prompt_template)}. Config: {config}")
        logger.info(f"Agent '{agent_id}' initialized. LLM: {llm.model_id if llm else 'None'}. Tools: {list(self.tools.keys())}.")


    def _render_system_prompt(self, **kwargs_for_template: Any) -> Optional[Message]:
        agent_internal_debug_logger.debug(f"[{self.agent_id}] _render_system_prompt called. Template: '{self.system_prompt_template}', Args: {kwargs_for_template}")
        if not self.system_prompt_template:
            agent_internal_debug_logger.debug(f"[{self.agent_id}] No system_prompt_template defined.")
            return None
        try:
            content = self.system_prompt_template.format(**kwargs_for_template)
            msg = Message(role="system", content=content)
            agent_internal_debug_logger.debug(f"[{self.agent_id}] Rendered system prompt: {msg}")
            return msg
        except KeyError as e:
            agent_internal_debug_logger.warning(f"[{self.agent_id}] Missing key '{e}' for system_prompt_template. Template: '{self.system_prompt_template}'")
            logger.warning(f"Agent '{self.agent_id}': Missing key '{e}' for system_prompt_template. Template: '{self.system_prompt_template}'")
            # Return unformatted, still useful for debugging
            return Message(role="system", content=self.system_prompt_template) 

    @abstractmethod
    async def run(self, input_message: Union[str, Message], **kwargs: Any) -> Message:
        """
        Primary execution method. Takes input, returns a single Message from the assistant.
        kwargs can be used for runtime overrides or additional context.
        """
        agent_internal_debug_logger.debug(f"[{self.agent_id}] Abstract run method invoked with input: {input_message}, kwargs: {kwargs}. (Implementation specific logs will follow)")
        pass

    def add_tool(self, tool: Tool):
        agent_internal_debug_logger.debug(f"[{self.agent_id}] add_tool called. Tool: {tool.name}")
        if tool.name in self.tools:
            agent_internal_debug_logger.warning(f"[{self.agent_id}] Tool '{tool.name}' already exists. Overwriting.")
            logger.warning(f"Tool '{tool.name}' already exists in agent '{self.agent_id}'. Overwriting.")
        self.tools[tool.name] = tool
        logger.info(f"Tool '{tool.name}' added to agent '{self.agent_id}'.")

    @classmethod
    def get_agent_type_id(cls) -> str: return f"tframex.agents.{cls.__name__}"
    @classmethod
    def get_display_name(cls) -> str: return cls.__name__