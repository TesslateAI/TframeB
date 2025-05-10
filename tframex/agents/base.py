import logging
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Union

from tframex.llms import BaseLLMWrapper
from tframex.tools import Tool, ToolDefinition # Added ToolDefinition
from tframex.memory import BaseMemoryStore, InMemoryMemoryStore
from tframex.primitives import Message

logger = logging.getLogger(__name__)
agent_internal_debug_logger = logging.getLogger("tframex.agent_internal_debug")
agent_internal_debug_logger.setLevel(logging.DEBUG)


class BaseAgent(ABC):
    def __init__(self,
                 agent_id: str,
                 description: Optional[str] = None, # NEW
                 llm: Optional[BaseLLMWrapper] = None,
                 tools: Optional[List[Tool]] = None,
                 memory: Optional[BaseMemoryStore] = None,
                 system_prompt_template: Optional[str] = None,
                 callable_agent_definitions: Optional[List[ToolDefinition]] = None, # NEW
                 **config: Any):
        self.agent_id = agent_id
        self.description = description or f"Agent performing its designated role: {agent_id.split('_ctx')[0]}" # NEW, cleaner default
        self.llm = llm
        self.tools: Dict[str, Tool] = {tool.name: tool for tool in tools} if tools else {}
        self.memory: BaseMemoryStore = memory or InMemoryMemoryStore()
        self.system_prompt_template = system_prompt_template
        # Store callable agent definitions if this agent is a supervisor
        self.callable_agent_definitions: List[ToolDefinition] = callable_agent_definitions or [] # NEW
        self.config = config

        agent_internal_debug_logger.debug(
            f"[{self.agent_id}] BaseAgent.__init__ called. Description: '{self.description}'. "
            f"LLM: {llm.model_id if llm else 'None'}. Tools: {list(self.tools.keys())}. "
            f"Callable Agents: {[cad.function['name'] for cad in self.callable_agent_definitions]}. " # NEW log
            f"System Prompt: {bool(system_prompt_template)}. Config: {config}"
        )
        logger.info(
            f"Agent '{agent_id}' initialized. LLM: {llm.model_id if llm else 'None'}. "
            f"Tools: {list(self.tools.keys())}. "
            f"Callable Agents: {[cad.function['name'] for cad in self.callable_agent_definitions]}." # NEW log
        )


    def _render_system_prompt(self, **kwargs_for_template: Any) -> Optional[Message]:
        agent_internal_debug_logger.debug(f"[{self.agent_id}] _render_system_prompt called. Template: '{self.system_prompt_template}', Args: {kwargs_for_template}")
        if not self.system_prompt_template:
            agent_internal_debug_logger.debug(f"[{self.agent_id}] No system_prompt_template defined.")
            return None
        try:
            # Prepare description of available tools and callable agents for the system prompt if placeholders exist
            prompt_format_args = kwargs_for_template.copy()

            tool_descriptions = "\n".join(
                [f"- {name}: {tool.description}" for name, tool in self.tools.items()]
            )
            prompt_format_args["available_tools_descriptions"] = tool_descriptions or "No tools available."

            callable_agent_tool_descs = "\n".join(
                [f"- {cad.function['name']}: {cad.function['description']}" for cad in self.callable_agent_definitions]
            )
            prompt_format_args["available_agents_descriptions"] = callable_agent_tool_descs or "No callable agents available."


            content = self.system_prompt_template.format(**prompt_format_args)
            msg = Message(role="system", content=content)
            agent_internal_debug_logger.debug(f"[{self.agent_id}] Rendered system prompt: {msg}")
            return msg
        except KeyError as e:
            agent_internal_debug_logger.warning(f"[{self.agent_id}] Missing key '{e}' for system_prompt_template. Template: '{self.system_prompt_template}'")
            logger.warning(f"Agent '{self.agent_id}': Missing key '{e}' for system_prompt_template. Template: '{self.system_prompt_template}'")
            # Return unformatted, still useful for debugging, but try a simple format first
            try:
                content = self.system_prompt_template.format(**kwargs_for_template) # Try without tool/agent descriptions
                return Message(role="system", content=content)
            except KeyError: # Still fails, return raw
                 return Message(role="system", content=self.system_prompt_template)


    @abstractmethod
    async def run(self, input_message: Union[str, Message], **kwargs: Any) -> Message:
        """
        Primary execution method. Takes input, returns a single Message from the assistant.
        kwargs can be used for runtime overrides or additional context, like 'template_vars'.
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