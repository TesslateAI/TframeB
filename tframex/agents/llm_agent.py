import logging
import json
from typing import Optional, List, Union, Dict, Any

from .base import BaseAgent
from tframex.llms import BaseLLMWrapper
from tframex.tools import Tool
from tframex.memory import BaseMemoryStore
from tframex.primitives import Message, ToolCall, ToolDefinition

logger = logging.getLogger(__name__)

class LLMAgent(BaseAgent):
    """
    An agent that uses an LLM to decide actions, potentially using tools and memory.
    """
    def __init__(self, agent_id: str, llm: BaseLLMWrapper, 
                 tools: Optional[List[Tool]] = None, 
                 memory: Optional[BaseMemoryStore] = None,
                 system_prompt_template: Optional[str] = "You are a helpful assistant.",
                 max_tool_iterations: int = 3, # Max consecutive tool calls before forcing text response
                 **config: Any):
        super().__init__(agent_id, llm, tools, memory, system_prompt_template, **config)
        self.max_tool_iterations = max_tool_iterations
        if not self.llm:
            raise ValueError(f"LLMAgent '{self.agent_id}' requires an LLM instance.")

    async def run(self, input_message: Union[str, Message], **kwargs: Any) -> Message:
        if isinstance(input_message, str):
            current_user_message = Message(role="user", content=input_message)
        else:
            current_user_message = input_message
        
        await self.memory.add_message(current_user_message)

        for iteration_count in range(self.max_tool_iterations + 1): # +1 for final text response
            history = await self.memory.get_history(limit=self.config.get("history_limit", 10)) # Configurable history
            messages_for_llm: List[Message] = []
            
            system_message = self._render_system_prompt(**kwargs.get("template_vars", {}))
            if system_message:
                messages_for_llm.append(system_message)
            
            messages_for_llm.extend(history)

            llm_call_kwargs = kwargs.copy()
            llm_call_kwargs.pop("template_vars", None)
            
            tool_definitions_for_llm: Optional[List[Dict[str, Any]]] = None
            if self.tools:
                tool_definitions_for_llm = [tool.get_openai_tool_definition().model_dump() for tool in self.tools.values()]
                llm_call_kwargs["tools"] = tool_definitions_for_llm
                # tool_choice can be "auto", "none", or {"type": "function", "function": {"name": "my_tool"}}
                llm_call_kwargs["tool_choice"] = self.config.get("tool_choice", "auto") 

            logger.debug(f"Agent '{self.agent_id}' calling LLM (Iter {iteration_count}). History depth: {len(history)}. Tools defined: {bool(tool_definitions_for_llm)}")
            
            # The actual LLM call
            assistant_response_message = await self.llm.chat_completion(
                messages_for_llm, 
                stream=False, # LLMAgent internal loop is non-streaming for now for easier logic
                **llm_call_kwargs 
            )
            await self.memory.add_message(assistant_response_message)

            if not assistant_response_message.tool_calls or iteration_count >= self.max_tool_iterations:
                # No tool calls requested by LLM, or max iterations reached
                logger.info(f"Agent '{self.agent_id}' concluding with textual response. Iter: {iteration_count}.")
                return assistant_response_message

            # LLM requested tool calls
            logger.info(f"Agent '{self.agent_id}' LLM requested tool_calls: {len(assistant_response_message.tool_calls)}")
            
            tool_response_messages: List[Message] = []
            for tool_call in assistant_response_message.tool_calls:
                tool_name = tool_call.function.name
                tool_call_id = tool_call.id
                
                if tool_name not in self.tools:
                    logger.warning(f"Agent '{self.agent_id}': LLM requested unknown tool '{tool_name}'.")
                    tool_result_content = f"Error: Tool '{tool_name}' is not available to me."
                else:
                    tool_to_execute = self.tools[tool_name]
                    tool_result_content = str(await tool_to_execute.execute(tool_call.function.arguments))
                
                tool_response_messages.append(Message(
                    role="tool",
                    tool_call_id=tool_call_id,
                    name=tool_name,
                    content=tool_result_content
                ))
            
            for tr_msg in tool_response_messages: # Add tool results to memory for next LLM turn
                await self.memory.add_message(tr_msg)
            # Loop continues for next LLM call with tool results in history

        # Should not be reached if max_tool_iterations logic is correct
        return Message(role="assistant", content="Error: Exceeded tool processing logic.")