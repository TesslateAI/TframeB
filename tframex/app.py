import asyncio
import inspect
import logging
from typing import Any, Callable, Coroutine, Dict, List, Optional, Union, Type

from .llms import BaseLLMWrapper, OpenAIChatLLM
from .tools import Tool, ToolParameters, ToolParameterProperty
from .memory import BaseMemoryStore, InMemoryMemoryStore
from .primitives import Message, MessageChunk
from .agents.llm_agent import LLMAgent 
from .agents.base import BaseAgent
from .agents.tool_agent import ToolAgent # Ensure ToolAgent is imported if you define agents using it
from .flows import Flow 
from .patterns import BasePattern 
from .flow_context import FlowContext 


logger = logging.getLogger("tframex.app")

class TFrameXApp:
    def __init__(self, default_llm: Optional[BaseLLMWrapper] = None, 
                 default_memory_store_factory: Callable[[], BaseMemoryStore] = InMemoryMemoryStore):
        
        self._tools: Dict[str, Tool] = {}
        self._agents: Dict[str, Dict[str, Any]] = {} 
        self._flows: Dict[str, Flow] = {}
        
        self.default_llm = default_llm
        self.default_memory_store_factory = default_memory_store_factory
        
        if not default_llm and not os.getenv("TFRAMEX_ALLOW_NO_DEFAULT_LLM"): # Allow override for testing/specific use cases
            logger.warning("TFrameXApp initialized without a default LLM. LLM must be provided to run_context or agent if they don't have an override.")

    def tool(self, name: Optional[str] = None, description: Optional[str] = None, 
             parameters_schema: Optional[Dict[str, Dict[str, Any]]] = None) -> Callable:
        """Decorator to register a Python function as a Tool."""
        def decorator(func: Callable[..., Any]) -> Callable:
            tool_name = name or func.__name__
            if tool_name in self._tools:
                raise ValueError(f"Tool '{tool_name}' already registered.")

            parsed_params_schema = None
            if parameters_schema:
                props = {
                    p_name: ToolParameterProperty(**p_def) 
                    for p_name, p_def in parameters_schema.get("properties", {}).items()
                }
                parsed_params_schema = ToolParameters(
                    properties=props, 
                    required=parameters_schema.get("required")
                )

            self._tools[tool_name] = Tool(
                name=tool_name, 
                func=func, 
                description=description,
                parameters_schema=parsed_params_schema
            )
            logger.debug(f"Registered tool: '{tool_name}'")
            return func
        return decorator

    def agent(self, name: Optional[str] = None, 
              system_prompt: Optional[str] = None, 
              tools: Optional[List[str]] = None, 
              llm: Optional[BaseLLMWrapper] = None, 
              memory_store: Optional[BaseMemoryStore] = None, 
              agent_class: type[BaseAgent] = LLMAgent, # Default to LLMAgent
              **agent_config: Any
              ) -> Callable:
        """
        Decorator to define an agent.
        The 'agent_class' parameter determines the type of agent created.
        """
        def decorator(target: Union[Callable, type]) -> Union[Callable, type]:
            agent_name = name or getattr(target, '__name__', str(target))
            if agent_name in self._agents:
                raise ValueError(f"Agent '{agent_name}' already registered.")

            final_config = {
                "system_prompt_template": system_prompt,
                "tool_names": tools or [], 
                "llm_override": llm,
                "memory_override": memory_store,
                "agent_class_ref": agent_class, # Store the explicit agent_class
                **agent_config
            }
            
            # The 'ref' points to the original decorated function or class for identification
            # or if a future agent type directly uses the decorated function's body.
            # For LLMAgent or ToolAgent, their own .run() logic is primary.
            is_class_based_agent = inspect.isclass(target) and issubclass(target, BaseAgent)
            
            self._agents[agent_name] = {
                "type": "custom_class_agent" if is_class_based_agent else "framework_managed_agent", 
                "ref": target, 
                "config": final_config
            }
            logger.debug(f"Registered agent: '{agent_name}' (Framework Managed Class: {agent_class.__name__ if not is_class_based_agent else target.__name__})")
            return target
        return decorator

    def get_tool(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def register_flow(self, flow_instance: Flow) -> None:
        if not isinstance(flow_instance, Flow):
            raise TypeError("Can only register an instance of the Flow class.")
        if flow_instance.flow_name in self._flows:
            raise ValueError(f"Flow with name '{flow_instance.flow_name}' already registered.")
        self._flows[flow_instance.flow_name] = flow_instance
        logger.debug(f"Registered flow: '{flow_instance.flow_name}' with {len(flow_instance.steps)} steps.")

    def get_flow(self, name: str) -> Optional[Flow]:
        return self._flows.get(name)

    def run_context(self, llm_override: Optional[BaseLLMWrapper] = None,
                    context_memory_override: Optional[BaseMemoryStore] = None # Renamed
                    ) -> "TFrameXRuntimeContext":
        ctx_llm = llm_override or self.default_llm
        # Check for LLM presence is now more lenient based on TFRAMEX_ALLOW_NO_DEFAULT_LLM
        # Individual components (agents/flows) will fail if they require an LLM and don't get one.
        
        # context_memory_override is for the TFrameXRuntimeContext itself, not directly for agents
        # unless agents are configured to use it or it's passed down.
        ctx_memory = context_memory_override # Can be None
        
        return TFrameXRuntimeContext(self, llm=ctx_llm, context_memory=ctx_memory)


class TFrameXRuntimeContext:
    def __init__(self, app: TFrameXApp, llm: Optional[BaseLLMWrapper], # LLM can be optional now
                 context_memory: Optional[BaseMemoryStore] = None):
        self._app = app
        self.llm = llm 
        self.context_memory = context_memory 
        self._agent_instances: Dict[str, BaseAgent] = {} 

    async def __aenter__(self) -> "TFrameXRuntimeContext":
        llm_id = self.llm.model_id if self.llm else "None"
        ctx_mem_type = type(self.context_memory).__name__ if self.context_memory else "None"
        logger.info(f"TFrameXRuntimeContext entered. LLM: {llm_id}. Context Memory: {ctx_mem_type}")
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.llm: 
            await self.llm.close()
            logger.info(f"TFrameXRuntimeContext exited. LLM client closed for {self.llm.model_id}.")
        else:
            logger.info("TFrameXRuntimeContext exited. No LLM client to close.")


    def _get_agent_instance(self, agent_name: str) -> BaseAgent:
        # This caching is per TFrameXRuntimeContext instance.
        # If you need agents to be truly stateless across `call_agent` calls within the *same* runtime context,
        # then don't cache in self._agent_instances, or provide a way to get a "fresh" instance.
        if agent_name not in self._agent_instances:
            if agent_name not in self._app._agents:
                raise ValueError(f"Agent '{agent_name}' not registered with the TFrameXApp.")
            
            reg_info = self._app._agents[agent_name]
            agent_config_from_registration = reg_info["config"]

            # Determine LLM for the agent: config override > context's LLM > app's default LLM
            agent_llm = agent_config_from_registration.get("llm_override") or self.llm # self.llm is context's LLM
            if not agent_llm: # If context LLM is also None, try app default
                agent_llm = self._app.default_llm 
            
            # Agent memory: config override > new instance from app's default factory
            agent_memory_override = agent_config_from_registration.get("memory_override")
            agent_memory = agent_memory_override if agent_memory_override is not None \
                           else self._app.default_memory_store_factory()
            
            agent_tools: List[Tool] = []
            if agent_config_from_registration.get("tool_names"):
                for tool_name_ref in agent_config_from_registration["tool_names"]:
                    tool_obj = self._app.get_tool(tool_name_ref)
                    if tool_obj:
                        agent_tools.append(tool_obj)
                    else:
                        logger.warning(f"Tool '{tool_name_ref}' for agent '{agent_name}' not found in app registry.")

            instance_id = f"{agent_name}_ctx{id(self)}" # Unique ID within this context

            # Get the agent class to instantiate from registration info
            AgentClassToInstantiate: Type[BaseAgent] = agent_config_from_registration.get("agent_class_ref")
            if not AgentClassToInstantiate: # Should have been set during @app.agent
                 raise ValueError(f"Agent class reference not found for agent '{agent_name}'. This is an internal error.")

            # Filter out keys specific to TFrameXApp's registration logic that are not part of BaseAgent constructor
            # or are handled explicitly (like llm, memory, tools, system_prompt_template).
            internal_config_keys = {"llm_override", "memory_override", "tool_names", "system_prompt_template", "agent_class_ref"}
            
            # Pass all other keys from agent_config directly to the agent's constructor
            additional_constructor_args = {
                k: v for k, v in agent_config_from_registration.items() if k not in internal_config_keys
            }
            
            # Check if the agent class needs an LLM if one isn't provided.
            # LLMAgent and its children typically do. ToolAgent might not.
            if issubclass(AgentClassToInstantiate, LLMAgent) and not agent_llm:
                 raise ValueError(f"Agent '{agent_name}' (type {AgentClassToInstantiate.__name__}) requires an LLM, but none was available from agent config, context, or app default.")

            self._agent_instances[agent_name] = AgentClassToInstantiate(
                agent_id=instance_id,
                llm=agent_llm, # Can be None if AgentClassToInstantiate doesn't require it
                tools=agent_tools,
                memory=agent_memory,
                system_prompt_template=agent_config_from_registration.get("system_prompt_template"),
                **additional_constructor_args 
            )
            logger.debug(f"Instantiated agent '{instance_id}' (Type: {AgentClassToInstantiate.__name__}) for context.")
        
        return self._agent_instances[agent_name]


    async def call_agent(self, agent_name: str, input_message: Union[str, Message], **kwargs: Any) -> Message:
        if isinstance(input_message, str):
            input_msg_obj = Message(role="user", content=input_message)
        else:
            input_msg_obj = input_message # Assume it's already a Message object
            
        agent_instance = self._get_agent_instance(agent_name)
        # The agent's .run() method is responsible for handling the input_msg_obj,
        # including adding it to its own memory store if its logic dictates.
        return await agent_instance.run(input_msg_obj, **kwargs)

    async def call_tool(self, tool_name: str, arguments_json_str: str) -> Any:
        tool = self._app.get_tool(tool_name)
        if not tool:
            logger.error(f"Attempted to call unregistered tool '{tool_name}'.")
            return {"error": f"Tool '{tool_name}' not found in app registry."}
        return await tool.execute(arguments_json_str)

    async def run_flow(self, flow_ref: Union[str, Flow], 
                       initial_input: Message,
                       initial_shared_data: Optional[Dict[str, Any]] = None
                       ) -> FlowContext: 
        flow_to_run: Optional[Flow] = None
        if isinstance(flow_ref, str):
            flow_to_run = self._app.get_flow(flow_ref)
            if not flow_to_run:
                raise ValueError(f"Flow with name '{flow_ref}' not found.")
        elif isinstance(flow_ref, Flow):
            flow_to_run = flow_ref
        else:
            raise TypeError("flow_ref must be a flow name (str) or a Flow instance.")

        return await flow_to_run.execute(initial_input, self, initial_shared_data=initial_shared_data)

    async def interactive_chat(self, default_flow_name: Optional[str] = None) -> None:
        """
        Starts an interactive CLI session, allowing the user to choose a flow
        and then chat with it. Each user input triggers a new execution of the flow.
        """
        print("\n--- TFrameX Interactive Flow Chat ---")
        
        flow_to_use: Optional[Flow] = None
        if default_flow_name:
            flow_to_use = self._app.get_flow(default_flow_name)
            if flow_to_use:
                print(f"Default flow: '{default_flow_name}'")
            else:
                print(f"Warning: Default flow '{default_flow_name}' not found.")
        
        if not flow_to_use:
            if not self._app._flows:
                print("No flows registered in the application. Exiting interactive chat.")
                return
            
            print("Available flows:")
            for i, name in enumerate(self._app._flows.keys()):
                print(f"  {i+1}. {name}")
            
            while True:
                try:
                    choice_str = await asyncio.to_thread(input, "Select a flow to chat with (number or name, or 'exit'): ")
                    if choice_str.lower() == 'exit': return
                    
                    selected_flow_name: Optional[str] = None
                    if choice_str.isdigit():
                        choice_idx = int(choice_str) - 1
                        if 0 <= choice_idx < len(self._app._flows):
                            selected_flow_name = list(self._app._flows.keys())[choice_idx]
                    else: # Try by name
                        if choice_str in self._app._flows:
                            selected_flow_name = choice_str
                    
                    if selected_flow_name:
                        flow_to_use = self._app.get_flow(selected_flow_name)
                        break
                    else:
                        print("Invalid selection. Please try again.")
                except ValueError:
                    print("Invalid input. Please enter a number or flow name.")
                except KeyboardInterrupt:
                    print("\nExiting.")
                    return

        if not flow_to_use: # Should not happen if selection logic is correct
            print("No flow selected. Exiting.")
            return

        print(f"\n--- Chatting with Flow: '{flow_to_use.flow_name}' ---")
        print(f"Description: {flow_to_use.description or 'No description'}")
        print("Type 'exit' or 'quit' to end this chat session.")

        while True:
            try:
                user_input_str = await asyncio.to_thread(input, "\nYou: ")
                if user_input_str.lower() in ["exit", "quit"]:
                    break
                if not user_input_str.strip():
                    continue
                
                initial_message = Message(role="user", content=user_input_str)
                
                logger.info(f"CLI: Running flow '{flow_to_use.flow_name}' with input: '{user_input_str}'")
                
                # Each chat turn executes the entire flow with the new user message
                final_flow_context: FlowContext = await self.run_flow(flow_to_use, initial_message)
                
                final_output_message = final_flow_context.current_message
                
                print(f"\nFlow Output ({final_output_message.role}):")
                if final_output_message.content:
                    print(f"  Content: {final_output_message.content}")
                
                # Optional: Display tool calls if the final message has them (though ideally handled within flow)
                if final_output_message.tool_calls:
                    print(f"  Final Message Tool Calls (Unprocessed by Flow): {final_output_message.tool_calls}")
                
                # Optional: Display shared data from the flow context for inspection
                if final_flow_context.shared_data:
                    print("  Flow Shared Data (at end of execution):")
                    for key, value in final_flow_context.shared_data.items():
                        # Basic string representation, careful with large objects
                        value_str = str(value)
                        print(f"    {key}: {value_str[:200]}{'...' if len(value_str) > 200 else ''}")

            except KeyboardInterrupt:
                print("\nExiting chat session.")
                break
            except Exception as e:
                print(f"Error during interactive chat with flow '{flow_to_use.flow_name}': {e}")
                logger.error(f"Error in interactive_chat with flow '{flow_to_use.flow_name}'", exc_info=True)
        
        print(f"--- Ended chat with Flow: '{flow_to_use.flow_name}' ---")