import asyncio
import logging
from abc import ABC, abstractmethod
from typing import List, Union, Dict, Any, Tuple, Optional

from .primitives import Message, ToolCall # Added ToolCall
from .flow_context import FlowContext
# Forward declaration for TFrameXRuntimeContext to avoid circular import
if False: # TYPE_CHECKING:
    from .app import TFrameXRuntimeContext


logger = logging.getLogger(__name__)

class BasePattern(ABC):
    """Abstract base class for all execution patterns."""
    def __init__(self, pattern_name: str):
        self.pattern_name = pattern_name
        logger.debug(f"Pattern '{self.pattern_name}' initialized.")

    @abstractmethod
    async def execute(self, flow_ctx: FlowContext, app_runtime: 'TFrameXRuntimeContext') -> FlowContext:
        """
        Executes the pattern.
        Updates the flow_ctx.current_message with the result of this pattern.
        Returns the modified FlowContext.
        """
        pass

    def __str__(self):
        return f"{self.__class__.__name__}(name='{self.pattern_name}')"


class SequentialPattern(BasePattern):
    """Executes a sequence of agents or patterns one after another."""
    def __init__(self, pattern_name: str, steps: List[Union[str, BasePattern]]):
        super().__init__(pattern_name)
        self.steps = steps # List of agent names (str) or BasePattern instances

    async def execute(self, flow_ctx: FlowContext, app_runtime: 'TFrameXRuntimeContext') -> FlowContext:
        logger.info(f"Executing SequentialPattern '{self.pattern_name}' with {len(self.steps)} steps. Input: {str(flow_ctx.current_message.content)[:50]}...")
        original_input_for_pattern = flow_ctx.current_message
        
        for i, step in enumerate(self.steps):
            step_name = str(step) if isinstance(step, BasePattern) else step
            logger.info(f"SequentialPattern '{self.pattern_name}' - Step {i+1}/{len(self.steps)}: Executing '{step_name}'. Current input: {str(flow_ctx.current_message.content)[:50]}...")
            
            if isinstance(step, str): # Agent name
                agent_name = step
                try:
                    # The agent receives the current_message from flow_ctx as its input
                    output_message = await app_runtime.call_agent(agent_name, flow_ctx.current_message)
                    flow_ctx.update_current_message(output_message)
                except Exception as e:
                    logger.error(f"Error in SequentialPattern '{self.pattern_name}' calling agent '{agent_name}': {e}", exc_info=True)
                    error_msg = Message(role="assistant", content=f"Error executing agent '{agent_name}' in sequence '{self.pattern_name}': {e}")
                    flow_ctx.update_current_message(error_msg)
                    return flow_ctx # Halt sequence on error
            elif isinstance(step, BasePattern): # Nested pattern
                try:
                    # The nested pattern also operates on the current flow_ctx
                    flow_ctx = await step.execute(flow_ctx, app_runtime)
                except Exception as e:
                    logger.error(f"Error in SequentialPattern '{self.pattern_name}' executing nested pattern '{step.pattern_name}': {e}", exc_info=True)
                    error_msg = Message(role="assistant", content=f"Error executing nested pattern '{step.pattern_name}' in sequence '{self.pattern_name}': {e}")
                    flow_ctx.update_current_message(error_msg)
                    return flow_ctx # Halt sequence on error
            else:
                logger.error(f"SequentialPattern '{self.pattern_name}': Invalid step type: {type(step)}")
                error_msg = Message(role="assistant", content=f"Invalid step type in sequence '{self.pattern_name}'.")
                flow_ctx.update_current_message(error_msg)
                return flow_ctx

            logger.info(f"SequentialPattern '{self.pattern_name}' - Step {i+1} ('{step_name}') completed. Output: {str(flow_ctx.current_message.content)[:50]}...")
        
        logger.info(f"SequentialPattern '{self.pattern_name}' completed all steps.")
        return flow_ctx


class ParallelPattern(BasePattern):
    """Executes a list of agents or patterns simultaneously with the same initial input."""
    def __init__(self, pattern_name: str, tasks: List[Union[str, BasePattern]]):
        super().__init__(pattern_name)
        self.tasks = tasks # List of agent names (str) or BasePattern instances

    async def execute(self, flow_ctx: FlowContext, app_runtime: 'TFrameXRuntimeContext') -> FlowContext:
        logger.info(f"Executing ParallelPattern '{self.pattern_name}' with {len(self.tasks)} tasks. Input: {str(flow_ctx.current_message.content)[:50]}...")
        initial_input_message_for_parallel_tasks = flow_ctx.current_message

        coroutines = []
        task_identifiers = [] # To keep track of what each coroutine corresponds to

        for task_item in self.tasks:
            task_name = str(task_item) if isinstance(task_item, BasePattern) else task_item
            task_identifiers.append(task_name)

            if isinstance(task_item, str): # Agent name
                # Create a new FlowContext for each parallel branch to avoid interference IF NEEDED.
                # For simple parallel calls with same input, we can pass the initial message.
                # Agents manage their own memory, so flow_ctx.current_message is just the input.
                coroutines.append(app_runtime.call_agent(task_item, initial_input_message_for_parallel_tasks))
            elif isinstance(task_item, BasePattern):
                # For nested patterns in parallel, they need their own FlowContext branch.
                # Create a copy of the flow_ctx state for this branch.
                # IMPORTANT: Deep copy or careful branching of context is needed if patterns modify shared state.
                # For now, assume patterns receive input and produce output without complex side effects on a global shared context.
                # A more robust solution might involve "branching" flow_ctx.
                
                # Simplified: Each parallel pattern gets its own new flow_ctx with the initial input.
                branch_flow_ctx = FlowContext(initial_input=initial_input_message_for_parallel_tasks, shared_data=flow_ctx.shared_data.copy())
                coroutines.append(task_item.execute(branch_flow_ctx, app_runtime)) # Pattern returns FlowContext
            else:
                logger.error(f"ParallelPattern '{self.pattern_name}': Invalid task type: {type(task_item)}")
                # Add a placeholder for this error
                async def error_coro():
                    return Message(role="assistant", content=f"Invalid task type in parallel pattern '{self.pattern_name}'.")
                coroutines.append(error_coro())


        results = await asyncio.gather(*coroutines, return_exceptions=True)
        
        # Aggregate results
        # For now, create a summary message and store individual results as structured data or artifacts.
        aggregated_content_parts = []
        result_artifacts = [] # A2A-like artifacts

        for i, res_item in enumerate(results):
            task_id = task_identifiers[i]
            if isinstance(res_item, Exception):
                logger.error(f"ParallelPattern '{self.pattern_name}' - Task '{task_id}' failed: {res_item}", exc_info=True)
                aggregated_content_parts.append(f"Task '{task_id}' failed: {str(res_item)}")
                result_artifacts.append({
                    "name": f"Result_for_{task_id.replace(' ', '_')}",
                    "parts": [{"type": "text", "text": f"Error: {str(res_item)}"}]
                })
            elif isinstance(res_item, FlowContext): # Result from a nested pattern
                logger.info(f"ParallelPattern '{self.pattern_name}' - Task '{task_id}' (pattern) completed. Output: {str(res_item.current_message.content)[:50]}...")
                aggregated_content_parts.append(f"Task '{task_id}' (pattern) completed. Result: {str(res_item.current_message.content)[:100]}...")
                result_artifacts.append({
                    "name": f"Result_for_{task_id.replace(' ', '_')}",
                    "parts": [{"type": "data", "data": res_item.current_message.model_dump(exclude_none=True)}] # Store full message as data
                })

            elif isinstance(res_item, Message): # Result from an agent call
                logger.info(f"ParallelPattern '{self.pattern_name}' - Task '{task_id}' (agent) completed. Output: {str(res_item.content)[:50]}...")
                aggregated_content_parts.append(f"Task '{task_id}' (agent) completed. Result: {str(res_item.content)[:100]}...")
                result_artifacts.append({
                    "name": f"Result_for_{task_id.replace(' ', '_')}",
                    "parts": [{"type": "data", "data": res_item.model_dump(exclude_none=True)}] # Store full message as data
                })
            else: # Should not happen
                logger.warning(f"ParallelPattern '{self.pattern_name}' - Task '{task_id}' returned unexpected type: {type(res_item)}")
                aggregated_content_parts.append(f"Task '{task_id}' returned unexpected data.")
                result_artifacts.append({
                    "name": f"Result_for_{task_id.replace(' ', '_')}",
                    "parts": [{"type": "text", "text": "Unexpected result type."}]
                })


        summary_content = (f"Parallel execution of '{self.pattern_name}' completed with {len(self.tasks)} tasks.\n"
                           + "\n".join(aggregated_content_parts))
        
        # Create a new message that aggregates the parallel results
        # This message could be inspired by A2A Task object with multiple artifacts
        final_output_message = Message(
            role="assistant", 
            content=summary_content,
            # Simulating A2A artifacts for each parallel task's output
            # For proper A2A, tool_calls would be used to structure this better, or dedicated artifact fields.
            # Here, we'll just put them in a custom field in shared_data or as part of the content string.
            # A simpler way is to have flow_ctx.shared_data["parallel_results"] = results
            # For this example, we'll put a simplified artifact structure into the message content for visibility
            # or use a dedicated "data" part if the next agent can parse it.
            # Let's put structured results into shared_data for now
        )
        flow_ctx.shared_data[f"{self.pattern_name}_results"] = result_artifacts
        flow_ctx.update_current_message(final_output_message)
        
        logger.info(f"ParallelPattern '{self.pattern_name}' completed. Aggregated output: {str(final_output_message.content)[:100]}...")
        return flow_ctx


class RouterPattern(BasePattern):
    """Uses a router_agent to decide which sub_task (agent or pattern) to execute."""
    def __init__(self, pattern_name: str, router_agent_name: str,
                 routes: Dict[str, Union[str, BasePattern]],
                 default_route: Optional[Union[str, BasePattern]] = None):
        super().__init__(pattern_name)
        self.router_agent_name = router_agent_name
        self.routes = routes  # Key: string returned by router_agent, Value: agent_name or BasePattern
        self.default_route = default_route

    async def execute(self, flow_ctx: FlowContext, app_runtime: 'TFrameXRuntimeContext') -> FlowContext:
        logger.info(f"Executing RouterPattern '{self.pattern_name}'. Input: {str(flow_ctx.current_message.content)[:50]}...")
        
        # 1. Call the router agent
        try:
            # Router agent's system prompt should guide it to output a specific route key.
            # For example, it could be asked to respond with ONLY the name of the next agent/task.
            # Or, its response content can be parsed.
            logger.info(f"RouterPattern '{self.pattern_name}': Calling router agent '{self.router_agent_name}'.")
            router_response: Message = await app_runtime.call_agent(self.router_agent_name, flow_ctx.current_message)
            flow_ctx.history.append(router_response) # Add router's decision to this flow's history
            
            route_key = (router_response.content or "").strip()
            logger.info(f"RouterPattern '{self.pattern_name}': Router agent '{self.router_agent_name}' decided route_key: '{route_key}'.")

        except Exception as e:
            logger.error(f"Error calling router agent '{self.router_agent_name}' in RouterPattern '{self.pattern_name}': {e}", exc_info=True)
            error_msg = Message(role="assistant", content=f"Error in router agent '{self.router_agent_name}': {e}")
            flow_ctx.update_current_message(error_msg)
            return flow_ctx

        # 2. Determine the target based on route_key
        target_step = self.routes.get(route_key)
        if target_step is None:
            logger.warning(f"RouterPattern '{self.pattern_name}': Route key '{route_key}' not found in routes. Using default route if available.")
            target_step = self.default_route
        
        if target_step is None:
            logger.error(f"RouterPattern '{self.pattern_name}': No route found for key '{route_key}' and no default route defined.")
            error_msg = Message(role="assistant", content=f"Routing error: No path for '{route_key}'.")
            flow_ctx.update_current_message(error_msg)
            return flow_ctx

        # 3. Execute the target step
        target_name = str(target_step) if isinstance(target_step, BasePattern) else target_step
        logger.info(f"RouterPattern '{self.pattern_name}': Executing routed step '{target_name}'.")
        
        # The input to the routed step is the original input to the RouterPattern
        # (or potentially the router_agent's response, depending on design - for now, original input)
        # Let's use flow_ctx.current_message, which was the input to the router agent.
        # Or, if router should pass its output, flow_ctx.current_message = router_response

        # For this version, the routed agent/pattern gets the *original* input that the router saw.
        # If router_response should be the input, then flow_ctx.update_current_message(router_response) before this.
        # For now, we assume router just decides, and the next step gets the same context the router got.

        if isinstance(target_step, str): # Agent name
            try:
                output_message = await app_runtime.call_agent(target_step, flow_ctx.current_message) # Input is what router agent received
                flow_ctx.update_current_message(output_message)
            except Exception as e:
                logger.error(f"Error in RouterPattern '{self.pattern_name}' calling routed agent '{target_step}': {e}", exc_info=True)
                error_msg = Message(role="assistant", content=f"Error executing routed agent '{target_step}': {e}")
                flow_ctx.update_current_message(error_msg)
        elif isinstance(target_step, BasePattern): # Nested pattern
            try:
                flow_ctx = await target_step.execute(flow_ctx, app_runtime)
            except Exception as e:
                logger.error(f"Error in RouterPattern '{self.pattern_name}' executing routed pattern '{target_step.pattern_name}': {e}", exc_info=True)
                error_msg = Message(role="assistant", content=f"Error executing routed pattern '{target_step.pattern_name}': {e}")
                flow_ctx.update_current_message(error_msg)
        
        logger.info(f"RouterPattern '{self.pattern_name}' completed. Final output from routed step: {str(flow_ctx.current_message.content)[:50]}...")
        return flow_ctx


class DiscussionPattern(BasePattern):
    """Multiple agents discuss a topic for a set number of rounds."""
    def __init__(self, pattern_name: str,
                 participant_agent_names: List[str],
                 discussion_rounds: int = 1,
                 moderator_agent_name: Optional[str] = None, # Optional moderator for summarizing rounds
                 stop_phrase: Optional[str] = None): # Phrase an agent can say to end discussion
        super().__init__(pattern_name)
        if not participant_agent_names:
            raise ValueError("DiscussionPattern requires at least one participant agent.")
        self.participant_agent_names = participant_agent_names
        self.discussion_rounds = discussion_rounds
        self.moderator_agent_name = moderator_agent_name
        self.stop_phrase = stop_phrase.lower() if stop_phrase else None


    async def execute(self, flow_ctx: FlowContext, app_runtime: 'TFrameXRuntimeContext') -> FlowContext:
        logger.info(f"Executing DiscussionPattern '{self.pattern_name}' with {len(self.participant_agent_names)} participants for {self.discussion_rounds} rounds. Topic: {str(flow_ctx.current_message.content)[:50]}...")
        
        # Discussion history is managed within the flow_ctx.history for this pattern's scope.
        # The initial message in flow_ctx.current_message is the topic.
        
        current_discussion_topic = flow_ctx.current_message # This is the initial prompt/topic

        for round_num in range(1, self.discussion_rounds + 1):
            logger.info(f"DiscussionPattern '{self.pattern_name}' - Round {round_num}/{self.discussion_rounds}")
            
            round_messages: List[Tuple[str, Message]] = [] # (agent_name, message)

            for agent_name in self.participant_agent_names:
                logger.info(f"DiscussionPattern '{self.pattern_name}' - Round {round_num}: Agent '{agent_name}' speaking. Current topic/context: {str(current_discussion_topic.content)[:50]}...")
                
                # Each agent gets the current discussion topic/latest message as input.
                # The agent's own memory will contain its prior turns if it's stateful.
                # The 'current_discussion_topic' acts as the prompt for this turn.
                try:
                    # Create a message that frames the current state of discussion for the agent
                    # This could be just the last message, or a summary.
                    # For simplicity, each agent gets the `current_discussion_topic` which is either
                    # the initial prompt or the moderator's summary from the previous round.
                    
                    # Important: The agent should "know" it's in a discussion.
                    # Its system prompt might need to be tailored for this.
                    # The input message content could also be prefixed e.g., "Continuing discussion on X, [agent_name], your thoughts?"
                    # For now, we pass the current_discussion_topic directly.
                    
                    # To provide more context, we could build a small history for this agent turn
                    # input_for_agent_turn = flow_ctx.current_message # This is the previous agent's message or initial topic
                    # What if the agent needs the whole discussion so far?
                    # This implies passing more than just `current_discussion_topic`.
                    # Let `current_discussion_topic` be the prompt to this agent.
                    
                    # Get agent instance to potentially clear its memory if we want stateless turns for discussion pattern
                    # agent_instance = app_runtime._get_agent_instance(agent_name)
                    # await agent_instance.memory.clear() # Make each turn stateless to the overall discussion history
                    # Or, let agents use their memory to be consistent.

                    # The input to the agent is the current_discussion_topic message.
                    agent_response: Message = await app_runtime.call_agent(agent_name, current_discussion_topic)
                    # The agent_response is added to flow_ctx.history by call_agent if it modifies memory.
                    # For this discussion, we are interested in the agent's direct response.
                    
                    # We should add *this specific* agent_response to the flow_ctx's main history
                    # to record the discussion turn.
                    flow_ctx.history.append(agent_response) # Record agent's contribution to the main flow history
                                        
                    round_messages.append((agent_name, agent_response))
                    current_discussion_topic = agent_response # Next agent in this round responds to this.

                    logger.info(f"DiscussionPattern '{self.pattern_name}' - Round {round_num}: Agent '{agent_name}' responded: {str(agent_response.content)[:50]}...")

                    if self.stop_phrase and self.stop_phrase in (agent_response.content or "").lower():
                        logger.info(f"DiscussionPattern '{self.pattern_name}': Agent '{agent_name}' said stop phrase. Ending discussion.")
                        # Final message is this agent's response
                        flow_ctx.update_current_message(agent_response)
                        return flow_ctx
                except Exception as e:
                    logger.error(f"Error during DiscussionPattern '{self.pattern_name}' with agent '{agent_name}': {e}", exc_info=True)
                    # Continue with other agents or end round? For now, log and continue.
                    error_response = Message(role="assistant", content=f"Agent {agent_name} encountered an error: {e}")
                    round_messages.append((agent_name, error_response))
                    current_discussion_topic = error_response # Next agent sees this error.


            if not round_messages: # Should not happen if participants exist
                logger.warning(f"DiscussionPattern '{self.pattern_name}' - Round {round_num}: No messages generated.")
                break # End discussion if a round produces nothing

            # After all participants in a round have spoken:
            if self.moderator_agent_name and round_num < self.discussion_rounds:
                # Prepare input for moderator: a summary of the round's discussion
                moderator_input_content_parts = [f"Summary of Round {round_num} for discussion on '{str(flow_ctx.current_message.content)[:50]}...':"]
                for name, msg in round_messages:
                    moderator_input_content_parts.append(f"- {name}: {msg.content}")
                
                moderator_input_str = "\n".join(moderator_input_content_parts)
                moderator_input_msg = Message(role="user", content=moderator_input_str + "\n\nPlease moderate this round and provide a summary or next question for the group.")
                
                logger.info(f"DiscussionPattern '{self.pattern_name}' - Round {round_num}: Calling moderator '{self.moderator_agent_name}'.")
                try:
                    moderator_response: Message = await app_runtime.call_agent(self.moderator_agent_name, moderator_input_msg)
                    flow_ctx.history.append(moderator_response) # Record moderator's summary
                    current_discussion_topic = moderator_response # Next round starts with moderator's output
                    logger.info(f"DiscussionPattern '{self.pattern_name}' - Round {round_num}: Moderator responded: {str(moderator_response.content)[:50]}...")
                except Exception as e:
                    logger.error(f"Error calling moderator agent '{self.moderator_agent_name}': {e}", exc_info=True)
                    current_discussion_topic = Message(role="assistant", content=f"Moderator error: {e}. Continuing without moderation for next round.")
            elif round_messages: # No moderator, or last round
                # The last message from the last participant becomes the topic for the next round or the final output
                current_discussion_topic = round_messages[-1][1]


        # Final message of the discussion is the last `current_discussion_topic`
        flow_ctx.update_current_message(current_discussion_topic)
        logger.info(f"DiscussionPattern '{self.pattern_name}' completed. Final message: {str(flow_ctx.current_message.content)[:50]}...")
        return flow_ctx

# Add other patterns like LoopPattern later.