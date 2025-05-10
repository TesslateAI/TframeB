import logging
from typing import List, Union, Optional, Dict, Any

from .primitives import Message
from .flow_context import FlowContext
from .patterns import BasePattern
# Forward declaration for TFrameXRuntimeContext
if False: #TYPE_CHECKING:
    from .app import TFrameXRuntimeContext


logger = logging.getLogger(__name__)

class Flow:
    """
    Represents a defined sequence of operations (agents or patterns) to be executed.
    """
    def __init__(self, flow_name: str, description: Optional[str] = None):
        self.flow_name = flow_name
        self.description = description
        self.steps: List[Union[str, BasePattern]] = [] # str for agent_name, or BasePattern instance
        logger.debug(f"Flow '{self.flow_name}' initialized.")

    def add_step(self, step: Union[str, BasePattern]):
        """Adds a step to the flow. A step can be an agent name or a Pattern instance."""
        if not isinstance(step, (str, BasePattern)):
            raise TypeError("Flow step must be an agent name (str) or a BasePattern instance.")
        self.steps.append(step)
        logger.debug(f"Flow '{self.flow_name}': Added step '{str(step)}'. Total steps: {len(self.steps)}.")
        return self # For chaining

    async def execute(self, initial_input: Message, app_runtime: 'TFrameXRuntimeContext',
                      initial_shared_data: Optional[Dict[str, Any]] = None) -> FlowContext:
        """
        Executes the flow with the given initial input and runtime context.
        Returns the final FlowContext after all steps.
        """
        logger.info(f"Executing Flow '{self.flow_name}' with {len(self.steps)} steps. Initial input: {str(initial_input.content)[:50]}...")
        
        flow_ctx = FlowContext(initial_input=initial_input, shared_data=initial_shared_data)

        for i, step in enumerate(self.steps):
            step_name = str(step) if isinstance(step, BasePattern) else step
            logger.info(f"Flow '{self.flow_name}' - Step {i+1}/{len(self.steps)}: Executing '{step_name}'. Current input: {str(flow_ctx.current_message.content)[:50]}...")
            
            try:
                if isinstance(step, str): # Agent name
                    output_message = await app_runtime.call_agent(step, flow_ctx.current_message)
                    flow_ctx.update_current_message(output_message)
                elif isinstance(step, BasePattern): # Pattern instance
                    # Patterns are responsible for updating flow_ctx.current_message internally
                    flow_ctx = await step.execute(flow_ctx, app_runtime) 
                else: # Should not happen due to add_step validation
                    raise TypeError(f"Invalid step type in flow '{self.flow_name}': {type(step)}")

                logger.info(f"Flow '{self.flow_name}' - Step {i+1} ('{step_name}') completed. Output: {str(flow_ctx.current_message.content)[:50]}...")

                # Check for early exit signal from shared_data if needed (e.g., flow_ctx.shared_data.get("STOP_FLOW"))
                if flow_ctx.shared_data.get("STOP_FLOW", False):
                    logger.info(f"Flow '{self.flow_name}' - STOP_FLOW signal received. Halting execution.")
                    break

            except Exception as e:
                logger.error(f"Error during Flow '{self.flow_name}' at step '{step_name}': {e}", exc_info=True)
                error_msg = Message(role="assistant", content=f"Error in flow '{self.flow_name}' at step '{step_name}': {e}")
                flow_ctx.update_current_message(error_msg)
                # Decide if flow should halt on error or continue (for now, it halts by returning)
                return flow_ctx
        
        logger.info(f"Flow '{self.flow_name}' completed. Final output: {str(flow_ctx.current_message.content)[:50]}...")
        return flow_ctx