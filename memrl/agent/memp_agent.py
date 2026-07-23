# FILE: memp/agent/memp_agent.py

import logging
from typing import List, Dict, Any, Tuple, Optional
import copy
import ast

from .base import BaseAgent
from .history import EpisodeHistory
from . import prompts
from memrl.providers.llm import OpenAILLM

logger = logging.getLogger(__name__)

class MempAgent(BaseAgent):
    """
    A stateless agent that uses an LLM to make decisions.
    It receives all necessary context (history, retrieved memories) from an
    external controller (the Runner) at the moment of action.
    """
    def __init__(self, llm_provider: OpenAILLM):
        # The agent is now independent of the memory service.
        self.llm = llm_provider

    def reset(self, task_description: str) -> None:
        """Resets the agent for a new episode and retrieves relevant long-term memories."""
        self.task_description = task_description.strip()
        logger.info(f"Agent has been reset for new task: '{self.task_description}'")

    def _split_retrieved_memory_content(self, raw_content: str) -> Tuple[str, str, str]:
        """Split retrieved memory into header/body and describe the body type."""
        if '\n\nTRAJECTORY:\n' in raw_content:
            header, body = raw_content.split('\n\nTRAJECTORY:\n', 1)
            return header, body, "trajectory"

        if '\n\nFailed approach:\n' in raw_content:
            header, body = raw_content.split('\n\nFailed approach:\n', 1)
            return header, body, "trajectory"

        if raw_content.startswith("Task:") and '\n\n' in raw_content:
            header, body = raw_content.split('\n\n', 1)
            return header, body, "unknown"

        return "", raw_content, "raw"

    def _parse_trajectory_list(
        self, trajectory_str: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Parse a stored trajectory if the content looks like a Python list literal."""
        trajectory_str = trajectory_str.strip()
        if not trajectory_str.startswith("["):
            return None

        trajectory_list = ast.literal_eval(trajectory_str)
        if not isinstance(trajectory_list, list):
            raise ValueError("Retrieved memory trajectory is not a list.")
        return trajectory_list

    def _clean_trajectory_messages(self, trajectory_list: List[Dict[str, Any]]) -> str:
        """Keep only the relevant user/assistant turns from a stored trajectory."""
        turn_idx = -1
        for i, msg in enumerate(trajectory_list):
            if (
                isinstance(msg, dict)
                and msg.get("role") == "user"
                and isinstance(msg.get("content", ""), str)
                and "Now, it's your turn" in msg["content"]
            ):
                turn_idx = i

        if turn_idx != -1:
            trajectory_list = trajectory_list[turn_idx:]

        clean_trajectory = []
        for message in trajectory_list:
            if not isinstance(message, dict):
                continue

            role = message.get("role")
            content = message.get("content", "")
            if role == "assistant":
                clean_trajectory.append(f"> {content}")
            elif role == "user" and isinstance(content, str):
                clean_trajectory.append(content)

        return "\n".join(clean_trajectory)

    def _format_retrieved_memory(self, raw_content: str) -> str:
        """
        [NEW HELPER METHOD]
        Parses the raw memory content to extract only the most useful parts
        (SCRIPT and the core Thought/Action/Observation sequence), removing
        redundant headers, system prompts, and old task descriptions.
        """
        raw_content = (raw_content or "").strip()
        if not raw_content:
            return ""

        header, body, body_type = self._split_retrieved_memory_content(raw_content)
        header = header.strip()
        body = body.strip()
        clean_parts = []

        if 'SCRIPT:' in header:
            script_part = header.split('SCRIPT:', 1)[1].strip()
            if script_part:
                clean_parts.append(f"Archived Script:\n{script_part}")
        if 'What went wrong:' in header:
            reflection_part = header.split('What went wrong:', 1)[1].strip()
            if reflection_part:
                clean_parts.append(f"Archived Script:\n{reflection_part}")

        try:
            trajectory_list = self._parse_trajectory_list(body)
        except Exception as e:
            logger.warning(
                "Could not parse retrieved memory trajectory, using raw content. Error: %s",
                e,
            )
            trajectory_label = (
                "Archived Trajectory"
                if body_type == "trajectory" or body.startswith("[")
                else "Archived Script"
            )
            if body:
                clean_parts.append(f"{trajectory_label}:\n{body}")
                return "\n\n".join(clean_parts)
            return raw_content

        if trajectory_list is not None:
            clean_trajectory = self._clean_trajectory_messages(trajectory_list)
            if clean_trajectory:
                clean_parts.append("Archived Trajectory:\n" + clean_trajectory)
            return "\n\n".join(clean_parts) or raw_content

        if body and body != raw_content:
            body_label = "Archived Trajectory" if body_type == "trajectory" else "Archived Script"
            clean_parts.append(f"{body_label}:\n{body}")
            return "\n\n".join(clean_parts)

        return "\n\n".join(clean_parts) or raw_content
        
    def _construct_messages(
        self,
        task_description: str,
        retrieved_memories: List[Dict],
        admissible_commands: Optional[List[str]] = None,
    ) -> List[Dict[str, str]]:
        """
        Builds the message list in a conversational ReAct style.
        """
        # 1. Start with the system prompt
        messages = [{"role": "system", "content": prompts.SYSTEM_PROMPT}]

        # 2. Add retrieved memories as additional context for the agent
        if retrieved_memories:
            successful_mems = retrieved_memories.get('successed', [])
            failed_mems = retrieved_memories.get('failed', [])

            successful_mems_formatted = [
                self._format_retrieved_memory(mem['content']) for mem in successful_mems
            ] if successful_mems else []

            failed_mems_formatted = [
                self._format_retrieved_memory(mem['content']) for mem in failed_mems
            ] if failed_mems else []

            memory_parts = [
                "You have the following memories from your own past experiences. "
                "Use them to help you if they are relevant:"
            ]

            if successful_mems_formatted:
                memory_parts.append(
                    "--- SUCCESSFUL MEMORIES (Examples to follow) ---\n" +
                    "\n".join(successful_mems_formatted)
                )

            if failed_mems_formatted:
                memory_parts.append(
                    "--- FAILED MEMORIES (Examples to avoid or learn from) ---\n" +
                    "\n".join(failed_mems_formatted)
                )

            if successful_mems_formatted or failed_mems_formatted:
                memory_context = "\n\n".join(memory_parts)
                messages.append({"role": "system", "content": memory_context})

        # 4. Add the current task description as the new user prompt
        # The history of the current task will be appended in the `act` method
        current_task_prompt = f"Now, it's your turn to solve a new task.\n{task_description}"
        current_task_prompt += f"\n{self._format_admissible_commands(admissible_commands)}"
        messages.append({"role": "user", "content": current_task_prompt})
        # logger.info(f"\nPrompt {messages}")
        return messages

    @staticmethod
    def _format_admissible_commands(admissible_commands: Optional[List[str]]) -> str:
        if not admissible_commands:
            return "Admissible actions: (none available)"
        listed = "\n".join(f"- {command}" for command in admissible_commands)
        return f"Admissible actions:\n{listed}"

    def _parse_action(self, llm_response: str) -> str:
        """
        Extracts the 'Action:' part from the ReAct response.
        """
        if llm_response:
            if "Action:" in llm_response:
                return llm_response.split("Action:")[-1].strip()
            # Fallback if the model doesn't follow the format correctly
            logger.warning(f"Could not find 'Action:' in LLM response. Returning the full response: '{llm_response}'")
            return llm_response.strip()
        else:
            return 'look around'
    def act(
        self,
        observation: str,
        history_messages: List[Dict[str, str]],
        first_step: bool = False,
        admissible_commands: Optional[List[str]] = None,
    ):
        """
        Agent performs one step of action generation.
        Ensures robustness: if LLM fails or returns invalid output, action=None is returned.
        """
        import json

        observation_content = (
            f"Observation: {observation.strip()}\n{self._format_admissible_commands(admissible_commands)}"
        )

        current_messages = copy.deepcopy(history_messages)
        if not first_step:
            current_messages.append({"role": "user", "content": observation_content})

        filtered_messages = []
        for i, m in enumerate(current_messages):
            if m.get("content") is None:
                logger.warning(f"[Message Filter] Message {i} has None content, removed: {m}")
                continue
            if isinstance(m.get("content"), str) and not m["content"].strip():
                logger.warning(f"[Message Filter] Message {i} has empty content, removed: {m}")
                continue
            filtered_messages.append(m)
        current_messages = filtered_messages

        logger.debug("Querying LLM for the next action...")

        response = None
        try:
            response = self.llm.generate(current_messages)
        except Exception as e:
            logger.error("LLM generation failed: %s", str(e))
            logger.error("Messages before failure:\n%s", json.dumps(current_messages, indent=2, ensure_ascii=False))
            response = None  # fallback

        if not first_step:
            history_messages.append({"role": "user", "content": observation_content})
        history_messages.append({"role": "assistant", "content": response if response is not None else "No response."})

        action = None
        if response:
            try:
                action = self._parse_action(response)
            except Exception as e:
                logger.warning(f"Action parsing failed for response='{response}': {e}")
                action = "inventory"

        return action



    def get_trajectory(self) -> List[Dict[str, str]]:
        """Returns the complete trajectory for the finished episode."""
        pass
