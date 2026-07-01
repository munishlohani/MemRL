# memp/agent/history.py
from typing import List, Dict, Any

class EpisodeHistory:
    """
    Manages the action-observation history for a single episode.
    This acts as the agent's short-term or "working" memory.
    """
    def __init__(self):
        self.trajectory: List[Dict[str, str]] = []
        self.messages: List[Dict[str, str]] = []
        self._last_action: str | None = None

    def add_step(self, observation: str):
        """
        Adds a new step to the history. A "step" consists of the action
        that was taken previously and the new observation that resulted from it.
        
        Args:
            observation (str): The new observation received from the environment.
        """
        # A step is only complete when we have both the action and the resulting observation.
        if self._last_action is not None:
            action = self._last_action
            self.trajectory.append({
                "action": action,
                "observation": observation,
            })
            self.append_message(
                {
                    "role": "user",
                    "content": f"Action: {action}\nObservation: {observation}",
                }
            )
        
        # The last action has now been consumed and paired with an observation.
        self._last_action = None

    def record_action(self, action: str):
        """
        Records the action chosen by the agent, holding it until the next
        observation is received.
        
        Args:
            action (str): The action command chosen by the agent.
        """
        self._last_action = action

    def append_message(self, message: Dict[str, Any]) -> None:
        """Append a conversational message to the episode transcript."""
        if not isinstance(message, dict):
            return

        role = str(message.get("role", "")).strip()
        content = str(message.get("content", "")).strip()
        if not role or not content:
            return

        payload: Dict[str, str] = {"role": role, "content": content}
        name = message.get("name")
        if isinstance(name, str) and name.strip():
            payload["name"] = name.strip()
        self.messages.append(payload)

    def get_messages(self) -> List[Dict[str, str]]:
        """Return the current conversational transcript."""
        return [dict(message) for message in self.messages]

    def get_formatted_history(self, max_steps: int = 10) -> str:
        """
        Returns the last few steps of the episode as a formatted string,
        suitable for inclusion in a prompt.

        Args:
            max_steps (int): The maximum number of recent steps to include.

        Returns:
            str: A formatted, human-readable string of the recent history.
        """
        if not self.trajectory:
            return "You are at the beginning of the task. No steps taken yet."
        
        # Get the most recent steps
        recent_steps = self.trajectory[-max_steps:]
        
        formatted = []
        # We calculate the step number relative to the full trajectory
        start_step_num = len(self.trajectory) - len(recent_steps) + 1
        for i, step in enumerate(recent_steps):
            step_num = start_step_num + i
            formatted.append(f"--- Step {step_num} ---\nAction: {step['action']}\nObservation: {step['observation']}")
            
        return "\n".join(formatted)

    def clear(self):
        """Resets the history for a new episode."""
        self.trajectory = []
        self.messages = []
        self._last_action = None
