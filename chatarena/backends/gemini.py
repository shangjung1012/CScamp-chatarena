import os
import re
from typing import List

from tenacity import retry, stop_after_attempt, wait_random_exponential

from ..message import SYSTEM_NAME, Message
from .base import IntelligenceBackend, register_backend

try:
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    is_gemini_available = True
except (ImportError, KeyError):
    is_gemini_available = False

DEFAULT_MODEL = "gemini-2.0-flash"
END_OF_MESSAGE = "<EOS>"
BASE_PROMPT = f"The messages always end with the token {END_OF_MESSAGE}."

@register_backend
class GeminiChat(IntelligenceBackend):
    """Interface to Google's Gemini chat models."""

    stateful = False
    type_name = "gemini-chat"

    def __init__(
        self,
        temperature: float = 0.7,
        max_tokens: int = 256,
        model: str = DEFAULT_MODEL,
        merge_other_agents_as_one_user: bool = True,
        **kwargs,
    ):
        assert is_gemini_available, "google-generativeai not installed or API key missing"
        super().__init__(
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            merge_other_agents_as_one_user=merge_other_agents_as_one_user,
            **kwargs,
        )

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model = model
        self.merge_other_agent_as_user = merge_other_agents_as_one_user

    @retry(stop=stop_after_attempt(6), wait=wait_random_exponential(min=1, max=60))
    def _get_response(self, messages):
        model = genai.GenerativeModel(self.model)
        chat = model.start_chat(history=[])
        response = chat.send_message(messages, generation_config={
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
        })
        return response.text.strip()

    def query(
        self,
        agent_name: str,
        role_desc: str,
        history_messages: List[Message],
        global_prompt: str = None,
        request_msg: Message = None,
        *args,
        **kwargs,
    ) -> str:

        if global_prompt:
            system_prompt = f"You are a helpful assistant.\n{global_prompt.strip()}\n{BASE_PROMPT}\n\nYour name is {agent_name}.\n\nYour role:{role_desc}"
        else:
            system_prompt = f"You are a helpful assistant. Your name is {agent_name}.\n\nYour role:{role_desc}\n\n{BASE_PROMPT}"

        all_messages = [(SYSTEM_NAME, system_prompt)]
        for msg in history_messages:
            if msg.agent_name == SYSTEM_NAME:
                all_messages.append((SYSTEM_NAME, msg.content))
            else:
                all_messages.append((msg.agent_name, f"{msg.content}{END_OF_MESSAGE}"))

        if request_msg:
            all_messages.append((SYSTEM_NAME, request_msg.content))
        else:
            all_messages.append((SYSTEM_NAME, f"Now you speak, {agent_name}.{END_OF_MESSAGE}"))

        final_input = ""
        for msg in all_messages:
            if msg[0] == SYSTEM_NAME:
                final_input += f"[System]: {msg[1]}\n"
            else:
                final_input += f"[{msg[0]}]: {msg[1]}\n"

        response = self._get_response(final_input)

        response = re.sub(rf"^\s*\[.*]:", "", response).strip()
        response = re.sub(rf"^\s*{re.escape(agent_name)}\s*:", "", response).strip()
        response = re.sub(rf"{END_OF_MESSAGE}$", "", response).strip()

        return response
