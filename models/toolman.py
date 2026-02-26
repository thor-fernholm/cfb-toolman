from typing import Any, Dict
import os
import json
import sys
import copy
import os

from sympy import false

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts.prompts import SimpleTemplatePrompt
from utils.utils import *
import requests
import json
from pydantic import BaseModel, Field
from typing import List, Optional, Any
from openai.types.chat import ChatCompletion

class ToolmanModel:
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.api_url = "http://localhost:8080/cfb"

    def __call__(self, prefix, prompt: SimpleTemplatePrompt, **kwargs: Any):
        filled_prompt = prompt(**kwargs)
        prediction = self._predict(prefix, filled_prompt, **kwargs)
        return prediction
    
    @retry(max_attempts=10)
    def _predict(self, prefix, text, **kwargs):
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": text}
            ],
            "system_prompt": prefix,
            "temperature": 0.0
        }

        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            # get completion and responses (errors)
            completion_res = response.json()['completion']
            completion = ChatCompletion.model_validate(completion_res)
            return completion.choices[0].message.content
            # completion = OpenAIResponse(**response_json)
            # return completion.choices[0].message.content
        except Exception as e:
            print(f"Exception: {e}")
            return None


class FunctionCallToolman(ToolmanModel):
    def __init__(self, model_name, enable_ptc=True):
        super().__init__(None)
        # super().__init__(model_name)
        self.model_name = model_name
        self.enable_ptc = enable_ptc
        self.messages = []
        self.toolman_history = []
        # self.toolman_calls = []
        self.new_tool_responses = []

    @retry(max_attempts=5, delay=10)
    def __call__(self, messages, tools=None, **kwargs: Any):
        if "function_call" not in json.dumps(messages, ensure_ascii=False):
            self.messages = copy.deepcopy(messages)

        payload = {
            "model": self.model_name,
            "messages": self.messages,
            "temperature": 0.0,
            "tools": tools,
            "tool_choice": "auto", # unused?
            "max_tokens": 2048, # unused?
            "enable_ptc": self.enable_ptc,
            "toolman_history": self.toolman_history, # pass back toolman history
            # "toolman_calls": self.toolman_calls # pass newest calls
            "new_tool_responses": self.new_tool_responses
        }

        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            # get completion and responses (errors)
            completion_res = response.json()['completion']
            self.toolman_history = response.json()['toolman_history'] # will be passed back into toolman
            # self.toolman_calls = response.json()['toolman_calls'] # will be passed back into toolman

            # reset new tool responses for next turn!
            self.new_tool_responses = []

            completion = ChatCompletion.model_validate(completion_res)
            msg_dict =  completion.choices[0].message.model_dump()
            return to_dict_obj(msg_dict)
            # completion = OpenAIResponse(**response_json)
            # return completion.choices[0].message
        except Exception as e:
            print(f"Exception: {e}")
            return None

class DictObj(dict):
    def __init__(self, *args, **kwargs):
        super(DictObj, self).__init__(*args, **kwargs)

    def __getattr__(self, key):
        # Do not intercept special methods (like __deepcopy__, __reduce__)
        if key.startswith("__"):
            raise AttributeError(key)

        try:
            return self[key]
        except KeyError:
            return None

    def __setattr__(self, key, value):
        self[key] = value

def to_dict_obj(obj):
    if isinstance(obj, dict):
        return DictObj({k: to_dict_obj(v) for k, v in obj.items()})
    elif isinstance(obj, list):
        return [to_dict_obj(i) for i in obj]
    else:
        return obj

class Function(BaseModel):
    name: str
    arguments: str  # Keeps it as string, you parse it when needed

class ToolCall(BaseModel):
    id: str
    type: str
    function: Function

class Message(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None

class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class OpenAIResponse(BaseModel):
    id: str
    choices: List[Choice]
    usage: Optional[Usage] = None


if __name__ == "__main__":
    model = ToolmanModel("OpenAI/gpt-4o-mini")
    response = model("You are a helpful assistant.", SimpleTemplatePrompt(template=("What is the capital of France?"), args_order=[]))
    print(response)