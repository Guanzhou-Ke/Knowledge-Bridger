from openai import OpenAI
from .constants import (VLM_BASE_URL, VLM_MODEL_NAME, 
                        API_KEY)
                

class LMAgent:

    def __init__(self, base_url, api_key, model_name):
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.model_name = model_name


    def chat(self, msg, max_token=256, temperature=0.):
        chat_completion_from_url = self.client.chat.completions.create(
            messages=msg,
            model=self.model_name,
            max_tokens=max_token,
            temperature=temperature
        )
        result = chat_completion_from_url.choices[0].message.content
        return result


def create_default_vlm_agent():
    return LMAgent(VLM_BASE_URL, API_KEY, VLM_MODEL_NAME)

def create_sdxl_and_refiner():
    pass

def create_sdxl():
    pass