import os
import time
import logging
from abc import ABC, abstractmethod
import openai

# Configure logging
logging.basicConfig(
    filename='language_models.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)

class LanguageModel(ABC):
    """
    Abstract base class for language models.
    """
    @abstractmethod
    def generate_response(self, prompt, temperature):
        pass

class OpenAIModel(LanguageModel):
    def __init__(self, api_key, model_name):
        self.api_key = api_key
        self.model_name = model_name
        openai.api_key = self.api_key

    def generate_response(self, prompt, temperature, max_retries=5):
        client = openai.OpenAI(
            # This is the default and can be omitted
            api_key=self.api_key,
        )
        for attempt in range(1, max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=prompt,
                    temperature=temperature
                )
                final_response = response.choices[0].message.content
                logging.info(f"OpenAI: Successfully received response on attempt {attempt}")
                return final_response, 'OK'
            except openai.OpenAIError as e:
                wait_time = 2 ** attempt
                logging.error(f"OpenAI: Attempt {attempt} failed with error: {e}. Retrying in {wait_time} seconds.")
                time.sleep(wait_time)
        logging.critical("OpenAI: Max retries exceeded. Returning 'GPT Fail'.")
        return 'GPT Fail', 'fail'

# Add other models similarly
