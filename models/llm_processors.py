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

class LLMExtractor:
    """
    Processor class for using an LLM to process text.
    """
    def __init__(self, model_info):
        self.model_type = model_info['model_type']
        self.model_name = model_info['model_name']
        self.temperature = model_info['temperature']
        self.api_key = model_info['api_key']
        if self.model_type == 'openai':
            self.model = OpenAIModel(api_key=self.api_key, model_name=self.model_name)
        else:
            # Implement other models if needed
            pass

    def extract_class_with_llm(self, text, classes):
        """
        Uses the language model to extract a class label from the text.

        Parameters:
        - text (str): Text to analyze.
        - classes (list): List of possible classes.
        - llm_model (str): Name of the LLM model.
        - llm_temperature (float): Temperature for LLM.

        Returns:
        - final_response (str): The extracted class label.
        """
        prompt = [
            {'role': 'system', 'content': f"You are a classifier that determines the personality class of the author based on the following text. You should return strictly one class from the following list: {', '.join(classes)}. Provide only the class name, no explanations."},
            {'role': 'user', 'content': text}
        ]
        try:
            temperature = self.temperature
            response, status = self.model.generate_response(prompt, temperature)
            return response
        except Exception as e:
            logging.error(f"LLM extraction failed with error: {e}")
            return None

class LLMProcessor:
    """
    Processor class for using an LLM to process text.
    """
    def __init__(self, model_info):
        self.model_type = model_info['model_type']
        self.model_name = model_info['model_name']
        self.temperature = model_info['temperature']
        self.api_key = model_info['api_key']
        if self.model_type == 'openai':
            self.model = OpenAIModel(api_key=self.api_key, model_name=self.model_name)
        else:
            # Implement other models if needed
            pass

    def process_text(self, prompt, temperature=None):
        """
        Processes text using the language model.

        Parameters:
        - prompt (list): List of messages for the conversation.
        - temperature (float): Sampling temperature.

        Returns:
        - response (str): The generated response.
        - status (str): 'OK' if successful, 'fail' otherwise.
        """
        try:
            if temperature is None:
                temperature = self.temperature
            response, status = self.model.generate_response(prompt, temperature)
            return response, status
        except Exception as e:
            logging.error(f"LLM extraction failed with error: {e}")
            return None, 'fail'