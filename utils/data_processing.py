import pandas as pd
import numpy as np
import re
import logging
import json
from transformers import pipeline
from models.llm_connectors import OpenAIModel #написать подробно
from models.llm_processors import LLMExtractor
from utils.response_processors import get_answer_processor

# Configure logging
logging.basicConfig(
    filename='data_processing.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)

# Load

def load_data(filename, sample_size=None):
    """
    Loads data from a CSV file.

    Parameters:
    - filename (str): Path to the CSV file.
    - sample_size (int, optional): Sample size. If None, all data is loaded.

    Returns:
    - df (DataFrame): Loaded DataFrame.
    """
    df = pd.read_csv(filename)
    if sample_size:
        df = df[:sample_size]
    return df

def load_prompts():
    """
    Loads prompt templates from a JSON file.

    Returns:
    - prompts (dict): Dictionary containing prompts.
    """
    json_path = 'prompts/trait_prompts.json'
    with open(json_path, 'r', encoding='utf-8') as f:
        prompts = json.load(f)
    return prompts

def load_patterns():
    """
    Loads patterns from a JSON file.
    """
    json_path = 'extraction_patterns/data_preparation_patterns.json'
    with open(json_path, 'r', encoding='utf-8') as f:
        patterns = json.load(f)
    return patterns

def load_analyzer():
    """
    Loads patterns from a JSON file.
    """
    json_path = 'prompts/analyzers.json'
    with open(json_path, 'r', encoding='utf-8') as f:
        patterns = json.load(f)
    return patterns

# Extract trait from model response

def extract_personality_from_json(text, classifications=None):
    """
    Classifies personality based on JSON content within the text.

    Parameters:
    - text (str): Text that may contain JSON.
    - classifications (list): List of possible classes.
    """
    if classifications is None:
        classifications = ['Introvert', 'Extravert', 'Ambivert']
    
    # Attempt to find JSON in the text
    json_text = text.strip()
    
    # Regular expression to find JSON within code blocks
    code_block_pattern = re.compile(r'```(?:json)?\s*(\{.*?\})\s*```', re.DOTALL | re.IGNORECASE)
    match = code_block_pattern.search(json_text)
    
    if match:
        # Extract JSON from code block
        json_content = match.group(1)
    else:
        # Assume the entire text is JSON
        json_content = json_text
    
    try:
        # Attempt to load text as JSON
        data = json.loads(json_content)
    except json.JSONDecodeError:
        # If loading fails, return None
        return None

    if not data:
        return None  # Пустой JSON

    # Check for 'answer' or 'Answer' keys
    answer = data.get('answer') or data.get('Answer')
    if not answer:
        return None

    # Ensure the 'answer' value is a string
    if not isinstance(answer, str):
        return None

    # Create regex pattern based on classifications
    escaped_classifications = [re.escape(word) for word in classifications]
    pattern_str = r'\b(' + '|'.join(escaped_classifications) + r')\w*\b'
    pattern = re.compile(pattern_str, re.IGNORECASE)
    match = pattern.search(answer)

    if match:
        # Return the matched class with first letter capitalized
        classification_found = match.group(1).capitalize()
        return classification_found

    # If nothing is found
    return None

    def extract_personality(df, column_name, labels_configs, tait_patterns, use_huggingface=False, use_llm=False, llm_model=None, llm_temperature=0):
    """
    Processes a DataFrame by classifying texts based on patterns.

    Parameters:
    - df (DataFrame): Input DataFrame.
    - column_name (str): Column name containing text for analysis.
    - labels 
    - use_huggingface (bool): Whether to use HuggingFace for classification.
    - use_llm (bool): Whether to use LLM for class extraction.
    - llm_model (str): LLM model name.
    - llm_temperature (float): Temperature for LLM.
    """
    df['result'] = np.nan
    df['extraction_method'] = 'error'

    if use_huggingface:
        # Initialize the HuggingFace zero-shot classifier
        classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
        candidate_labels = list(labels_configs.values())

    for index, row in df.iterrows():
        text = row[column_name].lower()
        classified = False

        # Try classification using JSON parsing
        json_class = extract_personality_from_json(text, classifications=candidate_labels)
        if json_class:
            df.at[index, 'result'] = json_class
            df.at[index, 'extraction_method'] = 'json'
            classified = True
            logging.info(f"Row {index}: Classified as {json_class} using JSON parser.")
            continue

        # Try classification using regex patterns
        for trait_score, pattern in trait_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                df.at[index, 'result'] = labels_configs[trait_score]
                df.at[index, 'extraction_method'] = 'regex'
                classified = True
                logging.info(f"Row {index}: Classified as {trait} using regex.")
                break

        if not classified and use_huggingface:
            # Use HuggingFace zero-shot classification
            result = classifier(text, candidate_labels)
            top_score = result['scores'][0]
            if top_score >= 0.95:
                df.at[index, 'result'] = result['labels'][0]
                df.at[index, 'extraction_method'] = 'huggingface'
                classified = True
                logging.info(f"Row {index}: Classified as {result['labels'][0]} using HuggingFace with confidence {top_score}.")
            else:
                logging.info(f"Row {index}: HuggingFace confidence below threshold with score {top_score}.")

        if not classified and use_llm and llm_model is not None:
            # Use LLM for extraction
             # Initialize LLMProcessor for processing
            llm_extractor = LLMExtractor(extraction_model_info)
            llm_class = extract_class_with_llm(text, candidate_labels)
            if llm_class:
                df.at[index, 'result'] = llm_class
                df.at[index, 'extraction_method'] = 'llm'
                classified = True
                logging.info(f"Row {index}: Classified as {llm_class} using LLM.")
            else:
                logging.info(f"Row {index}: LLM failed to classify.")

        if not classified:
            # If all methods fail, set 'error'
            df.at[index, 'result'] = 'error'
            df.at[index, 'extraction_method'] = 'error'
            logging.info(f"Row {index}: Classification failed. Set as 'error'.")

    return df

# Process model response

def process_annotated_files(filenames, prompts, parse_patterns, extraction_model_info, analyzer_model_info):
    """
    Processes annotated files using processors and analyzers.

    Parameters:
    - filenames (list): List of filenames to process.
    - prompts (dict): Loaded prompts.
    - parse_patterns (dict): Loaded parse patterns.
    - extraction_model_info (dict): Information about the model used for processing.
    - analyzer_model_info (dict): Information about the model used for analysis.
    """
    for filename in filenames:
        df = pd.read_csv(filename)
        technique = df['technique'].iloc[0]
        trait = df['trait'].iloc[0]
        version = df['version'].iloc[0]

        # Method settings
        prompt_data = prompts[trait][technique][version]
        method_settings = {
            'standard_processor': get_answer_processor(technique),
            'use_llm_processor': True,  # User can change this setting
            'llm_processor_prompt': prompt_data.get('llm_processor_prompt')
        }

        # Initialize LLMProcessor for processing
        llm_processor = LLMProcessor(extraction_model_info)

        # Process model responses
        df = process_model_responses(
            df,
            'model_response',
            method_settings,
            llm_processor=llm_processor
        )

        # Save processed results in 'results/processed' folder
        processed_filename = filename.replace('results/annotated', 'results/processed').replace('.csv', '_processed.csv')
        save_results(df, processed_filename)



def analyze_processed_files(filenames, analyzer, analuzer_type analyzer_model_info):
        # Analyzer settings
        analyzer_prompt = analyzers['default']['system_prompt']
        categories = prompts['analyzers']['default']['categories']
        analyzer_model = LLMProcessor(analyzer_model_info)

        # Analyze processed texts
        df = analyze_processed_texts(
            df,
            'processed_text',
            analyzer_prompt,
            analyzer_model,
            categories
        )

        # Save analysis results in 'results/analyzed' folder
        analysis_filename = processed_filename.replace('results/processed', 'results/analyzed').replace('_processed.csv', '_analyzed.csv')
        save_results(df, analysis_filename)
