import os
import glob
from utils.data_processing import process_annotated_files
from prompts.prompt_templates import load_prompts
from configs.parse_patterns import load_parse_patterns

def run_processing_pipeline(
    annotated_files=None,
    process_all=False,
    extraction_model_info=None,
    analyzer_model_info=None
):
    """
    Runs the processing and analysis pipeline on annotated files.

    Parameters:
    - annotated_files (list): List of annotated files to process.
    - process_all (bool): If True, processes all annotated files.
    - extraction_model_info (dict): Information about the model used for processing.
    - analyzer_model_info (dict): Information about the model used for analysis.
    """
    prompts = load_prompts()
    parse_patterns = load_parse_patterns()

    if process_all:
        annotated_files = glob.glob('results/annotated/*.csv')

    if not annotated_files:
        print("No files to process.")
        return

    process_annotated_files(
        filenames=annotated_files,
        prompts=prompts,
        parse_patterns=parse_patterns,
        extraction_model_info=extraction_model_info,
        analyzer_model_info=analyzer_model_info
    )
