# experiment_runner.py

import datetime
import pandas as pd
import logging
from models.language_models import OpenAIModel
from utils.data_processing import process_dataframe, save_results
from utils.metrics import calculate_classification_metrics

def run_experiment(
    model_info,
    trait,
    technique,
    version,
    prompts,
    patterns,
    df,
    extraction_model,
    class_mapping,
    run_id=None,
    use_huggingface=False,
    use_llm=False
):
    """
    Runs a single experiment with specified parameters.
    """
    # Get prompts based on trait, technique, and version
    try:
        prompt_data = prompts[trait][technique][version]
    except KeyError:
        print(f"Prompt not found for trait '{trait}', technique '{technique}', version '{version}'")
        return

    system_prompt = prompt_data.get('system_prompt', '')
    user_prompt_template = prompt_data['user_prompt']

    # Select the model
    if model_info['model_type'] == 'openai':
        model = OpenAIModel(
            api_key=model_info['api_key'],
            model_name=model_info['model_name']
        )
    else:
        # Handle other model types if necessary
        pass

    # Function to generate responses
    def generate_response(text):
        user_prompt = user_prompt_template.format(text=text)
        if system_prompt:
            prompt = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ]
        else:
            prompt = [
                {'role': 'user', 'content': user_prompt}
            ]
        response, _ = model.generate_response(prompt, temperature=model_info['temperature'])
        return response

    # Copy DataFrame for the experiment
    df_experiment = df.copy()

    # Get model responses
    df_experiment['model_response'] = df_experiment['TEXT'].apply(generate_response)

    # Process responses
    trait_patterns = patterns[trait]
    #use_huggingface = True  # Set according to your preference
    #use_llm = True  # Set according to your preference

    # Prepare the extraction model
    if extraction_model['model_type'] == 'openai':
        llm_model = extraction_model['model_name']
        llm_temperature = extraction_model['temperature']
    else:
        # Handle other extraction model types
        llm_model = None
        llm_temperature = None

    df_experiment = process_dataframe(
        df_experiment,
        'model_response',
        trait_patterns,
        use_huggingface=use_huggingface,
        use_llm=use_llm,
        llm_model=llm_model,
        llm_temperature=llm_temperature
    )

    # Include technique and version in the DataFrame
    df_experiment['technique'] = technique
    df_experiment['version'] = version
    df_experiment['trait'] = trait  # Add trait for completeness
    df_experiment['model_name'] = model_info['model_name']

    # Save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") if run_id is None else run_id
    output_filename = f"results/annotated/{model_info['model_name']}_{trait}_{technique}_v{version}_temp{model_info['temperature']}_{timestamp}.csv"
    save_results(df_experiment, output_filename)

    # Collect statistics for the report
    num_errors = df_experiment[df_experiment['result'] == 'error'].shape[0]
    total = len(df_experiment)
    extraction_methods = df_experiment['extraction_method'].value_counts()

    # Save stats for reporting
    experiment_stats = {
        'model_name': model_info['model_name'],
        'trait': trait,
        'technique': technique,
        'version': version,
        'errors': num_errors,
        'total': total,
        'extraction_methods': extraction_methods.to_dict()
    }

    return {
        'model_name': model_info['model_name'],
        'trait': trait,
        'technique': technique,
        'version': version,
        'timestamp': timestamp,
        'df': df_experiment,
        'metrics': None,  # Metrics can be calculated later
        'stats': experiment_stats
    }

def run_all_experiments(
    models_to_run,
    traits,
    technique_versions,
    num_iterations,
    prompts,
    patterns,
    df,
    extraction_model,
    class_mapping,
    use_huggingface,
    use_llm
):
    """
    Runs all experiments based on the configurations provided.
    """
    results = []
    for iteration in range(num_iterations):
        for model_info in models_to_run:
            for trait in traits:
                for technique, versions in technique_versions.items():
                    for version in versions:
                        # Generate a unique run_id
                        run_id = f"run_{iteration}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        result = run_experiment(
                            model_info=model_info,
                            trait=trait,
                            technique=technique,
                            version=version,
                            prompts=prompts,
                            patterns=patterns,
                            df=df,
                            extraction_model=extraction_model,
                            class_mapping=class_mapping,
                            run_id=run_id,
                            use_huggingface = use_huggingface,
                            use_llm = use_llm
                        )
                        if result:
                            results.append(result)
    return results

def generate_report(results=None, filenames=None):
    """
    Generates a report from experiment results or specified CSV files.

    Parameters:
    - results (list): List of results from the current run (optional).
    - filenames (list): List of paths to CSV files with results (optional).
    """
    if filenames:
        # Load data from CSV files
        results = []
        for filename in filenames:
            df_experiment = pd.read_csv(filename)
            # Extract information from the DataFrame
            stats = {
                'model_name': df_experiment['model_name'].iloc[0] if 'model_name' in df_experiment.columns else 'unknown',
                'trait': df_experiment['trait'].iloc[0] if 'trait' in df_experiment.columns else 'unknown',
                'technique': df_experiment['technique'].iloc[0] if 'technique' in df_experiment.columns else 'unknown',
                'version': df_experiment['version'].iloc[0] if 'version' in df_experiment.columns else 'unknown',
                'errors': df_experiment[df_experiment['result'] == 'error'].shape[0],
                'total': len(df_experiment),
                'extraction_methods': df_experiment['extraction_method'].value_counts().to_dict()
            }
            results.append({'stats': stats})
    elif results is None:
        print("Please provide either 'results' or 'filenames'")
        return

    # Generate report
    for res in results:
        stats = res['stats']
        print(f"Model: {stats['model_name']}, Trait: {stats['trait']}, Technique: {stats['technique']}, Version: {stats['version']}")
        print(f"Total Samples: {stats['total']}, Errors: {stats['errors']}")
        print("Extraction Methods Used:")
        for method, count in stats['extraction_methods'].items():
            print(f" - {method}: {count}")
        print("\n")

def compute_metrics_from_results(results, class_mapping, true_label):
    """
    Calculates metrics from the results data.

    Parameters:
    - results (list): List of experiment results.
    - class_mapping (dict): Class mapping dictionary.
    """
    all_metrics = []
    for res in results:
        df_experiment = res['df']
        metrics_df = calculate_classification_metrics(
            df_experiment,
            true_label=true_label,
            predicted_label='result',
            experiment=f"{res['technique']}_v{res['version']}",
            dataset=res['trait'],
            class_mapping=class_mapping,
        )
        res['metrics'] = metrics_df
        all_metrics.append(metrics_df)

    # Combine all metrics into a single DataFrame if needed
    combined_metrics = pd.concat(all_metrics, ignore_index=True)
    # Display or save metrics
    print(combined_metrics)
    return combined_metrics

def compute_metrics_from_files(filenames, class_mapping, true_label):
    """
    Calculates metrics from specified CSV files.

    Parameters:
    - filenames (list): List of paths to CSV files with results.
    - class_mapping (dict): Class mapping dictionary.
    """
    all_metrics = []
    for filename in filenames:
        df_experiment = pd.read_csv(filename)
        technique = df_experiment['technique'].iloc[0] if 'technique' in df_experiment.columns else 'unknown'
        version = df_experiment['version'].iloc[0] if 'version' in df_experiment.columns else 'unknown'
        trait = df_experiment['trait'].iloc[0] if 'trait' in df_experiment.columns else 'unknown'
        metrics_df = calculate_classification_metrics(
            df_experiment,
            true_label=true_label,
            predicted_label='result',
            experiment=f"{technique}_v{version}",
            dataset=trait,
            class_mapping=class_mapping,
        )
        all_metrics.append(metrics_df)
    combined_metrics = pd.concat(all_metrics, ignore_index=True)
    # Display or save metrics
    print(combined_metrics)
    return combined_metrics
