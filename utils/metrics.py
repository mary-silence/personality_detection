# utils/metrics.py

import pandas as pd
import numpy as np
import logging
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.metrics.agreement import AnnotationTask

# Configure logging
logging.basicConfig(
    filename='metrics.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)


def calculate_classification_metrics(df, true_label, predicted_label, experiment=None, dataset=None, class_mapping=None):
    """
    Calculates classification metrics and returns them as a DataFrame.

    Parameters:
    - df (DataFrame): Input DataFrame containing true and predicted labels.
    - true_label (str): Column name for true labels.
    - predicted_label (str): Column name for predicted labels.
    - experiment (str): Experiment name (optional).
    - dataset (str): Dataset name (optional).
    - class_mapping (dict): Dictionary for class mapping.

    Returns:
    - report_df (DataFrame): DataFrame containing classification metrics.
    """
    if class_mapping is None:
        logging.critical('Error in class mapping')
        return 'Fail'

    # Extract unique labels
    labels = list(class_mapping.values())

    df[predicted_label] = df[predicted_label].str.lower().map(class_mapping)

    # Handle unknown classes
    unknown_pred = df[predicted_label].isna().sum()
    total_samples = len(df)
    unknown_percentage = (unknown_pred / total_samples) * 100

    # Fill NaN with a separate code for unknown classes
    df[predicted_label] = df[predicted_label].fillna(max(class_mapping.values())) 


    # Calculate classification report
    report = classification_report(
        df[true_label], df[predicted_label], labels=labels, output_dict=True, zero_division=0
    )

    # Extract main metrics
    metrics = {
        'Accuracy': report['accuracy'],
        'Precision (macro avg)': report['macro avg']['precision'],
        'Recall (macro avg)': report['macro avg']['recall'],
        'F1-score (macro avg)': report['macro avg']['f1-score'],
        'Precision (weighted avg)': report['weighted avg']['precision'],
        'Recall (weighted avg)': report['weighted avg']['recall'],
        'F1-score (weighted avg)': report['weighted avg']['f1-score']
    }

    # Convert metrics to DataFrame
    metrics_df = pd.DataFrame([metrics])

    # Add experiment and dataset information
    if experiment:
        metrics_df['Experiment'] = experiment
    if dataset:
        metrics_df['Dataset'] = dataset

    # Log metrics calculation
    logging.info(f"Classification metrics calculated for experiment '{experiment}' on dataset '{dataset}'.")

    return metrics_df


def get_cosine_similarity(df1, df2, text_column):
    """
    Calculates the mean cosine similarity between the text columns of two DataFrames.

    Parameters:
    - df1 (DataFrame): First DataFrame.
    - df2 (DataFrame): Second DataFrame.
    - text_column (str): Name of the text column to compare.

    Returns:
    - mean_cos_sim (float): Mean cosine similarity.
    """
    # Check if DataFrames have the same number of rows
    if len(df1) != len(df2):
        logging.error("DataFrames have different number of rows.")
        return None

    # Check if indices match
    if not df1.index.equals(df2.index):
        logging.error("DataFrames have mismatched indices.")
        return None

    # Join DataFrames on index
    df_joined = df1.join(df2, lsuffix='_1', rsuffix='_2')

    text_col1 = f"{text_column}_1"
    text_col2 = f"{text_column}_2"

    cos_similarities = []
    for i, row in df_joined.iterrows():
        texts = [str(row[text_col1]), str(row[text_col2])]
        vectorizer = CountVectorizer(stop_words="english")
        vectors = vectorizer.fit_transform(texts)
        cos_sim = cosine_similarity(vectors)[0, 1]
        cos_similarities.append(cos_sim)

    mean_cos_sim = np.mean(cos_similarities)
    return mean_cos_sim


def get_cosine_similarity_from_files(file1, file2, text_column):
    """
    Calculates the mean cosine similarity between the text columns of two CSV files.

    Parameters:
    - file1 (str): Path to the first CSV file.
    - file2 (str): Path to the second CSV file.
    - text_column (str): Name of the text column to compare.

    Returns:
    - mean_cos_sim (float): Mean cosine similarity.
    """
    try:
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
    except Exception as e:
        logging.error(f"Error reading files: {e}")
        return None

    return get_cosine_similarity(df1, df2, text_column)


def get_agreement_metrics(df1, df2, category_column):
    """
    Calculates agreement metrics between two sets of annotations.

    Parameters:
    - df1 (DataFrame): First DataFrame with annotations.
    - df2 (DataFrame): Second DataFrame with annotations.
    - category_column (str): Name of the column with category labels.

    Returns:
    - metrics (dict): Dictionary containing agreement metrics.
    """
    # Check if DataFrames have the same number of rows
    if len(df1) != len(df2):
        logging.error("DataFrames have different number of rows.")
        return None

    # Check if indices match
    if not df1.index.equals(df2.index):
        logging.error("DataFrames have mismatched indices.")
        return None

    y_true = df1[category_column].tolist()
    y_pred = df2[category_column].tolist()

    data = []
    for i, label in enumerate(y_true):
        data.append(('coder1', str(i), label))
    for i, label in enumerate(y_pred):
        data.append(('coder2', str(i), label))

    try:
        task = AnnotationTask(data)
        avg_Ao = task.avg_Ao()
        kappa = task.kappa()
        alpha = task.alpha()
        S_value = task.S()
        pi_value = task.pi()

        metrics = {
            'Average Observed Agreement (avg_Ao)': avg_Ao,
            'Kappa': kappa,
            'Alpha': alpha,
            'S': S_value,
            'Pi': pi_value
        }
        return metrics
    except Exception as e:
        logging.error(f"Error in computing agreement metrics: {e}")
        return None


def get_agreement_metrics_from_files(file1, file2, category_column):
    """
    Calculates agreement metrics between two CSV files.

    Parameters:
    - file1 (str): Path to the first CSV file.
    - file2 (str): Path to the second CSV file.
    - category_column (str): Name of the column with category labels.

    Returns:
    - metrics (dict): Dictionary containing agreement metrics.
    """
    try:
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
    except Exception as e:
        logging.error(f"Error reading files: {e}")
        return None

    return get_agreement_metrics(df1, df2, category_column)
