import os
from preprocessing.preprocess import preprocess_data
from services.analysis_cleaning import  preprocess_flow


def run_preprocessing_news(input_path: str) -> str:
    base_dir = os.path.dirname(__file__)
    output_path = os.path.join(base_dir, "..", "test_data", "preprocessed_data_news.json")
    output_path = os.path.abspath(output_path)
    preprocess_data(input_path, output_path)
    return output_path

def run_preprocessing_general(input_path: str) -> str:
    base_dir = os.path.dirname(__file__)
    output_path = os.path.join(base_dir, "..", "test_data", "preprocessed_data_general.json")
    output_path = os.path.abspath(output_path)
    preprocess_data(input_path, output_path)
    return output_path

def run_preprocessing_focus(input_path: str) -> str:
    base_dir = os.path.dirname(__file__)
    output_path = os.path.join(base_dir, "..", "test_data", "preprocessed_data_focus.json")
    output_path = os.path.abspath(output_path)
    preprocess_data(input_path, output_path)
    return output_path

def run_coinfinder_focus(input_path: str) -> str:
    base_dir = os.path.dirname(__file__)
    output_path = os.path.join(base_dir, "..", "test_data", "preprocessed_data_run_coinfinder_focus.json")
    output_path = os.path.abspath(output_path)
    preprocess_data(input_path, output_path)
    return output_path

def run_coinflow_focus(input_path: str) -> str:
    base_dir = os.path.dirname(__file__)
    output_path = os.path.join(base_dir, "..", "test_data", "preprocessed_data_run_coinflow_focus.json")
    output_path = os.path.abspath(output_path)
    preprocess_flow(input_path, output_path)
    return output_path

