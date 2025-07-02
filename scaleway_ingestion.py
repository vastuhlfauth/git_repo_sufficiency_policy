#!/usr/bin/env python3

"""
Scaleway Abstract Ingestion Script

This script extracts policies, outcomes and correlations from academic abstracts using Scaleway's OpenAI API.
It processes a Parquet file containing abstracts and outputs a list of Pydantic objects with extracted features.

Usage:
    python scaleway_ingestion.py --input <input_parquet_path> --output <output_json_path>

Required arguments:
    --input: Path to the input Parquet file containing abstracts
    --output: Path where the JSON file with extracted features will be saved
"""

import argparse
import os
import json
import time
from typing import Optional, List, Dict, Tuple
from multiprocessing import Pool, cpu_count
import backoff

import pandas as pd
from openai import OpenAI
# from persist_policies import persist_policies
from prompts import prompt_without_correlation


def init_client():
    """Initialize the OpenAI client for each worker process."""
    global client
    scaleway_api_key = os.getenv('SCW_SECRET_KEY', "") # Amine
    client = OpenAI(
        base_url="https://api.scaleway.ai/v1",
        api_key=scaleway_api_key
    )


@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def extract_features_and_correlations(abstract: str, openalex_id: str, doi: str) -> Tuple[Optional[str], Dict]:
    """
    Extract features and correlations from an abstract using Scaleway's OpenAI API.
    Includes retry logic with exponential backoff.
    """
    if not abstract.strip():
        return None, {"tokens": 0, "time": 0}

    # Generate the prompt using the template
    prompt = prompt_without_correlation(abstract)

    # Define the JSON schema for the response
    json_schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "properties": {
                        "FACTOR": {
                            "type": "object",
                            "additionalProperties": {
                                "type": "object",
                                "properties": {
                                    "CORRELATION": {"type": "string"}
                                },
                                "required": ["CORRELATION"]
                            },
                        },
                        "SECTOR": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            }
                        },
                        "GEOGRAPHIC": {"type": "string"},
                    },
                    "required": ["FACTOR, GEOGRAPHIC", "SECTOR"]
                }
            }
        },
        "required": ["items"]
    }

    start_time = time.time()
    try:
        print(f"Processing abstract: {abstract[:100]}...")  # Print only first 100 chars
        response = client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=[
                {"role": "user", "content": prompt}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "schema": json_schema,
                    "name": "output_schema",
                    "strict": True
                }
            }
        )

        # Calculate metrics
        end_time = time.time()
        processing_time = end_time - start_time

        # Get token usage
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens

        metrics = {
            "tokens": {
                "prompt": prompt_tokens,
                "completion": completion_tokens,
                "total": total_tokens
            },
            "time": processing_time
        }

        extracted_data = json.loads(response.choices[0].message.content.strip())
        # persisted_data = persist_policies(extracted_data, text)
        extracted_data.update({
            "abstract": abstract,
            "openalex_id": openalex_id,
            "doi": doi
        })
        print(f"Extracted data: {extracted_data}")
        print(f"Metrics: {metrics}")
        return extracted_data, metrics

    except Exception as e:
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"Error processing abstract: {e}")
        raise  # Let backoff handle the retry


def process_row(row_data: Tuple, row_num: int, total_rows: int, output_file: str) -> Tuple[Optional[Dict], int]:
    """Process a single row of data and save results."""
    abstract, openalex_id, doi = row_data
    total_tokens = 0
    
    try:
        result, metrics = extract_features_and_correlations(abstract, openalex_id, doi)
        if result is not None:
            total_tokens = metrics["tokens"]["total"]
            
            # Save result immediately
            with open(output_file, 'r+') as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    existing_data = []
                
                existing_data.append(result)
                f.seek(0)
                json.dump(existing_data, f, indent=2)
                f.truncate()
            
            print(f"Completed row {row_num}/{total_rows}")
            return result, total_tokens
    except Exception as e:
        print(f"Failed to process row {row_num} after retries: {e}")
    
    return None, total_tokens


def process_in_parallel(df: pd.DataFrame, output_file: str) -> Tuple[List[Dict], int]:
    """
    Process abstracts in parallel, one row at a time.
    
    Args:
        df: DataFrame containing abstracts
        output_file: Path to output JSON file
        
    Returns:
        Tuple of (list of processed results, total tokens used)
    """
    # Initialize output file
    if not os.path.exists(output_file):
        with open(output_file, 'w') as f:
            json.dump([], f)
    
    # Prepare row data
    total_rows = len(df)
    row_data = list(zip(df['abstract'], df['openalex_id'], df['doi']))
    
    # Process rows in parallel
    num_workers = min(cpu_count(), 4)  # Use up to 4 workers
    print(f"Processing {total_rows} rows using {num_workers} workers")
    
    all_results = []
    total_tokens = 0
    
    with Pool(processes=num_workers, initializer=init_client) as pool:
        # Create arguments for each row
        row_args = [(data, i+1, total_rows, output_file) for i, data in enumerate(row_data)]
        
        # Process rows in parallel
        row_results = pool.starmap(process_row, row_args)
        
        for result, tokens in row_results:
            if result is not None:
                all_results.append(result)
                total_tokens += tokens
    
    return all_results, total_tokens


def main():
    parser = argparse.ArgumentParser(description='Process abstracts to extract features and correlations.')
    parser.add_argument('--input', required=True, help='Path to input Parquet file containing abstracts')
    parser.add_argument('--output', required=True, help='Path to save the JSON file with extracted features')

    args = parser.parse_args()

    # Read input data
    print(f"Reading input data from {args.input}")
    df = pd.read_parquet(args.input)
    df["abstract"] = df["abstract"].fillna("").astype(str)

    # Start timing the entire process
    total_start_time = time.time()

    # Process abstracts using multiprocessing
    results, total_tokens = process_in_parallel(df, args.output)

    # Calculate total processing time
    total_time = time.time() - total_start_time

    # Calculate and print summary statistics
    print("\nSummary Statistics:")
    print(f"Total abstracts processed: {len(results)}")
    print(f"Total tokens used: {total_tokens}")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average tokens per request: {total_tokens / len(results) if results else 0:.2f}")
    print(f"Average time per request: {total_time / len(results) if results else 0:.2f} seconds")
    print("Processing complete!")


if __name__ == "__main__":
    main()
