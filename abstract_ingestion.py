#!/usr/bin/env python3

"""
Abstract Ingestion Script

This script extracts policies, outcomes and correlations from academic abstracts using Together.ai's models.
It processes a Parquet file containing abstracts from Hugging Face and outputs an enhanced version with extracted features.

Usage:
    python abstract_ingestion.py --input <input_parquet_path> --output <output_parquet_path> [--batch-size <batch_size>]

Required arguments:
    --input: Path to the input Parquet file containing abstracts
    --output: Path where the enhanced Parquet will be saved

Optional arguments:
    --batch-size: Number of abstracts to process in parallel (default: 10)
    --model: Model to use (default: deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free)
"""

import argparse
import concurrent.futures
import json
import os
import time
from typing import Optional, Dict
from threading import Lock
from datetime import datetime, timedelta
from enum import Enum

import pandas as pd
from persist_policies import persist_policies
from together import Together
from prompts import prompt_without_correlation


class ModelType(Enum):
    DEEPSEEK = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"
    LLAMA = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"


class ModelRateLimiter:
    """Rate limiter that enforces different rate limits for different models."""
    
    # Default rate limits (requests per second) for each model
    DEFAULT_RATE_LIMITS = {
        ModelType.DEEPSEEK: 0.1,  # 6 requests per minute
        ModelType.LLAMA: 0.5
    }
    
    def __init__(self, model: str):
        """
        Initialize the rate limiter for a specific model.
        
        Args:
            model: The model identifier string
        """
        self.model = self._get_model_type(model)
        self.rate = self.DEFAULT_RATE_LIMITS[self.model]
        self.last_request_time = datetime.now()
        self.lock = Lock()
        self.min_interval = 1.0 / self.rate
        
    def _get_model_type(self, model: str) -> ModelType:
        """
        Determine the model type from the model string.
        
        Args:
            model: The model identifier string
            
        Returns:
            The corresponding ModelType enum value
            
        Raises:
            ValueError: If the model is not supported
        """
        if model == ModelType.DEEPSEEK.value:
            return ModelType.DEEPSEEK
        elif model == ModelType.LLAMA.value:
            return ModelType.LLAMA
        else:
            raise ValueError(f"Unsupported model: {model}. Supported models are: {[m.value for m in ModelType]}")
    
    def wait(self):
        """Wait if necessary to respect the rate limit."""
        with self.lock:
            now = datetime.now()
            time_since_last = (now - self.last_request_time).total_seconds()
            if time_since_last < self.min_interval:
                sleep_time = self.min_interval - time_since_last
                time.sleep(sleep_time)
            self.last_request_time = datetime.now()
    
    @property
    def requests_per_second(self) -> float:
        """Get the current rate limit in requests per second."""
        return self.rate
    
    def __str__(self) -> str:
        return f"ModelRateLimiter(model={self.model.value}, rate={self.rate} req/s)"


def extract_features_and_correlations(text: str, model: str = ModelType.DEEPSEEK.value, rate_limiter: Optional[ModelRateLimiter] = None) -> Optional[str]:
    """
    Extract features and correlations from an abstract using Together.ai's model.
    
    Args:
        text: The abstract text to analyze
        model: The Together.ai model to use
        rate_limiter: Optional rate limiter to control request rate
        
    Returns:
        JSON string containing extracted features or None if processing failed
    """
    # Generate the prompt using the template
    prompt = prompt_without_correlation(text)
    
    # Define the JSON schema for the response
    json_schema = {
        "type": "object",
        "properties": {
            "GEOGRAPHIC": {"type": "string"},
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
                            }
                        }
                    },
                    "required": ["FACTOR"]
                }
            }
        },
        "required": ["GEOGRAPHIC", "items"]
    }
    
  
    retry_attempts = 2
    retry_delay = 2  # seconds

    for attempt in range(retry_attempts):
        try:
            if rate_limiter:
                rate_limiter.wait()
            
            print(f"Abstract: {text}")
            response = client.chat.completions.create(
                model=model,
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
            extracted_data = json.loads(response.choices[0].message.content.strip())
            print(f"Extracted data: {extracted_data}")
            persisted_data = persist_policies(extracted_data, text)
            extracted_data.update({"abstract": text})
            print(f"Extracted data: {extracted_data}")  
            return extracted_data
        except Exception as e:
            if attempt < retry_attempts - 1:
                print(f"Exception: {e}")
                print(f"Error occurred. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"Failed after {retry_attempts} attempts: {e}")
                return None


def process_abstract(abstract: str, model: str, rate_limiter: Optional[ModelRateLimiter] = None) -> str:
    """
    Process a single abstract.
    
    Args:
        abstract: The abstract text to process
        model: The model to use
        rate_limiter: Optional rate limiter to control request rate
        
    Returns:
        Extracted features as JSON string or "No abstract" if empty
    """
    if not abstract.strip():
        return "No abstract"
    else:
        return extract_features_and_correlations(abstract, model=model, rate_limiter=rate_limiter)


def process_in_batches(df: pd.DataFrame, model: str, batch_size: int = 10, rate_limiter: Optional[ModelRateLimiter] = None) -> list:
    """
    Process abstracts in batches using parallel processing.
    
    Args:
        df: DataFrame containing abstracts
        model: The model to use
        batch_size: Number of abstracts to process in parallel
        rate_limiter: Optional rate limiter to control request rate
        
    Returns:
        List of extracted features for each abstract
    """
    results = []
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i + batch_size]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            batch_results = list(executor.map(
                lambda x: process_abstract(x, model=model, rate_limiter=rate_limiter), 
                batch['abstract']
            ))
        results.extend(batch_results)
    return results


def main():
    parser = argparse.ArgumentParser(description='Process abstracts to extract features and correlations.')
    parser.add_argument('--input', required=True, help='Path to input Parquet file containing abstracts')
    parser.add_argument('--output', required=True, help='Path to save the enhanced Parquet file')
    parser.add_argument('--batch-size', type=int, default=10, help='Number of abstracts to process in parallel')
    parser.add_argument('--model', 
                       choices=[m.value for m in ModelType],
                       default=ModelType.DEEPSEEK.value,
                       help='Model to use for processing')
    
    args = parser.parse_args()
    
    # Get Together.ai API key from environment variable
    together_api_key = os.getenv('TOGETHER_API_KEY')
    if not together_api_key:
        raise ValueError("TOGETHER_API_KEY environment variable is not set")
    
    # Initialize Together client
    global client
    client = Together()
    
    # Initialize model-specific rate limiter
    rate_limiter = ModelRateLimiter(args.model)
    print(f"Using {rate_limiter}")
    
    # Read input data
    print(f"Reading input data from {args.input}")
    df = pd.read_parquet(args.input)
    df["abstract"] = df["abstract"].fillna("").astype(str)
    
    # Process abstracts
    print(f"Processing {len(df)} abstracts in batches of {args.batch_size}")
    df['extracted_features_and_correlations'] = process_in_batches(
        df, 
        model=args.model,
        batch_size=args.batch_size,
        rate_limiter=rate_limiter
    )
    
    # Save results
    print(f"Saving results to {args.output}")
    df.to_parquet(args.output, index=False)
    print("Processing complete!")


if __name__ == "__main__":
    main() 