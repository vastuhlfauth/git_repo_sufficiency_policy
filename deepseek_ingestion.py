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
from persist_policies import is_in_base, persist_policies, ExtractedPolicies
from prompts import prompt_without_correlation
import logging
from datetime import datetime
from sqlmodel import SQLModel, Session, create_engine
from sqlalchemy.pool import QueuePool

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://u4axloluqibskgvdikuy:g2rXgpHSbztokCbFxSyR@bk8htvifqendwt1wlzat-postgresql.services.clever-cloud.com:7327/bk8htvifqendwt1wlzat"
)

# Create engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,  # Maximum number of connections to keep
    max_overflow=10,  # Maximum number of connections that can be created beyond pool_size
    pool_timeout=30,  # Seconds to wait before giving up on getting a connection from the pool
    pool_pre_ping=True  # Enable connection health checks
)

def init_client():
    """Initialize the OpenAI client and database session for each worker process."""
    global client, session
    deepseek_api_key = os.getenv('DS_SECRET_KEY', "sk-da2decacb37a45ddad71aaf79cac2505")  # DS key
    client = OpenAI(
        base_url="https://api.deepseek.com",
        api_key=deepseek_api_key
    )
    # Create a new session for this process
    session = Session(engine)

@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def extract_features_and_correlations(abstract: str, openalex_id: str, doi: str) -> Tuple[Optional[str], Dict]:
    """
    Extract features and correlations from an abstract using Deepseek's API.
    Includes retry logic with exponential backoff.
    """
    if not abstract.strip():
        return None, {"tokens": 0, "time": 0, "error": "Empty abstract"}

    # Generate the prompt using the template
    prompt = prompt_without_correlation(abstract)

    start_time = time.time()
    if is_in_base(session, openalex_id):
        return None, {"tokens": 0, "time": 0, "error": "Already processed"}
    try:
        logger.debug(f"Processing abstract: {abstract[:100]}...")  # Print only first 100 chars
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "user", "content": prompt}
            ],
            response_format={'type': 'json_object'},
            stream=False
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
            "time": processing_time,
            "error": None
        }

        extracted_data = json.loads(response.choices[0].message.content.strip())
        extracted_data.update({
            "abstract": abstract,
            "openalex_id": openalex_id,
            "doi": doi
        })
        return extracted_data, metrics

    except Exception as e:
        end_time = time.time()
        processing_time = end_time - start_time
        logger.error(f"Error processing abstract: {str(e)}")
        return None, {
            "tokens": 0,
            "time": processing_time,
            "error": str(e)
        }

def process_row(row_data: Tuple, row_num: int, total_rows: int, output_file: str) -> Tuple[Optional[Dict], Dict]:
    """Process a single row of data and save results."""
    abstract, openalex_id, doi = row_data
    metrics = {
        "tokens": 0,
        "time": 0,
        "error": None,
        "status": "skipped"
    }

    try:
        result, metrics = extract_features_and_correlations(abstract, openalex_id, doi)
        if result is not None:
            metrics["status"] = "success"
            # Persist to database
            persist_policies(session, result, abstract, openalex_id, doi)
        else:
            metrics["status"] = "failed"
    except Exception as e:
        logger.error(f"Failed to process row {row_num}: {str(e)}")
        metrics["status"] = "error"
        metrics["error"] = str(e)

    return result, metrics

def process_in_parallel(df: pd.DataFrame, output_file: str) -> Tuple[List[Dict], Dict]:
    """
    Process abstracts in parallel, one row at a time.
    
    Args:
        df: DataFrame containing abstracts
        output_file: Path to output JSON file
        
    Returns:
        Tuple of (list of processed results, processing statistics)
    """
    # Prepare row data
    total_rows = len(df)
    row_data = list(zip(df['abstract'], df['openalex_id'], df['doi']))

    # Process rows in parallel
    num_workers = 16 # min(cpu_count(), 16)  # Use up to 16 workers
    logger.info(f"Processing {total_rows} rows using {num_workers} workers")

    all_results = []
    stats = {
        "total_rows": total_rows,
        "processed": 0,
        "successful": 0,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
        "total_tokens": 0,
        "total_time": 0
    }

    with Pool(processes=num_workers, initializer=init_client) as pool:
        # Create arguments for each row
        row_args = [(data, i + 1, total_rows, output_file) for i, data in enumerate(row_data)]

        # Process rows in parallel
        for result, metrics in pool.starmap(process_row, row_args):
            stats["processed"] += 1
            stats["total_tokens"] += metrics.get("tokens", 0)
            stats["total_time"] += metrics.get("time", 0)

            if metrics["status"] == "success":
                stats["successful"] += 1
                if result is not None:
                    all_results.append(result)
            elif metrics["status"] == "failed":
                stats["failed"] += 1
            elif metrics["status"] == "skipped":
                stats["skipped"] += 1
            elif metrics["status"] == "error":
                stats["errors"] += 1

            # Log progress every 100 rows
            if stats["processed"] % 100 == 0:
                logger.info(f"Progress: {stats['processed']}/{total_rows} rows processed")
                logger.info(f"Status: {stats['successful']} successful, {stats['failed']} failed, "
                          f"{stats['skipped']} skipped, {stats['errors']} errors")

    return all_results, stats

def main():
    parser = argparse.ArgumentParser(description='Process abstracts to extract features and correlations.')
    parser.add_argument('--input', required=True, help='Path to input Parquet file containing abstracts')
    parser.add_argument('--output', required=True, help='Path to save the JSON file with extracted features')

    args = parser.parse_args()

    # Create database tables if they don't exist
    SQLModel.metadata.create_all(engine)

    # Read input data
    logger.info(f"Reading input data from {args.input}")
    df = pd.read_parquet(args.input)
    df["abstract"] = df["abstract"].fillna("").astype(str)

    # Start timing the entire process
    total_start_time = time.time()

    # Process abstracts using multiprocessing
    results, stats = process_in_parallel(df, args.output)

    # Calculate total processing time
    total_time = time.time() - total_start_time

    # Calculate and print summary statistics
    logger.info("\nProcessing Summary:")
    logger.info(f"Total rows in input: {stats['total_rows']}")
    logger.info(f"Successfully processed: {stats['successful']}")
    logger.info(f"Failed processing: {stats['failed']}")
    logger.info(f"Skipped (already processed): {stats['skipped']}")
    logger.info(f"Errors encountered: {stats['errors']}")
    logger.info(f"Total tokens used: {stats['total_tokens']}")
    logger.info(f"Total processing time: {total_time:.2f} seconds")
    logger.info(f"Average tokens per request: {stats['total_tokens'] / stats['successful'] if stats['successful'] else 0:.2f}")
    logger.info(f"Average time per request: {total_time / stats['successful'] if stats['successful'] else 0:.2f} seconds")
    logger.info("Processing complete!")

if __name__ == "__main__":
    main()
