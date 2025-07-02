#!/usr/bin/env python3

"""
Document Splitter Script

This script splits a large parquet file into 5 smaller files for parallel processing.
The files are saved in a data folder with sequential numbering.

Usage:
    python split_documents.py --input <input_parquet_path>

Required arguments:
    --input: Path to the input parquet file containing all documents
"""

import argparse
import os
import math
import pandas as pd
from pathlib import Path


def split_parquet(input_path: str, num_splits: int = 5):
    """
    Split a parquet file into multiple smaller files.
    
    Args:
        input_path: Path to the input parquet file
        num_splits: Number of files to split into (default: 5)
    """
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Read the parquet file
    print(f"Reading input file: {input_path}")
    df = pd.read_parquet(input_path)
    
    # Calculate the size of each split
    total_rows = len(df)
    rows_per_split = math.ceil(total_rows / num_splits)
    
    print(f"Total documents: {total_rows}")
    print(f"Documents per split: {rows_per_split}")
    
    # Split and save the files
    for i in range(num_splits):
        start_idx = i * rows_per_split
        end_idx = min((i + 1) * rows_per_split, total_rows)
        
        # Get the split
        split_df = df.iloc[start_idx:end_idx]
        
        # Create output filename
        output_path = data_dir / f"documents_{i+1}.parquet"
        
        # Save the split
        print(f"Saving split {i+1} to {output_path} ({len(split_df)} documents)")
        split_df.to_parquet(output_path, index=False)
    
    print("\nSplit complete!")
    print(f"Files saved in: {data_dir.absolute()}")


def main():
    parser = argparse.ArgumentParser(description='Split a parquet file into multiple smaller files.')
    parser.add_argument('--input', required=True, help='Path to input parquet file')
    parser.add_argument('--splits', type=int, default=5, help='Number of files to split into (default: 5)')
    
    args = parser.parse_args()
    
    split_parquet(args.input, args.splits)


if __name__ == "__main__":
    main() 