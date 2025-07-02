import pandas as pd
import numpy as np
import json
import os
import sys
import argparse
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import HDBSCAN, KMeans, DBSCAN, AgglomerativeClustering
from sklearn.manifold import TSNE
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import logging
from tqdm import tqdm
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Sector mapping and definitions
SECTOR_MAPPING = {
    'A': 'Agriculture, Forestry, fishing',
    'B': 'Land-use and urbanisation',
    'C': 'Trade',
    'D': 'Transportation',
    'E': 'Information and communication',
    'F': 'Waste management',
    'G': 'Tourism, Arts, entertainment and recreation',
    'H': 'Financial activities',
    'I': 'Scientific activities and technical activities',
    'J': 'Social affairs',
    'K': 'Public administration, administration and support service activities'
}

def safe_json_loads(x):
    """Safely load JSON string, handling potential errors."""
    if pd.isna(x) or not isinstance(x, str):
        return {}
    try:
        return json.loads(x)
    except json.JSONDecodeError:
        logging.warning(f"Failed to parse JSON: {x[:100]}...")
        return {}

def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    """
    Load the CSV data and preprocess it for clustering.
    """
    logging.info(f"Loading data from {file_path}")
    start_time = time.time()
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    logging.info(f"Loaded {len(df)} rows from CSV")
    
    # Convert string representations of FACTOR and SECTOR back to Python objects
    logging.info("Converting JSON strings to Python objects...")
    df['FACTOR'] = df['FACTOR'].apply(safe_json_loads)
    df['SECTOR'] = df['SECTOR'].apply(safe_json_loads)
    
    # Create a text representation of factors for clustering
    logging.info("Creating text representations for clustering...")
    def create_factor_text(factor_dict):
        try:
            if isinstance(factor_dict, str):
                return factor_dict
            if isinstance(factor_dict, list):
                return ' '.join(str(x) for x in factor_dict)
            if not isinstance(factor_dict, dict):
                logging.warning(f"Unexpected factor_dict type: {type(factor_dict)}, value: {factor_dict}")
                return ""
            
            # Handle both nested dict and direct CORRELATION key cases
            if 'CORRELATION' in factor_dict:
                return f"correlation {factor_dict['CORRELATION']}"
            
            return ' '.join([f"{k} {v.get('CORRELATION', '')}" for k, v in factor_dict.items()])
        except Exception as e:
            logging.error(f"Error processing factor_dict: {factor_dict}, Error: {str(e)}")
            import pdb; pdb.set_trace()
            return ""
    
    df['factor_text'] = df['FACTOR'].apply(create_factor_text)
    
    # Map sector codes to full names
    def map_sectors(sector_list):
        if not isinstance(sector_list, list):
            return ['Unknown']
        return [SECTOR_MAPPING.get(s, 'Unknown') for s in sector_list]
    
    df['sector_name'] = df['SECTOR'].apply(map_sectors)
    
    # Log some sample data for verification
    logging.info("\nSample data after preprocessing:")
    sample_row = df.iloc[0]
    logging.info(f"FACTOR: {sample_row['FACTOR']}")
    logging.info(f"factor_text: {sample_row['factor_text']}")
    logging.info(f"SECTOR: {sample_row['SECTOR']}")
    logging.info(f"sector_name: {sample_row['sector_name']}")
    
    elapsed_time = time.time() - start_time
    logging.info(f"Data preprocessing completed in {elapsed_time:.2f} seconds")
    
    return df

def embed_batch(batch, embedder):
    """Embed a batch of texts using Sentence BERT."""
    return embedder.encode(batch, show_progress_bar=False)

def parallel_embedding(corpus, embedder, batch_size=512):
    """Generate embeddings in parallel batches."""
    total_batches = (len(corpus) + batch_size - 1) // batch_size
    logging.info(f"Starting parallel embedding with {total_batches} batches")
    
    embeddings = Parallel(n_jobs=-1)(
        delayed(embed_batch)(batch, embedder)
        for batch in tqdm([corpus[i:i + batch_size] for i in range(0, len(corpus), batch_size)], 
                         desc="Generating embeddings")
    )
    return np.vstack(embeddings)

def save_embeddings(embeddings: np.ndarray, filepath: str) -> None:
    """Save embeddings to a numpy file."""
    np.save(filepath, embeddings)
    logging.info(f"Saved embeddings to {filepath}")

def load_embeddings(filepath: str) -> np.ndarray:
    """Load embeddings from a numpy file."""
    embeddings = np.load(filepath)
    logging.info(f"Loaded embeddings from {filepath}")
    return embeddings

def perform_tsne_clustering(embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform t-SNE dimensionality reduction and HDBSCAN clustering.
    """
    logging.info("Performing t-SNE dimensionality reduction...")
    tsne = TSNE(
        n_components=2,
        perplexity=30.0,
        max_iter=2000,
        random_state=42
    )
    reduced_embeddings = tsne.fit_transform(embeddings)
    
    # Apply HDBSCAN clustering on t-SNE reduced dimensions
    logging.info("Performing HDBSCAN clustering...")
    hdbscan = HDBSCAN(
        min_cluster_size=40,        
        min_samples=10,             
        metric='euclidean',
        cluster_selection_epsilon=0.1,
    )
    cluster_assignment = hdbscan.fit_predict(reduced_embeddings)
    
    # Log cluster statistics
    n_clusters = len(set(cluster_assignment)) - (1 if -1 in cluster_assignment else 0)
    noise_points = np.sum(cluster_assignment == -1)
    total_points = len(cluster_assignment)
    logging.info(f"Found {n_clusters} clusters")
    logging.info(f"Noise points: {noise_points} ({noise_points/total_points*100:.1f}% of total)")
    
    return reduced_embeddings, cluster_assignment

def perform_clustering(df: pd.DataFrame, input_file: str, method: str = 'hdbscan') -> Tuple[object, np.ndarray, np.ndarray]:
    """
    Perform clustering using either HDBSCAN or t-SNE with K-means.
    """
    start_time = time.time()
    logging.info("Initializing Sentence BERT model...")
    
    # Initialize Sentence BERT model
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Prepare corpus
    preprocessed_corpus = df['ITEM'] # + ', ' + df['factor_text']
    preprocessed_corpus = preprocessed_corpus.reset_index(drop=True)
    corpus_list = preprocessed_corpus.tolist()
    logging.info(f"Prepared corpus with {len(corpus_list)} items")
    
    # Check if embeddings file exists in embeddings directory
    embeddings_file = os.path.join('embeddings', f'corpus_embeddings_{os.path.splitext(os.path.basename(input_file))[0]}_items_only.npy')
    if os.path.exists(embeddings_file):
        logging.info("Found existing embeddings file, loading...")
        corpus_embeddings = load_embeddings(embeddings_file)
    else:
        # Generate embeddings
        batch_size = 512
        logging.info("Generating embeddings...")
        corpus_embeddings = parallel_embedding(corpus_list, embedder, batch_size=batch_size)
        # Save embeddings for future use
        save_embeddings(corpus_embeddings, embeddings_file)
    
    if method == 'hdbscan':
        # Apply HDBSCAN clustering
        logging.info("Performing HDBSCAN clustering...")
        hdbscan_model = HDBSCAN(
            min_cluster_size=40,
            min_samples=10,
            metric='euclidean',
            cluster_selection_epsilon=0.1,
            cluster_selection_method='eom'
        )
        cluster_assignment = hdbscan_model.fit_predict(corpus_embeddings)
        model = hdbscan_model
    else:  # tsne
        # Apply t-SNE clustering
        logging.info("Performing t-SNE clustering...")
        reduced_embeddings, cluster_assignment = perform_tsne_clustering(corpus_embeddings)
        model = reduced_embeddings
    
    elapsed_time = time.time() - start_time
    logging.info(f"Clustering completed in {elapsed_time:.2f} seconds")
    logging.info(f"Found {len(set(cluster_assignment))} clusters")
    
    return model, cluster_assignment, corpus_embeddings

def analyze_clusters(df: pd.DataFrame, cluster_assignment: np.ndarray, sector_letter: str, method: str = 'hdbscan', model: object = None) -> None:
    """
    Analyze and visualize the clustering results.
    """
    logging.info("Analyzing and visualizing clusters...")
    
    if method == 'hdbscan':
        # Create a figure with multiple subplots
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Cluster sizes (excluding noise points)
        cluster_sizes = pd.Series(cluster_assignment).value_counts().sort_index()
        cluster_sizes = cluster_sizes[cluster_sizes.index != -1]  # Remove noise points
        sns.barplot(x=cluster_sizes.index, y=cluster_sizes.values, ax=axes[0])
        axes[0].set_title('Cluster Sizes')
        axes[0].set_xlabel('Cluster')
        axes[0].set_ylabel('Number of Items')
        
        # Plot 2: Sector distribution in clusters
        df['cluster'] = cluster_assignment
        sector_cluster = pd.crosstab(
            df[df['cluster'] != -1]['cluster'],
            df[df['cluster'] != -1]['sector_name'].apply(lambda x: x[0] if x else 'Unknown')
        )
        sector_cluster.plot(kind='bar', stacked=True, ax=axes[1])
        axes[1].set_title('Sector Distribution in Clusters')
        axes[1].set_xlabel('Cluster')
        axes[1].set_ylabel('Count')
        plt.xticks(rotation=45)
    else:  # tsne
        # Create scatter plot with cluster colors
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(model[:, 0], model[:, 1], 
                            c=cluster_assignment, cmap='tab20', alpha=0.6)
        plt.colorbar(scatter, label='Cluster')
        plt.title('t-SNE Clustering Results')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
    
    plt.tight_layout()
    plt.savefig(f'cluster_analysis_sector_{sector_letter}_{method}_items_only.png')
    plt.close()
    logging.info(f"Visualizations saved to cluster_analysis_sector_{sector_letter}_{method}_items_only.png")

def main():
    parser = argparse.ArgumentParser(description='Perform clustering analysis on CSV data')
    parser.add_argument('input_file', help='Input CSV file path')
    parser.add_argument('--method', choices=['hdbscan', 'tsne'], default='tsne',
                      help='Clustering method to use (default: hdbscan)')
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: File {args.input_file} does not exist")
        sys.exit(1)
        
    # Create embeddings directory if it doesn't exist
    embeddings_dir = 'embeddings'
    os.makedirs(embeddings_dir, exist_ok=True)
    
    start_time = time.time()
    logging.info("Starting clustering analysis...")
    
    # Extract base filename without extension for output files
    base_name = os.path.splitext(os.path.basename(args.input_file))[0]
    # Extract sector letter from filename (assuming format like 'results_A.csv')
    sector_letter = base_name.split('_')[-1]
    
    # Load and preprocess data
    df = load_and_preprocess_data(args.input_file)
    
    # Perform clustering
    model, cluster_assignment, corpus_embeddings = perform_clustering(df, args.input_file, args.method)
    
    # Analyze results
    analyze_clusters(df, cluster_assignment, sector_letter, args.method, model)
    
    # Save cluster assignments
    df['cluster'] = cluster_assignment
    output_file = f'clusters_on_items_{sector_letter}.csv'
    df[['ITEM', 'cluster', 'sector_name', 'FACTOR', 'openalex_id']].to_csv(output_file, index=False)
    logging.info(f"Cluster results saved to {output_file}")
    
    elapsed_time = time.time() - start_time
    logging.info(f"Clustering analysis completed in {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main() 