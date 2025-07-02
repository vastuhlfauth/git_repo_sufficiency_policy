import uuid
from typing import Optional
from sqlmodel import SQLModel, Field, Session
from sqlalchemy import Column, String, JSON
from datetime import datetime
import json
from pathlib import Path

class ExtractedPolicies(SQLModel, table=True):
    """Model for storing extracted policies from abstracts."""
    __tablename__ = "policies_abstracts_all"
    openalex_id: Optional[str] = Field(index=True, primary_key=True)
    geographic: Optional[str] = Field(default=None)
    doi: Optional[str] = Field(default=None, index=True)
    abstract: str = Field(sa_column=Column(String))
    extracted_data: dict = Field(sa_column=Column(JSON))

def persist_policies(session: Session, extracted_policy: dict, abstract: str, openalex_id: str, doi: str) -> str:
    """
    Persist the extracted policy data in the database.
    
    Args:
        session: SQLModel database session
        extracted_policy: Dictionary containing the extracted policy data
        abstract: The original abstract text
        
    Returns:
        str: The ID of the newly created record
        
    Raises:
        Exception: If there's an error during database operations
    """
    
    try:
        # print(extracted_policy)
        # Create a new ExtractedPolicies instance
        extracted_policy_data = extracted_policy.get('ITEM') or extracted_policy.get('ITEMs', None)
        db_article = ExtractedPolicies(
            geographic=extracted_policy.get('GEOGRAPHIC', ""),
            openalex_id=openalex_id,
            doi=doi,
            abstract=abstract,
            extracted_data=extracted_policy_data
        )

        # Add to session and commit
        session.add(db_article)
        session.commit()
        session.refresh(db_article)

        return db_article.openalex_id
    except Exception as e:
        session.rollback()
        raise Exception(f"Failed to persist policy: {str(e)}")
    
    
def is_in_base(session: Session, openalex_id: str) -> bool:
    return session.query(ExtractedPolicies).filter(ExtractedPolicies.openalex_id == openalex_id).first() is not None

def bulk_persist_policies(session: Session, json_file_path: str) -> tuple[int, int]:
    """
    Bulk persist policies from a JSON file into the database.
    
    Args:
        session: SQLModel database session
        json_file_path: Path to the JSON file containing policy data
        
    Returns:
        tuple[int, int]: A tuple containing (success_count, error_count)
        
    Raises:
        FileNotFoundError: If the JSON file doesn't exist
        json.JSONDecodeError: If the JSON file is malformed
    """
    success_count = 0
    error_count = 0
    
    # Check if file exists
    if not Path(json_file_path).exists():
        raise FileNotFoundError(f"JSON file not found: {json_file_path}")
    
    # Read and parse JSON file
    with open(json_file_path, 'r') as f:
        policies = json.load(f)
    
    # Process each policy
    for policy in policies:
        try:
            # Extract required fields
            openalex_id = policy.get('openalex_id')
            doi = policy.get('doi')
            abstract = policy.get('abstract')
            
            if not all([openalex_id, abstract]):
                print(f"Skipping policy due to missing required fields: {openalex_id}")
                error_count += 1
                continue
                
            # Check if policy already exists
            if is_in_base(session, openalex_id):
                print(f"Policy already exists: {openalex_id}")
                continue
                
            # Persist the policy
            persist_policies(session, policy, abstract, openalex_id, doi)
            success_count += 1
            
        except Exception as e:
            print(f"Error processing policy {openalex_id}: {str(e)}")
            error_count += 1
            session.rollback()
    
    return success_count, error_count

def main():
    """
    CLI interface for bulk uploading policies.
    """
    import argparse
    from sqlmodel import create_engine, Session
    import os
    
    parser = argparse.ArgumentParser(description='Bulk upload policies from a JSON file to the database.')
    parser.add_argument('json_file', help='Path to the JSON file containing policies')
    parser.add_argument('--db-url', 
                       default=os.getenv('DATABASE_URL', "postgresql://u4axloluqibskgvdikuy:g2rXgpHSbztokCbFxSyR@bk8htvifqendwt1wlzat-postgresql.services.clever-cloud.com:7327/bk8htvifqendwt1wlzat"),
                       help='Database URL (default: sqlite:///policies.db or DATABASE_URL env var)')
    parser.add_argument('--verbose', '-v', 
                       action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Create database engine and session
    engine = create_engine(args.db_url)
    session = Session(engine)
    
    try:
        success_count, error_count = bulk_persist_policies(session, args.json_file)
        
        if args.verbose:
            print(f"\nUpload Summary:")
            print(f"Successfully uploaded: {success_count} policies")
            print(f"Failed to upload: {error_count} policies")
            print(f"Total processed: {success_count + error_count} policies")
        else:
            print(f"Uploaded {success_count} policies, {error_count} failed")
            
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON file - {str(e)}")
        exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)
    finally:
        session.close()

if __name__ == '__main__':
    main()