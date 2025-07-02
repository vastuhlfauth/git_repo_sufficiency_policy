"""
Pydantic models for abstract analysis output.
"""

from typing import Dict, List
from pydantic import BaseModel, Field


class Correlation(BaseModel):
    """Correlation between an item and a factor."""
    CORRELATION: str = Field(..., description="Description of how the item affects the factor")


class Factor(BaseModel):
    """Factor affected by an item."""
    FACTOR: Dict[str, Correlation] = Field(..., description="Dictionary of factors and their correlations")


class Item(BaseModel):
    """Item extracted from the abstract."""
    type: str = Field(..., description="The specific practice, choice, lifestyle, public policy, private action, property, feature, technological device, system, or service mentioned in the abstract")
    FACTOR: List[Factor] = Field(..., description="List of factors and their correlations")


class AbstractAnalysis(BaseModel):
    """Complete analysis of an abstract."""
    GEOGRAPHIC: str = Field(..., description="Geographical scope of the study")
    items: List[Item] = Field(..., description="List of items and their factors") 
