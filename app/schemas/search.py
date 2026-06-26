from typing import List, Dict, Literal, Optional
from pydantic import BaseModel, Field


class SearchInterpretedAs(BaseModel):
    normalized: Optional[str] = Field(None, description="The normalized form of the query")
    intent: Optional[str] = Field(None, description="The detected intent (e.g., treatment, device, etc.)")
    concepts: Optional[List[str]] = Field(None, description="Key concepts extracted from the search query")
    exclusions: Optional[List[str]] = Field(None, description="Concepts to exclude from search results")
    is_negation: Optional[bool] = Field(None, description="Whether the query was classified as a negation search")
    confidence: Optional[Literal["high", "medium", "low"]] = Field(None, description="LLM classification confidence level")


class SearchResultItem(BaseModel):
    id: Optional[str] = Field(None, description="The UUID of the treatment or device entity")
    name: Optional[str] = Field(None, description="The English name of the entity")
    swedish_name: Optional[str] = Field(None, description="The Swedish name of the entity")
    type: Optional[str] = Field(None, description="The type of entity ('treatment' or 'device')")
    modality: Optional[str] = Field(None, description="Treatment/device modality description")
    family: Optional[str] = Field(None, description="Functional family of the entity")
    concern: Optional[str] = Field(None, description="Client concerns this entity addresses")
    description: Optional[str] = Field(None, description="Truncated description snippet of the entity")
    score: Optional[float] = Field(None, description="Relevance score indicating match quality")


class SearchResponse(BaseModel):
    query: str = Field(..., description="The original search query string")
    interpreted_as: SearchInterpretedAs = Field(..., description="LLM classification and interpretation of user query")
    results: Dict[str, List[SearchResultItem]] = Field(..., description="Grouped matching entities")
