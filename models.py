from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# Input/Output models for tools

class AddInput(BaseModel):
    a: int
    b: int

class AddOutput(BaseModel):
    result: int

class SqrtInput(BaseModel):
    a: int

class SqrtOutput(BaseModel):
    result: float

class StringsToIntsInput(BaseModel):
    string: str

class StringsToIntsOutput(BaseModel):
    ascii_values: List[int]

class ExpSumInput(BaseModel):
    int_list: List[int]

class ExpSumOutput(BaseModel):
    result: float

class Search2050ProductsInput(BaseModel):
    product_name: str = Field(..., description="The name of the product to search for.")

class ProductInfo(BaseModel):
    unique_product_uuid_v2: Optional[str] = None
    name: Optional[str] = None
    # Add any other relevant product fields you want to consistently return
    raw_data: Dict[str, Any] # To store the full product dictionary

class Search2050ProductsOutput(BaseModel):
    products: List[ProductInfo] = Field(default_factory=list, description="List of found products.")
    message: Optional[str] = None # For errors or status messages

class Get2050ProductDetailsInput(BaseModel):
    slug_id: str = Field(..., description="The slug ID of the product.")

class MaterialFacts(BaseModel):
    total_co2e_kg_mf: Optional[float] = None
    total_biogenic_co2e: Optional[float] = None
    # Add other material facts you need
    raw_data: Dict[str, Any] # To store the full material_facts dictionary


class Get2050ProductDetailsOutput(BaseModel):
    product_slug_id: Optional[str] = None
    material_facts: Optional[MaterialFacts] = None
    message: Optional[str] = None # For errors or status messages



# --- AiFormFinder Models ---
class AiFormFinderInput(BaseModel):
    extents_x_m: int = Field(..., description="Extents X in meters for the building.")
    extents_y_m: int = Field(..., description="Extents Y in meters for the building.")
    grid_spacing_x_m: int = Field(..., description="Grid Spacing X in meters.")
    grid_spacing_y_m: int = Field(..., description="Grid Spacing Y in meters.")
    no_of_floors: int = Field(..., description="Number of floors in the building.")

class AiFormFinderOutput(BaseModel):
    steel_tonnage_tons_per_m2: Optional[float] = None
    column_size_mm: Optional[int] = None
    structural_depth_mm: Optional[int] = None
    concrete_tonnage_tons_per_m2: Optional[float] = None
    trustworthiness: Optional[bool] = None
    message: Optional[str] = None # For errors or status messages