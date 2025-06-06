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
    name: Optional[str] = Field(None, description="Product name")
    material_type: Optional[str] = Field(None, description="Type of material")
    manufacturing_country: Optional[str] = Field(None, description="Country of manufacture")
    city: Optional[str] = Field(None, description="City of manufacture")
    declared_unit: Optional[str] = Field(None, description="Unit of measurement for emissions")
    manufacturing_emissions: Optional[float] = Field(None, description="Manufacturing emissions value")

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

class AiFormSchemerInput(BaseModel):
    grid_spacing_x: int = Field(..., description="Grid spacing in X direction")
    grid_spacing_y: int = Field(..., description="Grid spacing in Y direction")
    extents_x: int = Field(..., description="X extent of the building")
    extents_y: int = Field(..., description="Y extent of the building")
    no_of_floors: int = Field(..., description="Number of floors")

class AiFormSchemerOutput(BaseModel):
    steel_tonnage: float = Field(..., description="Total steel tonnage")
    column_size: int = Field(..., description="Size of columns")
    structural_depth: int = Field(..., description="Structural depth")
    concrete_tonnage: float = Field(..., description="Total concrete tonnage")
    trustworthy: bool = Field(..., description="Whether the results are trustworthy")
