from pydantic import BaseModel, Field, validator

class BoundingBox(BaseModel):
    """Standard bounding box representation in pixels."""
    x1: int = Field(..., description="Left coordinate in pixels", ge=0)
    y1: int = Field(..., description="Top coordinate in pixels", ge=0)
    x2: int = Field(..., description="Right coordinate in pixels", ge=0)
    y2: int = Field(..., description="Bottom coordinate in pixels", ge=0)

    @validator('x2')
    def validate_x2(cls, v, values):
        if 'x1' in values and v < values['x1']:
            raise ValueError("x2 must be greater than or equal to x1")
        return v

    @validator('y2')
    def validate_y2(cls, v, values):
        if 'y1' in values and v < values['y1']:
            raise ValueError("y2 must be greater than or equal to y1")
        return v

class Confidence(BaseModel):
    """Standard confidence score representation."""
    score: float = Field(..., ge=0, le=1, description="Confidence score between 0 and 1")
    method: str = Field(..., description="Method used to calculate confidence")

class PageInfo(BaseModel):
    """Standard page information."""
    page_number: int = Field(..., ge=1, description="Page number (1-indexed)")
    width: int = Field(..., gt=0, description="Page width in pixels")
    height: int = Field(..., gt=0, description="Page height in pixels")