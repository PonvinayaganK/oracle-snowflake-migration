# src/api/schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional

class MigrationRequest(BaseModel):
    """Schema for a migration request."""
    oracle_code: str = Field(..., description="The Oracle PL/SQL procedure or SQL view DDL to migrate.")
    object_type: str = Field(..., description="The type of the database object: 'Procedure' or 'View'.", pattern="^(Procedure|View)$")
    guidelines: Optional[str] = Field(None, description="Additional migration guidelines or best practices.")
    llm_model: str = Field("gpt-4o", description="The LLM model to use for migration (e.g., 'gpt-4o', 'ollama-mistral').")
    optimization_cycles: int = Field(3, ge=1, le=10, description="Maximum number of optimization cycles for the LLM.")

class MigrationResponse(BaseModel):
    """Schema for a migration response."""
    status: str = Field(..., description="Overall status of the migration (e.g., 'success', 'failed', 'partial_success').")
    migrated_code: Optional[str] = Field(None, description="The generated Snowflake-compatible DDL.")
    reflection_notes: Optional[str] = Field(None, description="LLM's internal reflection and optimization notes.")
    errors: List[str] = Field((), description="List of any errors or warnings encountered during migration.")
    object_type: str = Field(..., description="The type of the migrated database object.")
    request_id: str = Field(..., description="Unique identifier for this migration request.")