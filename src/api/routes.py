# src/api/routes.py
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
import logging
import uuid
from typing import List

from src.api.schemas import MigrationRequest, MigrationResponse
from src.core.agent_workflow import compile_migration_graph, AgentState
from src.llm.llm_factory import LLMFactory # Use LLMFactory
from src.utils.exceptions import MigrationError, InvalidInputError, ConfigurationError, LLMError, RAGError
from src.config import SUPPORTED_LLM_MODELS, MAX_OPTIMIZATION_CYCLES, SUPPORTED_OBJECT_TYPES
from langchain_core.documents import Document # Import Document for sample_snowflake_code

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/migrate", response_model=MigrationResponse)
async def migrate_object(request: Request, migration_req: MigrationRequest):
    """
    Endpoint to trigger the migration of an Oracle database object (Procedure or View)
    to Snowflake compatible DDL.
    """
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] Migration request received for object type: {migration_req.object_type}")
    logger.debug(f"[{request_id}] Request details: {migration_req.model_dump_json(indent=2)}")

    # Input validation
    if not migration_req.oracle_code.strip():
        logger.error(f"[{request_id}] InvalidInputError: Oracle DDL is empty.")
        raise HTTPException(status_code=400, detail="Oracle DDL cannot be empty.")
    if migration_req.object_type not in SUPPORTED_OBJECT_TYPES:
        logger.error(f"[{request_id}] InvalidInputError: Unsupported object type: {migration_req.object_type}")
        raise HTTPException(status_code=400, detail=f"Unsupported object type. Must be one of: {SUPPORTED_OBJECT_TYPES}")
    if migration_req.llm_model not in SUPPORTED_LLM_MODELS:
        logger.error(f"[{request_id}] InvalidInputError: Unsupported LLM model: {migration_req.llm_model}")
        raise HTTPException(status_code=400, detail=f"Unsupported LLM model. Must be one of: {SUPPORTED_LLM_MODELS}")

    llm_factory = LLMFactory(migration_req.llm_model) # Use LLMFactory
    try:
        llm_instance = llm_factory.get_llm()
        # Compile the graph for the specific object type
        compiled_graph = compile_migration_graph(llm_instance, migration_req.object_type)

        initial_state: AgentState = {
            "oracle_object_code": migration_req.oracle_code,
            "snowflake_guidelines": migration_req.guidelines if migration_req.guidelines else "",
            "sample_snowflake_code": [], # This will be populated by RAG from vector store
            "generated_snowflake_code": "",
            "reflection": "",
            "errors": [],
            "current_step": "Start",
            "optimization_cycle_count": 0,
            "max_optimization_cycles": migration_req.optimization_cycles if migration_req.optimization_cycles > 0 else MAX_OPTIMIZATION_CYCLES,
            "object_type": migration_req.object_type,
            "decomposed_oracle_view_parts": {},
            "translated_snowflake_view_parts": {},
            "is_decomposed": False
        }

        # Run the LangGraph workflow
        final_state: AgentState = {}
        for s in compiled_graph.stream(initial_state, config={"llm": llm_instance}):
            if "__end__" not in s:
                current_state_dict = list(s.values())[0]
                final_state = current_state_dict
            else:
                final_state = s["__end__"]

        response_status = "success" if not final_state.get("errors") else "partial_success" if final_state.get("generated_snowflake_code") else "failed"

        logger.info(f"[{request_id}] Migration finished with status: {response_status}")
        return MigrationResponse(
            request_id=request_id,
            status=response_status,
            migrated_code=final_state.get("generated_snowflake_code"),
            reflection_notes=final_state.get("reflection"),
            errors=final_state.get("errors"),
            object_type=final_state.get("object_type")
        )

    except (InvalidInputError, ConfigurationError) as e:
        logger.error(f"[{request_id}] Configuration/Input Error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except (LLMError, RAGError, MigrationError) as e: # APIError removed
        logger.error(f"[{request_id}] Migration specific error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Migration process failed: {e}")
    except Exception as e:
        logger.critical(f"[{request_id}] An unhandled error occurred: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "message": "Oracle to Snowflake Migrator API is running."}