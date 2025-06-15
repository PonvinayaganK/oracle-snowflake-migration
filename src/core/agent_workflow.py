# src/core/agent_workflow.py
import logging
import json # Import for handling JSON response from LLM for decomposition
from typing import TypedDict, List, Dict
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END, START
from langchain_core.runnables import RunnableConfig # Import RunnableConfig

# Import prompts from the updated prompts file
from src.core.prompts import (
    CODE_GENERATION_PROMPT_TEMPLATE, OPTIMIZATION_REFLECTION_PROMPT_TEMPLATE,
    INITIAL_ANALYSIS_PROMPT, RAG_QUERY_GENERATION_PROMPT, # RAG_QUERY_GENERATION_PROMPT is still used for LLM's internal thinking
    VIEW_DECOMPOSITION_PROMPT_TEMPLATE, VIEW_PART_TRANSLATION_PROMPT_TEMPLATE,
    VIEW_ASSEMBLY_PROMPT_TEMPLATE
)
# No longer importing HybridRetriever as ChromaDB is removed
# from src.rag.retriever import HybridRetriever
# No longer importing snowflake_connector for RAG here, as it's not a direct part of this RAG
# from src.data_processing.snowflake_connector import get_rag_context_from_snowflake_schema

from src.utils.exceptions import LLMError, RAGError
from src.config import VIEW_DECOMPOSITION_THRESHOLD # Import the threshold

logger = logging.getLogger(__name__)

# Define the Agent's State
class AgentState(TypedDict):
    oracle_object_code: str
    snowflake_guidelines: str
    sample_snowflake_code: str # Reverted to a single string containing all samples
    generated_snowflake_code: str
    reflection: str
    errors: List[str]
    current_step: str # To track progress for UI
    optimization_cycle_count: int
    max_optimization_cycles: int
    object_type: str # "Procedure" or "View"
    # NEW for view decomposition
    decomposed_oracle_view_parts: Dict[str, str] # Stores parts like 'main_query', 'cte_1', 'where_clause'
    translated_snowflake_view_parts: Dict[str, str] # Stores translated parts
    is_decomposed: bool # Flag to indicate if decomposition was performed

# Define Agent Tools (shared tools and helpers)
def validate_snowflake_syntax(snowflake_code: str, object_type: str) -> bool: # Added object_type param
    """
    Placeholder for actual Snowflake syntax validation.
    Validation rules might slightly differ based on object_type (e.g., procedures have DECLARE/BEGIN/END).
    """
    logger.info(f"Performing simulated Snowflake syntax validation for {object_type}.")

    is_valid = True
    errors = []

    # Basic checks applicable to both procedures and views
    if not snowflake_code.strip():
        errors.append("Generated code is empty.")
        is_valid = False

    # Specific checks for Procedures
    if object_type == "Procedure":
        if "CREATE OR REPLACE PROCEDURE" not in snowflake_code and "AS $$" not in snowflake_code and "AS BEGIN" not in snowflake_code:
            errors.append("Procedure missing expected 'CREATE OR REPLACE PROCEDURE' or 'AS $$'/'AS BEGIN' structure.")
            is_valid = False
        if "BEGIN" not in snowflake_code or "END;" not in snowflake_code:
            errors.append("Procedure missing BEGIN/END block.")
            is_valid = False

    # Specific checks for Views
    elif object_type == "View":
        if "CREATE OR REPLACE VIEW" not in snowflake_code or "AS SELECT" not in snowflake_code:
            errors.append("View missing expected 'CREATE OR REPLACE VIEW' or 'AS SELECT' structure.")
            is_valid = False
        if "(+)" in snowflake_code: # Check for Oracle outer join syntax
            errors.append("Oracle (+) outer join syntax detected in view. Please ensure it's converted to ANSI JOINs.")
            is_valid = False
        if "ROWNUM" in snowflake_code.upper():
            errors.append("Oracle ROWNUM pseudocolumn detected in view. Should be converted to ROW_NUMBER().")
            is_valid = False


    # Common SQL syntax checks (very basic)
    if "SELECT ;" in snowflake_code: # A common typo LLMs might make
        errors.append("Detected 'SELECT ;' which is usually a syntax error.")
        is_valid = False

    if errors:
        logger.warning(f"Simulated validation found errors for {object_type}: {', '.join(errors)}")
    else:
        logger.info(f"Simulated validation passed for {object_type}.")

    return is_valid

# --- Graph Nodes ---
def initial_analysis_node(state: AgentState, config: RunnableConfig) -> AgentState:
    logger.info(f"Executing initial_analysis_node for {state['object_type']}")
    logger.debug(f"initial_analysis_node received config: {config}")
    try:
        llm = config['configurable']['llm']
        prompt = INITIAL_ANALYSIS_PROMPT.invoke({
            "oracle_object_code": state['oracle_object_code'],
            "object_type": state['object_type']
        })
        response = llm.invoke(prompt)
        state["reflection"] = response.content # Store initial analysis in reflection
        state["current_step"] = f"Initial Analysis of {state['object_type']} Complete"
        logger.info(f"Initial Analysis: {response.content[:200]}...")
        return state
    except KeyError as e:
        logger.error(f"KeyError in initial_analysis_node: Missing config key {e}", exc_info=True)
        raise LLMError(f"LLM configuration missing: Key {e} not found in node config. Ensure LLM is passed correctly to graph.stream().")
    except Exception as e:
        logger.error(f"Error in initial_analysis_node: {e}", exc_info=True)
        raise LLMError(f"LLM failed during initial analysis: {e}")

def rag_retrieval_node(state: AgentState, config: RunnableConfig) -> AgentState:
    logger.info(f"Executing rag_retrieval_node for {state['object_type']}")
    logger.debug(f"rag_retrieval_node received config: {config}")
    try:
        llm = config['configurable']['llm']
        # For simplicity without ChromaDB, RAG retrieval means ensuring
        # sample_snowflake_code and guidelines are present in the state.
        # RAG_QUERY_GENERATION_PROMPT is used here just to let LLM "think" about relevant terms,
        # but it doesn't dynamically fetch from an external store.
        query_prompt = RAG_QUERY_GENERATION_PROMPT.invoke({
            "oracle_analysis": state['reflection'],
            "object_type": state['object_type']
        })
        llm_rag_thinking = llm.invoke(query_prompt).content
        logger.info(f"LLM thinking for RAG (not fetching from DB): {llm_rag_thinking[:100]}...")

        if not state.get("sample_snowflake_code") and not state.get("snowflake_guidelines"):
            logger.warning("No sample Snowflake code or guidelines provided. RAG context will be limited.")
            state["errors"].append("Warning: No RAG context (sample code or guidelines) provided. LLM performance might be reduced.")

        state["current_step"] = f"RAG Context Prepared for {state['object_type']}."
        logger.info("RAG Context (user-provided samples and guidelines) prepared.")
        return state
    except KeyError as e:
        logger.error(f"KeyError in rag_retrieval_node: Missing config key {e}", exc_info=True)
        raise LLMError(f"LLM configuration missing: Key {e} not found in node config.")
    except Exception as e:
        logger.error(f"Error in rag_retrieval_node: {e}", exc_info=True)
        raise LLMError(f"Failed during RAG context preparation: {e}")

# --- NEW NODE: View Decomposition ---
def view_decomposition_node(state: AgentState, config: RunnableConfig) -> AgentState:
    logger.info(f"Executing view_decomposition_node for {state['object_type']}")
    logger.debug(f"view_decomposition_node received config: {config}")

    oracle_view_lines = state['oracle_object_code'].count('\n') + 1
    # Check if it's a view and exceeds the threshold
    if state['object_type'] == "View" and oracle_view_lines > VIEW_DECOMPOSITION_THRESHOLD:
        logger.info(f"View is large ({oracle_view_lines} lines). Attempting decomposition.")
        try:
            llm = config['configurable']['llm']
            prompt = VIEW_DECOMPOSITION_PROMPT_TEMPLATE.invoke({
                "oracle_view_code": state['oracle_object_code'],
                "snowflake_guidelines": state['snowflake_guidelines'],
                "sample_snowflake_code": state['sample_snowflake_code'] # Direct string from state
            })
            response = llm.invoke(prompt)
            # The LLM should return a JSON string
            decomposed_parts = json.loads(response.content)
            state["decomposed_oracle_view_parts"] = decomposed_parts
            state["is_decomposed"] = True
            state["translated_snowflake_view_parts"] = {} # Initialize for translation
            state["current_step"] = "View Decomposition Complete"
            logger.info("View decomposition successful.")
        except json.JSONDecodeError as e:
            logger.error(f"LLM returned invalid JSON for decomposition: {response.content[:500]}... Error: {e}", exc_info=True)
            state["errors"].append(f"View decomposition failed (invalid JSON from LLM): {e}. Proceeding without decomposition.")
            state["is_decomposed"] = False # Fallback to non-decomposed
        except Exception as e:
            logger.error(f"Error during view decomposition: {e}", exc_info=True)
            state["errors"].append(f"View decomposition failed: {e}. Proceeding without decomposition.")
            state["is_decomposed"] = False # Fallback to non-decomposed
    else:
        state["is_decomposed"] = False
        logger.info(f"View is small ({oracle_view_lines} lines) or not a view. Skipping decomposition.")

    return state

def code_generation_node(state: AgentState, config: RunnableConfig) -> AgentState:
    logger.info(f"Executing code_generation_node for {state['object_type']}")
    logger.debug(f"code_generation_node received config: {config}")
    try:
        llm = config['configurable']['llm']

        if state['is_decomposed'] and state['object_type'] == "View":
            logger.info("Translating decomposed view parts.")
            for part_name, oracle_sql_part in state['decomposed_oracle_view_parts'].items():
                # Skip if part is empty or already translated
                if part_name in state['translated_snowflake_view_parts'] and state['translated_snowflake_view_parts'][part_name].strip():
                    logger.info(f"Part '{part_name}' already translated. Skipping.")
                    continue
                if not oracle_sql_part.strip():
                    logger.info(f"Part '{part_name}' is empty. Skipping translation.")
                    state['translated_snowflake_view_parts'][part_name] = ""
                    continue

                part_prompt = VIEW_PART_TRANSLATION_PROMPT_TEMPLATE.invoke({
                    "oracle_view_part_name": part_name,
                    "oracle_view_part_code": oracle_sql_part,
                    "full_oracle_view_code": state['oracle_object_code'], # Provide full view for context
                    "snowflake_guidelines": state['snowflake_guidelines'],
                    "sample_snowflake_code": state['sample_snowflake_code'] # Direct string from state
                })
                translated_part = llm.invoke(part_prompt).content
                state['translated_snowflake_view_parts'][part_name] = translated_part
                logger.info(f"Translated view part: {part_name}")
            state["current_step"] = "Translated Decomposed View Parts"
            # Actual generated_snowflake_code will be assembled in view_assembly_node
            state["generated_snowflake_code"] = "Decomposed parts translated. Awaiting assembly."
        else:
            logger.info("Generating code for full object or non-view object.")
            prompt = CODE_GENERATION_PROMPT_TEMPLATE.invoke({
                "oracle_object_code": state['oracle_object_code'],
                "snowflake_guidelines": state['snowflake_guidelines'],
                "sample_snowflake_code": state['sample_snowflake_code'], # Direct string from state
                "object_type": state['object_type']
            })
            response = llm.invoke(prompt)
            state["generated_snowflake_code"] = response.content
            state["current_step"] = f"Code Generation for {state['object_type']} Complete"
            logger.info(f"Initial Snowflake {state['object_type']} code generated.")
        return state
    except KeyError as e:
        logger.error(f"KeyError in code_generation_node: Missing config key {e}", exc_info=True)
        raise LLMError(f"LLM configuration missing: Key {e} not found in node config.")
    except Exception as e:
        logger.error(f"Error in code_generation_node: {e}", exc_info=True)
        raise LLMError(f"LLM failed during code generation: {e}")

# --- NEW NODE: View Assembly ---
def view_assembly_node(state: AgentState, config: RunnableConfig) -> AgentState:
    logger.info(f"Executing view_assembly_node for {state['object_type']}")
    logger.debug(f"view_assembly_node received config: {config}")

    if state['is_decomposed'] and state['object_type'] == "View":
        logger.info("Assembling translated view parts.")
        try:
            llm = config['configurable']['llm']
            assembly_prompt = VIEW_ASSEMBLY_PROMPT_TEMPLATE.invoke({
                "original_oracle_view_code": state['oracle_object_code'],
                "translated_snowflake_view_parts": json.dumps(state['translated_snowflake_view_parts'], indent=2),
                "snowflake_guidelines": state['snowflake_guidelines']
            })
            assembled_code = llm.invoke(assembly_prompt).content
            state["generated_snowflake_code"] = assembled_code
            state["current_step"] = "View Assembly Complete"
            logger.info("View assembly successful.")
        except Exception as e:
            logger.error(f"Error during view assembly: {e}", exc_info=True)
            state["errors"].append(f"View assembly failed: {e}. Generated code might be incomplete.")
            state["generated_snowflake_code"] = "Error during assembly. Check logs for partial translations."
    else:
        logger.info("Assembly node skipped (not a decomposed view).")

    return state

def optimization_and_reflection_node(state: AgentState, config: RunnableConfig) -> AgentState:
    logger.info(f"Executing optimization_and_reflection_node for {state['object_type']}")
    logger.debug(f"optimization_and_reflection_node received config: {config}")

    # Increment optimization cycle count
    state["optimization_cycle_count"] += 1
    logger.info(f"Starting optimization cycle {state['optimization_cycle_count']} / {state['max_optimization_cycles']}")

    try:
        llm = config['configurable']['llm']
        reflection_prompt = OPTIMIZATION_REFLECTION_PROMPT_TEMPLATE.invoke({
            "original_oracle_object_code": state['oracle_object_code'],
            "generated_snowflake_code": state['generated_snowflake_code'],
            "snowflake_guidelines": state['snowflake_guidelines'],
            "sample_snowflake_code": state['sample_snowflake_code'], # Direct string from state
            "object_type": state['object_type']
        })
        response = llm.invoke(reflection_prompt)
        state["reflection"] = response.content
        state["current_step"] = f"Optimization Cycle {state['optimization_cycle_count']} for {state['object_type']} Complete"
        logger.info(f"Reflection: {response.content[:200]}...")

        if "No further optimization required." in response.content:
            state["errors"] = [] # Clear errors if optimized
            logger.info(f"LLM determined no further optimization needed after {state['optimization_cycle_count']} cycles.")
        else:
            if "Further optimization/correction needed based on reflection." not in state["errors"]:
                state["errors"].append("Further optimization/correction needed based on reflection.")
            logger.warning(f"LLM identified areas for further optimization after {state['optimization_cycle_count']} cycles.")

        return state
    except KeyError as e:
        logger.error(f"KeyError in optimization_and_reflection_node: Missing config key {e}", exc_info=True)
        raise LLMError(f"LLM configuration missing: Key {e} not found in node config.")
    except Exception as e:
        logger.error(f"Error in optimization_and_reflection_node: {e}", exc_info=True)
        raise LLMError(f"LLM failed during optimization and reflection: {e}")

def validation_node(state: AgentState, config: RunnableConfig) -> AgentState:
    logger.info(f"Executing validation_node for {state['object_type']}")
    # Call validate_snowflake_syntax with object_type
    is_valid = validate_snowflake_syntax(state["generated_snowflake_code"], state["object_type"])
    if is_valid:
        state["errors"] = [err for err in state["errors"] if not err.startswith("Snowflake code has syntax errors.")]
        logger.info(f"Snowflake {state['object_type']} code passed syntax validation.")
    else:
        if "Snowflake code has syntax errors. Please regenerate." not in state["errors"]:
            state["errors"].append("Snowflake code has syntax errors. Please regenerate.")
        logger.warning(f"Snowflake {state['object_type']} code failed syntax validation.")
    state["current_step"] = f"Validation of {state['object_type']} Complete"
    return state

# --- Conditional Logic for Graph Flow ---
def should_decompose_view(state: AgentState) -> str:
    # Check if it's a view and exceeds the threshold
    oracle_view_lines = state['oracle_object_code'].count('\n') + 1
    if state['object_type'] == "View" and oracle_view_lines > VIEW_DECOMPOSITION_THRESHOLD:
        logger.info(f"Object is a large view ({oracle_view_lines} lines). Branching to decomposition.")
        return "decompose"
    logger.info(f"Object is not a large view or is a procedure. Branching to direct generation.")
    return "no_decompose"

def should_assemble_view(state: AgentState) -> str:
    # Only assemble if decomposition was performed and it's a view
    if state['is_decomposed'] and state['object_type'] == "View":
        logger.info("Decomposed view, branching to assembly.")
        return "assemble"
    logger.info("Not a decomposed view, skipping assembly.")
    return "no_assemble"

def should_continue_optimization(state: AgentState) -> str:
    needs_optimization = "Further optimization/correction needed based on reflection." in state.get("errors", [])
    has_syntax_errors = "Snowflake code has syntax errors. Please regenerate." in state.get("errors", [])

    if (needs_optimization or has_syntax_errors) and state["optimization_cycle_count"] < state["max_optimization_cycles"]:
        logger.info(f"Continuing optimization: Cycle {state['optimization_cycle_count']}/{state['max_optimization_cycles']}. Errors/needs: {state.get('errors')}")
        return "continue"
    else:
        logger.info(f"Stopping optimization. Max cycles reached or no further optimization needed. Errors/needs: {state.get('errors')}")
        return "stop"

# Define the graph compilation function
def compile_migration_graph(llm_instance, object_type: str):
    logger.info(f"Compiling LangGraph migration workflow for {object_type}.")
    graph_builder = StateGraph(AgentState)

    # Add all nodes
    graph_builder.add_node("initial_analysis", initial_analysis_node)
    graph_builder.add_node("rag_retrieval", rag_retrieval_node)
    graph_builder.add_node("view_decomposition", view_decomposition_node) # New node
    graph_builder.add_node("code_generation", code_generation_node)
    graph_builder.add_node("view_assembly", view_assembly_node) # New node
    graph_builder.add_node("optimization_and_reflection", optimization_and_reflection_node)
    graph_builder.add_node("validation", validation_node)

    # Define entry point and initial flow
    graph_builder.set_entry_point("initial_analysis")
    graph_builder.add_edge("initial_analysis", "rag_retrieval")

    # Branch after RAG based on object type and size
    graph_builder.add_conditional_edges(
        "rag_retrieval",
        should_decompose_view,
        {
            "decompose": "view_decomposition",
            "no_decompose": "code_generation",
        }
    )

    # All paths eventually lead to code generation (either direct or after decomposition)
    graph_builder.add_edge("view_decomposition", "code_generation")


    # Branch after code generation for assembly if decomposed, else go to optimization
    graph_builder.add_conditional_edges(
        "code_generation",
        should_assemble_view,
        {
            "assemble": "view_assembly",
            "no_assemble": "optimization_and_reflection",
        }
    )

    # After assembly, always go to optimization
    graph_builder.add_edge("view_assembly", "optimization_and_reflection")

    # Conditional loop for optimization and validation
    graph_builder.add_conditional_edges(
        "optimization_and_reflection",
        should_continue_optimization,
        {
            "continue": "code_generation", # Loop back to generation for refinement
            "stop": "validation", # If max cycles reached or no more errors, validate final
        }
    )

    # Validation leads to END or loops back if syntax errors persist (and cycles allow)
    graph_builder.add_conditional_edges(
        "validation",
        lambda state: (
            "code_generation"
            if "Snowflake code has syntax errors. Please regenerate." in state.get("errors", []) and
               state["optimization_cycle_count"] < state["max_optimization_cycles"]
            else END
        ),
        {
            "code_generation": "code_generation",
            END: END,
        }
    )

    compiled_graph = graph_builder.compile()
    logger.info(f"LangGraph migration workflow compiled successfully for {object_type}.")
    return compiled_graph

