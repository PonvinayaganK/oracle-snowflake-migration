# src/ui/app.py
import streamlit as st
import os
import tempfile
import json
import logging
import io
from PIL import Image

# Set up project-level logging before importing other modules that use it
from src.utils.logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

from src.core.agent_workflow import compile_migration_graph
from src.data_processing.file_utils import save_uploaded_file, read_file_content
from src.llm_config.llm_manager import LLMManager  # Use LLMFactory for consistency
from src.utils.exceptions import MigrationError, InvalidInputError
# Import settings from src.config.settings
from src.config import (
    SUPPORTED_LLM_MODELS, DEFAULT_LLM_MODEL, PROXY_CONFIG, OLLAMA_BASE_URL,
    MAX_OPTIMIZATION_CYCLES, SUPPORTED_OBJECT_TYPES, DEFAULT_OBJECT_TYPE,
    VIEW_DECOMPOSITION_THRESHOLD
)

st.set_page_config(layout="wide", page_title="Oracle to Snowflake Migrator")


def main():
    st.title("❄️ Oracle to Snowflake Database Object Migrator (AI Powered)")

    st.markdown("""
        This tool uses an LLM-powered AI agent to migrate complex Oracle PL/SQL procedures and SQL Views
        to Snowflake-compatible objects, leveraging user-defined guidelines and
        sample Snowflake objects for enhanced accuracy and optimization.
    """)

    # --- Configuration Section ---
    st.sidebar.header("Configuration")

    # LLM Selection
    st.sidebar.subheader("LLM Model Selection")
    selected_llm_model = st.sidebar.selectbox(
        "Choose an LLM Model:",
        options=SUPPORTED_LLM_MODELS,
        index=SUPPORTED_LLM_MODELS.index(DEFAULT_LLM_MODEL) if DEFAULT_LLM_MODEL in SUPPORTED_LLM_MODELS else 0
    )

    # Optional: Ollama Base URL override
    if selected_llm_model.startswith("ollama-"):
        st.sidebar.subheader("Ollama Server Configuration")
        current_ollama_url = os.getenv("OLLAMA_BASE_URL", OLLAMA_BASE_URL)
        ollama_url_input = st.sidebar.text_input(
            "Ollama Base URL (e.g., http://localhost:11434)",
            value=current_ollama_url,
            key="ollama_base_url_input"
        )
        if ollama_url_input != current_ollama_url:
            os.environ['OLLAMA_BASE_URL'] = ollama_url_input
            st.sidebar.info(f"Ollama Base URL set to: {ollama_url_input}. Restart application if issues persist.")

    # Optimization Cycles Input
    st.sidebar.subheader("Optimization Settings")
    optimization_cycles = st.sidebar.number_input(
        "Max Optimization Cycles (LLM reflection loops):",
        min_value=1,
        max_value=10,
        value=MAX_OPTIMIZATION_CYCLES,
        step=1,
        help="Number of times the LLM will attempt to refine the generated code based on its self-reflection."
    )

    # View Decomposition Threshold
    st.sidebar.number_input(
        "View Decomposition Threshold (lines):",
        min_value=1,
        max_value=5000,  # Max value can be adjusted
        value=VIEW_DECOMPOSITION_THRESHOLD,
        step=10,
        key="view_decomposition_threshold",
        help="Oracle Views with more lines than this will be decomposed by LLM for migration."
    )

    # Proxy configuration (optional)
    if st.sidebar.checkbox("Use Proxy Settings"):
        st.sidebar.warning("Note: Proxy settings are applied globally by setting OS environment variables.")
        proxy_http = st.sidebar.text_input("HTTP Proxy (e.g., http://user:pass@host:port)",
                                           value=PROXY_CONFIG.get("http_proxy", ""))
        proxy_https = st.sidebar.text_input("HTTPS Proxy (e.g., https://user:pass@host:port)",
                                            value=PROXY_CONFIG.get("https_proxy", ""))
        if proxy_http:
            os.environ['HTTP_PROXY'] = proxy_http
            os.environ['http_proxy'] = proxy_http
        if proxy_https:
            os.environ['HTTPS_PROXY'] = proxy_https
            os.environ['https_proxy'] = proxy_https
        if proxy_http or proxy_https:
            st.sidebar.info("Proxy environment variables set. Restart if issues persist.")
        else:
            st.sidebar.info("Proxy environment variables cleared.")
            if 'HTTP_PROXY' in os.environ: del os.environ['HTTP_PROXY']
            if 'http_proxy' in os.environ: del os.environ['http_proxy']
            if 'HTTPS_PROXY' in os.environ: del os.environ['HTTPS_PROXY']
            if 'https_proxy' in os.environ: del os.environ['https_proxy']

    # --- Input Section ---
    st.header("1. Provide Input Files")

    # New: Object Type Selection
    selected_object_type = st.radio(
        "Select Object Type to Migrate:",
        options=SUPPORTED_OBJECT_TYPES,
        index=SUPPORTED_OBJECT_TYPES.index(DEFAULT_OBJECT_TYPE) if DEFAULT_OBJECT_TYPE in SUPPORTED_OBJECT_TYPES else 0,
        horizontal=True,
        key="object_type_selector"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"Oracle {selected_object_type} DDL (.sql)")
        uploaded_oracle_files = st.file_uploader(
            f"Upload one or more Oracle {selected_object_type} DDL files",
            type=["sql", "prc", "vw"],  # Added .vw extension for clarity, though .sql is common
            accept_multiple_files=True,
            key="oracle_uploader"
        )
        oracle_code_inputs = {}
        if uploaded_oracle_files:
            for file in uploaded_oracle_files:
                file_path = save_uploaded_file(file, "oracle_objects")  # Generic folder
                oracle_code_inputs[file.name] = read_file_content(file_path)
            st.success(f"Uploaded {len(uploaded_oracle_files)} Oracle {selected_object_type}(s).")
        else:
            st.info(
                f"No Oracle {selected_object_type} DDL uploaded. You can provide it manually below or use the default example.")
            default_oracle_code_value = ""
            if selected_object_type == "Procedure":
                default_oracle_code_value = """
                CREATE OR REPLACE PROCEDURE GET_EMPLOYEE_DETAILS (
                    p_employee_id IN NUMBER,
                    p_employee_name OUT VARCHAR2,
                    p_salary OUT NUMBER
                )
                IS
                    v_department_id NUMBER;
                    CURSOR c_emp IS
                        SELECT employee_name, salary, department_id
                        FROM employees
                        WHERE employee_id = p_employee_id;
                BEGIN
                    OPEN c_emp;
                    FETCH c_emp INTO p_employee_name, p_salary, v_department_id;
                    IF c_emp%NOTFOUND THEN
                        RAISE_APPLICATION_ERROR(-20001, 'Employee not found.');
                    END IF;
                    CLOSE c_emp;

                    UPDATE employees SET last_access_date = SYSDATE WHERE employee_id = p_employee_id;

                    EXCEPTION
                        WHEN OTHERS THEN
                            DBMS_OUTPUT.PUT_LINE('Error: ' || SQLERRM);
                            RAISE;
                END;
                """
            elif selected_object_type == "View":
                default_oracle_code_value = """
                CREATE OR REPLACE VIEW MY_SCHEMA.EMPLOYEE_SALARY_VW AS
                SELECT /*+ NO_MERGE */
                    e.employee_id,
                    e.employee_name,
                    e.salary,
                    d.department_name,
                    ROWNUM as rn
                FROM
                    employees e,
                    departments d
                WHERE
                    e.department_id = d.department_id(+)
                AND
                    e.salary > (SELECT AVG(salary) FROM employees)
                AND ROWNUM <= 100
                ORDER BY e.employee_id
                WITH READ ONLY;
                """

            manual_oracle_input = st.text_area(f"Or paste Oracle {selected_object_type} DDL here:",
                                               value=default_oracle_code_value, height=300, key="manual_oracle_input")
            if manual_oracle_input.strip():
                oracle_code_inputs["manual_input.sql"] = manual_oracle_input

    with col2:
        st.subheader("Migration Guidelines (`guidelines.txt`)")
        uploaded_guidelines = st.file_uploader(
            "Upload a text file with migration guidelines",
            type=["txt"],
            accept_multiple_files=False,
            key="guidelines_uploader"
        )
        guidelines_content = ""
        if uploaded_guidelines:
            guidelines_path = save_uploaded_file(uploaded_guidelines, "guidelines")
            guidelines_content = read_file_content(guidelines_path)
            st.success("Guidelines file uploaded.")
        else:
            st.info("No guidelines file uploaded. Using default example.")
            guidelines_content = read_file_content("data/guidelines.txt")
        st.text_area("Review/Edit Guidelines:", value=guidelines_content, height=200, key="guidelines_editor")

        st.subheader(f"Snowflake Sample {selected_object_type}s (RAG input)")
        sample_file_path = "data/sample_snowflake_procedures.txt" if selected_object_type == "Procedure" else "data/sample_snowflake_views.txt"

        uploaded_samples = st.file_uploader(
            f"Upload a text file with sample Snowflake {selected_object_type}s (RAG input)",
            type=["txt"],
            accept_multiple_files=False,  # Reverted to single file as no vector store
            key="samples_uploader"
        )
        sample_procedures_content = ""  # This will actually store sample DDL for both procs and views
        if uploaded_samples:
            samples_path = save_uploaded_file(uploaded_samples, "samples")
            sample_procedures_content = read_file_content(samples_path)
            st.success(f"Sample Snowflake {selected_object_type}s file uploaded.")
        else:
            st.info(f"No sample {selected_object_type}s uploaded. Using default example for {selected_object_type}s.")
            sample_procedures_content = read_file_content(sample_file_path)
        st.text_area(f"Review/Edit Sample {selected_object_type}s:", value=sample_procedures_content, height=200,
                     key="samples_editor")

    st.markdown("---")

    # --- Workflow Visualization Section ---
    # st.header("LangGraph Workflow Visualization")
    # st.info("This visualization shows the state transitions of the AI agent.")
    # try:
    #     # Use a dummy LLM and a default object type for visualization, as it's static
    #     dummy_llm = LLMManager(DEFAULT_LLM_MODEL).get_llm()
    #     compiled_graph_viz = compile_migration_graph(dummy_llm,
    #                                                  DEFAULT_OBJECT_TYPE)  # Pass a default object_type for graph structure
    #
    #     # Get the underlying graph object
    #     graph_to_draw = compiled_graph_viz.get_graph()
    #
    #     # Draw to a BytesIO object
    #     graph_bytes = graph_to_draw.draw_png()
    #
    #     # Display the image
    #     st.image(Image.open(io.BytesIO(graph_bytes)), caption="LangGraph Agent Workflow", use_column_width=True)
    #
    # except FileNotFoundError:
    #     st.warning(
    #         "Graphviz not found. Please install Graphviz system-wide to view the workflow visualization. (e.g., `sudo apt-get install graphviz` on Linux, or download from graphviz.org for Windows/macOS)")
    #     logger.warning("Graphviz system installation not found for drawing workflow.")
    # except Exception as e:
    #     st.error(
    #         f"Could not generate workflow visualization: {e}. Ensure 'pydot' and 'graphviz' Python packages are installed, and Graphviz is installed on your system.")
    #     logger.error(f"Error generating workflow visualization: {e}", exc_info=True)
    #
    # st.markdown("---")

    # --- Run Migration Button ---
    st.header("2. Run Migration")
    if st.button("🚀 Start Migration", type="primary", use_container_width=True):
        if not oracle_code_inputs:
            st.error(f"Please upload or paste at least one Oracle {selected_object_type} DDL to migrate.")
            logger.error(f"No Oracle {selected_object_type} DDL provided for migration.")
            return

        st.info(f"Migration process started for {selected_object_type}(s). This may take a few moments...")
        logger.info(
            f"Migration initiated for {len(oracle_code_inputs)} Oracle {selected_object_type}(s) using LLM: {selected_llm_model}")

        llm_manager = LLMManager(selected_llm_model)
        try:
            llm_instance = llm_manager.get_llm()
            # Pass object_type to compile_migration_graph
            compiled_graph = compile_migration_graph(llm_instance, selected_object_type)

            st.session_state.results = {}
            for obj_name, oracle_code in oracle_code_inputs.items():
                st.subheader(f"Migrating {selected_object_type}: `{obj_name}`")
                current_status_placeholder = st.empty()
                progress_bar = st.progress(0)
                status_text = st.empty()

                initial_state = {
                    "oracle_object_code": oracle_code,
                    "snowflake_guidelines": guidelines_content,
                    "sample_snowflake_code": sample_procedures_content,  # Direct string from UI
                    "generated_snowflake_code": "",
                    "reflection": "",
                    "errors": [],
                    "current_step": "Start",
                    "optimization_cycle_count": 0,
                    "max_optimization_cycles": optimization_cycles,
                    "object_type": selected_object_type,
                    "decomposed_oracle_view_parts": {},  # Initialize
                    "translated_snowflake_view_parts": {},  # Initialize
                    "is_decomposed": False  # Initialize
                }

                # Approximate total steps adjusted for decomposition
                total_steps_adjusted = 5  # base steps
                if selected_object_type == "View" and (oracle_code.count('\n') + 1) > VIEW_DECOMPOSITION_THRESHOLD:
                    total_steps_adjusted += 2  # Add decomposition and assembly steps

                try:
                    full_stream = compiled_graph.stream(initial_state, config={"llm": llm_instance})
                    for i, s in enumerate(full_stream):
                        if "__end__" not in s:
                            current_state_dict = list(s.values())[0]
                            if "current_step" in current_state_dict:
                                status_text.text(f"Processing step: {current_state_dict['current_step']}")
                            # Adjust progress calculation to be more robust with dynamic loops
                            current_progress = min((i + 1) / (total_steps_adjusted * 2), 1.0)  # Heuristic for progress
                            progress_bar.progress(current_progress)
                            final_state = current_state_dict
                        else:
                            final_state = s["__end__"]

                    st.session_state.results[obj_name] = final_state
                    progress_bar.progress(1.0)
                    status_text.success(f"Migration for {selected_object_type} `{obj_name}` complete!")
                    logger.info(f"Migration successful for {selected_object_type} {obj_name}")

                except Exception as e:
                    logger.error(f"Error during migration for {obj_name}: {e}", exc_info=True)
                    st.error(f"An error occurred during migration for {selected_object_type} `{obj_name}`: {e}")
                    st.session_state.results[obj_name] = {
                        "generated_snowflake_code": "Error during migration. Please check logs.",
                        "reflection": f"Migration failed due to: {e}",
                        "errors": [str(e)],
                        "object_type": selected_object_type  # Ensure object type is present even on error
                    }

        except InvalidInputError as e:
            st.error(f"Input Error: {e}")
            logger.error(f"Input error: {e}", exc_info=True)
        except MigrationError as e:
            st.error(f"Migration specific error: {e}")
            logger.error(f"Migration specific error: {e}", exc_info=True)
        except Exception as e:
            st.error(f"An unexpected error occurred during LLM initialization or graph compilation: {e}")
            logger.critical(f"Critical error: {e}", exc_info=True)

    # --- Results Section ---
    st.header("3. Migration Results")

    if "results" in st.session_state and st.session_state.results:
        tabs = st.tabs(list(st.session_state.results.keys()))
        all_results_markdown = ""

        for i, obj_name in enumerate(st.session_state.results.keys()):
            with tabs[i]:
                result = st.session_state.results[obj_name]
                st.subheader(f"Result for `{obj_name}`")

                # Dynamically set code language based on object type (SQL for views, SQL/JS for procedures)
                code_lang = "sql"
                if result.get("object_type") == "Procedure":
                    # Heuristic: if JS-like keywords, use javascript, else sql
                    if "AS $$" in result.get("generated_snowflake_code", "") or "LANGUAGE JAVASCRIPT" in result.get(
                            "generated_snowflake_code", ""):
                        code_lang = "javascript"
                    else:
                        code_lang = "sql"
                st.markdown(f"#### Generated Snowflake {result.get('object_type', 'Object')}")
                st.code(result.get("generated_snowflake_code", "No code generated."), language=code_lang)

                st.markdown("#### LLM Reflection/Optimization Notes")
                st.text(result.get("reflection", "No reflection notes."))

                if result.get("errors") and any(result["errors"]):
                    st.markdown("#### Errors/Warnings")
                    for error in result["errors"]:
                        st.warning(error)
                else:
                    st.success("No errors reported for this object.")

                # Display decomposed/translated parts for views if applicable
                if result.get("object_type") == "View" and result.get("is_decomposed"):
                    st.markdown("#### Decomposed Oracle View Parts")
                    st.json(result.get("decomposed_oracle_view_parts", {}))
                    st.markdown("#### Translated Snowflake View Parts")
                    st.json(result.get("translated_snowflake_view_parts", {}))

                all_results_markdown += f"""
---
### {result.get('object_type', 'Object')}: `{obj_name}`

#### Generated Snowflake {result.get('object_type', 'Object')}
```{code_lang}
{result.get("generated_snowflake_code", "No code generated.")}
```

#### LLM Reflection/Optimization Notes
```
{result.get("reflection", "No reflection notes.")}
```

#### Errors/Warnings
```
{json.dumps(result.get("errors", []), indent=2)}
```
"""
        st.download_button(
            label="Download All Results (Markdown)",
            data=all_results_markdown,
            file_name="snowflake_migration_results.md",
            mime="text/markdown",
            use_container_width=True
        )
    else:
        st.info("No migration results yet. Upload Oracle DDL and click 'Start Migration'.")


if __name__ == "__main__":
    main()
