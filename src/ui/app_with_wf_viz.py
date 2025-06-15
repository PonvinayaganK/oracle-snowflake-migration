# src/ui/app_with_wf_viz.py
import streamlit as st
import os
import tempfile
import json
import logging
import io  # Import for handling image bytes
from PIL import Image  # Import Pillow for image display

# Set up project-level logging before importing other modules that use it
from src.utils.logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

from src.core.agent_workflow import compile_migration_graph
from src.data_processing.file_utils import save_uploaded_file, read_file_content
from src.llm_config.llm_manager import LLMManager
from src.utils.exceptions import MigrationError, InvalidInputError
# Import OLLAMA_BASE_URL and MAX_OPTIMIZATION_CYCLES from config
from src.config import SUPPORTED_LLM_MODELS, DEFAULT_LLM_MODEL, PROXY_CONFIG, OLLAMA_BASE_URL, MAX_OPTIMIZATION_CYCLES

st.set_page_config(layout="wide", page_title="Oracle to Snowflake Procedure Migrator")


def main():
    st.title("❄️ Oracle to Snowflake Procedure Migrator (AI Powered)")

    st.markdown("""
        This tool uses an LLM-powered AI agent to migrate complex Oracle PL/SQL procedures
        to Snowflake-compatible procedures, leveraging user-defined guidelines and
        sample Snowflake procedures for enhanced accuracy and optimization.
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
    # Show Ollama URL input only for Ollama models
    if selected_llm_model.startswith("ollama-"):
        st.sidebar.subheader("Ollama Server Configuration")
        # Get current Ollama base URL, allowing user to override
        current_ollama_url = os.getenv("OLLAMA_BASE_URL", OLLAMA_BASE_URL)
        ollama_url_input = st.sidebar.text_input(
            "Ollama Base URL (e.g., http://localhost:11434)",
            value=current_ollama_url,
            key="ollama_base_url_input"
        )
        # Update OS env var if user changes it, so LLMManager picks it up
        if ollama_url_input != current_ollama_url:
            os.environ['OLLAMA_BASE_URL'] = ollama_url_input
            st.sidebar.info(
                f"Ollama Base URL set to: {ollama_url_input}. Restart application for full effect if issues persist.")

    # Optimization Cycles Input
    st.sidebar.subheader("Optimization Settings")
    optimization_cycles = st.sidebar.number_input(
        "Max Optimization Cycles (LLM reflection loops):",
        min_value=1,
        max_value=10,  # Arbitrary upper limit, adjust as needed
        value=MAX_OPTIMIZATION_CYCLES,
        step=1,
        help="Number of times the LLM will attempt to refine the generated code based on its self-reflection."
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
            os.environ['http_proxy'] = proxy_http  # For some libraries
        if proxy_https:
            os.environ['HTTPS_PROXY'] = proxy_https
            os.environ['https_proxy'] = proxy_https  # For some libraries
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

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Oracle Procedures (.sql, .prc)")
        uploaded_oracle_files = st.file_uploader(
            "Upload one or more Oracle PL/SQL procedure files",
            type=["sql", "prc"],
            accept_multiple_files=True,
            key="oracle_uploader"
        )
        oracle_code_inputs = {}
        if uploaded_oracle_files:
            for file in uploaded_oracle_files:
                file_path = save_uploaded_file(file, "oracle_procedures")
                oracle_code_inputs[file.name] = read_file_content(file_path)
            st.success(f"Uploaded {len(uploaded_oracle_files)} Oracle procedure(s).")
        else:
            st.info("No Oracle procedures uploaded. You can provide them manually below or use the default example.")
            default_oracle_code = st.text_area("Or paste Oracle PL/SQL procedure here:",
                                               value="""
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
                """, height=300, key="manual_oracle_input")
            if default_oracle_code.strip():
                oracle_code_inputs["manual_input.sql"] = default_oracle_code

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

        st.subheader("Snowflake Sample Procedures (`sample_snowflake_procedures.txt`)")
        uploaded_samples = st.file_uploader(
            "Upload a text file with sample Snowflake procedures (RAG input)",
            type=["txt"],
            accept_multiple_files=False,
            key="samples_uploader"
        )
        sample_procedures_content = ""
        if uploaded_samples:
            samples_path = save_uploaded_file(uploaded_samples, "samples")
            sample_procedures_content = read_file_content(samples_path)
            st.success("Sample Snowflake procedures file uploaded.")
        else:
            st.info("No sample procedures uploaded. Using default example.")
            sample_procedures_content = read_file_content("data/sample_snowflake_procedures.txt")
        st.text_area("Review/Edit Sample Procedures:", value=sample_procedures_content, height=200,
                     key="samples_editor")

    st.markdown("---")

    # --- Workflow Visualization Section ---
    st.header("LangGraph Workflow Visualization")
    st.info("This visualization shows the state transitions of the AI agent.")
    try:
        # Compile graph to get the graph object for visualization
        # Pass a dummy LLM instance, as the visualization doesn't need to invoke LLM
        # This is a workaround if compile_migration_graph expects LLM to compile.
        # Ideally, graph compilation should be independent of LLM instance if possible.
        dummy_llm = LLMManager(DEFAULT_LLM_MODEL).get_llm()  # Using default model for dummy
        compiled_graph_viz = compile_migration_graph(dummy_llm)

        # Get the underlying graph object
        graph_to_draw = compiled_graph_viz.get_graph()

        # Draw to a BytesIO object
        # NOTE: This requires graphviz system installation (e.g., `sudo apt-get install graphviz` on Linux)
        # and `pip install pydot graphviz` (Python packages)
        graph_bytes = graph_to_draw.draw_png()  # This will return bytes

        # Display the image
        st.image(Image.open(io.BytesIO(graph_bytes)), caption="LangGraph Agent Workflow", use_column_width=True)

    except FileNotFoundError:
        st.warning(
            "Graphviz not found. Please install Graphviz system-wide to view the workflow visualization. (e.g., `sudo apt-get install graphviz` on Linux, or download from graphviz.org for Windows/macOS)")
        logger.warning("Graphviz system installation not found for drawing workflow.")
    except Exception as e:
        st.error(
            f"Could not generate workflow visualization: {e}. Ensure 'pydot' and 'graphviz' Python packages are installed, and Graphviz is installed on your system.")
        logger.error(f"Error generating workflow visualization: {e}", exc_info=True)

    st.markdown("---")

    # --- Run Migration Button ---
    st.header("2. Run Migration")
    if st.button("🚀 Start Migration", type="primary", use_container_width=True):
        if not oracle_code_inputs:
            st.error("Please upload or paste at least one Oracle procedure to migrate.")
            logger.error("No Oracle procedures provided for migration.")
            return

        st.info("Migration process started. This may take a few moments...")
        logger.info(
            f"Migration initiated for {len(oracle_code_inputs)} Oracle procedures using LLM: {selected_llm_model}")

        llm_manager = LLMManager(selected_llm_model)
        try:
            llm_instance = llm_manager.get_llm()
            compiled_graph = compile_migration_graph(llm_instance)  # Pass LLM instance to graph compilation

            st.session_state.results = {}
            for proc_name, oracle_code in oracle_code_inputs.items():
                st.subheader(f"Migrating: `{proc_name}`")
                # Placeholder for real-time status update
                current_status_placeholder = st.empty()
                progress_bar = st.progress(0)
                status_text = st.empty()  # THIS IS THE TEXT TO UPDATE FOR CURRENT STEP

                initial_state = {
                    "oracle_procedure_code": oracle_code,
                    "snowflake_guidelines": guidelines_content,
                    "sample_snowflake_code": [sample_procedures_content],
                    "generated_snowflake_code": "",
                    "reflection": "",
                    "errors": [],
                    "current_step": "Start",
                    "optimization_cycle_count": 0,  # Initialize optimization cycle count
                    "max_optimization_cycles": optimization_cycles  # Pass max cycles from UI
                }

                current_progress = 0
                total_steps = 5  # Approximate number of main steps in the graph, this might need dynamic adjustment

                try:
                    full_stream = compiled_graph.stream(initial_state,
                                                        config={"llm": llm_instance})  # Pass LLM instance in config
                    for i, s in enumerate(full_stream):
                        if "__end__" not in s:
                            # LangGraph stream yields {node_name: output_dict}
                            current_state_dict = list(s.values())[0]  # Get the dict from the current node's output
                            if "current_step" in current_state_dict:
                                status_text.text(f"Processing step: {current_state_dict['current_step']}")
                            current_progress = min((i + 1) / (total_steps * 2),
                                                   1.0)  # Multiply total_steps to account for potential loops
                            progress_bar.progress(current_progress)
                            # Update final_state in loop for display if loop terminates mid-stream
                            final_state = current_state_dict
                        else:
                            final_state = s["__end__"]

                    st.session_state.results[proc_name] = final_state
                    progress_bar.progress(1.0)
                    status_text.success(f"Migration for `{proc_name}` complete!")
                    logger.info(f"Migration successful for {proc_name}")

                except Exception as e:
                    logger.error(f"Error during migration for {proc_name}: {e}", exc_info=True)
                    st.error(f"An error occurred during migration for `{proc_name}`: {e}")
                    st.session_state.results[proc_name] = {
                        "generated_snowflake_code": "Error during migration. Please check logs.",
                        "reflection": f"Migration failed due to: {e}",
                        "errors": [str(e)]
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

        for i, proc_name in enumerate(st.session_state.results.keys()):
            with tabs[i]:
                result = st.session_state.results[proc_name]
                st.subheader(f"Result for `{proc_name}`")

                st.markdown("#### Generated Snowflake Procedure")
                st.code(result.get("generated_snowflake_code", "No code generated."), language="sql")

                st.markdown("#### LLM Reflection/Optimization Notes")
                st.text(result.get("reflection", "No reflection notes."))

                if result.get("errors") and any(
                        result["errors"]):  # Check if errors list is not empty and contains actual errors
                    st.markdown("#### Errors/Warnings")
                    for error in result["errors"]:
                        st.warning(error)
                else:
                    st.success("No errors reported for this procedure.")

                all_results_markdown += f"""
---
### Procedure: `{proc_name}`

#### Generated Snowflake Procedure
```sql
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
        st.info("No migration results yet. Upload Oracle procedures and click 'Start Migration'.")


if __name__ == "__main__":
    main()
