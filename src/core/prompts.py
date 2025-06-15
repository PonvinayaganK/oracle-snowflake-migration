# src/core/prompts.py
from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """
You are an expert AI Engineer specializing in Oracle PL/SQL procedures and SQL Views to Snowflake SQL/JavaScript migration. Your goal is to accurately and efficiently convert Oracle database objects into functionally equivalent and optimized Snowflake objects. You must ensure no logic is missed and leverage Snowflake's native capabilities for performance. Pay close attention to data types, control flow (for procedures), SQL dialect differences (for views), error handling, dynamic SQL, and Oracle-specific functions/syntax. You will be provided with user guidelines and sample Snowflake objects for reference.
"""

# NEW PROMPT for View Decomposition
VIEW_DECOMPOSITION_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT + """
    Your task is to decompose a large Oracle SQL View DDL into its logical, independent components.
    Identify and extract the main SELECT statement, all Common Table Expressions (CTEs), and key clauses like WHERE, GROUP BY, ORDER BY, and QUALIFY-relevant logic (e.g., related to ROWNUM).
    If a part is not explicitly present, return an empty string for it.
    Return the decomposition as a JSON object with keys:
    - `view_name`: The name of the view (e.g., 'MY_SCHEMA.MY_VIEW').
    - `ctes`: A dictionary where keys are CTE names and values are their Oracle SQL content.
    - `main_select_body`: The core SELECT statement, excluding the initial `CREATE VIEW ... AS` and any `WITH` clause. This should be the main query structure.
    - `where_clause`: Content of the WHERE clause, including `WHERE` keyword if present.
    - `group_by_clause`: Content of the GROUP BY clause, including `GROUP BY` keyword if present.
    - `order_by_clause`: Content of the ORDER BY clause, including `ORDER BY` keyword if present.
    - `oracle_specific_syntax_notes`: Any specific Oracle-isms (e.g., `(+)`, `ROWNUM`, `CONNECT BY`) and their location or context if found.

    Example JSON structure:
    ```json
    {{
      "view_name": "...",
      "ctes": {{
        "cte_name1": "...",
        "cte_name2": "..."
      }},
      "main_select_body": "SELECT col1, col2 FROM my_table JOIN another_table ON ...",
      "where_clause": "WHERE col3 > 100",
      "group_by_clause": "GROUP BY col1",
      "order_by_clause": "ORDER BY col1 DESC",
      "oracle_specific_syntax_notes": "ROWNUM used in subquery for pagination"
    }}
    ```
    Ensure the JSON is perfectly valid and self-contained.
    """),
    ("user", """
    Oracle View DDL to decompose:
    ```sql
    {oracle_view_code}
    ```
    Migration Guidelines:
    ```
    {snowflake_guidelines}
    ```
    Sample Snowflake Code for context on typical view structure:
    ```sql
    {sample_snowflake_code}
    ```
    Return only the JSON object.
    """)
])

# NEW PROMPT for View Part Translation
VIEW_PART_TRANSLATION_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT + """
    Your task is to translate a specific part of an Oracle SQL View into Snowflake-compatible SQL.
    You will be given the original full Oracle view and the specific part you need to translate.
    Focus only on translating the provided part. Do not attempt to translate the entire view.
    Ensure strict adherence to Snowflake SQL syntax and best practices.
    Pay attention to Oracle-specific functions, data types, and operators in this part.
    """),
    ("user", """
    Translate the following Oracle SQL view part: `{oracle_view_part_name}` into Snowflake SQL.

    Oracle SQL Part:
    ```sql
    {oracle_view_part_code}
    ```

    Full Original Oracle View DDL (for context on overall view structure and dependencies):
    ```sql
    {full_oracle_view_code}
    ```

    Migration Guidelines:
    ```
    {snowflake_guidelines}
    ```

    Relevant Sample Snowflake Code for context:
    ```sql
    {sample_snowflake_code}
    ```

    Return only the translated Snowflake SQL for this part.
    """)
])

# NEW PROMPT for View Assembly
VIEW_ASSEMBLY_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT + """
    Your task is to assemble previously translated Snowflake SQL view parts into a complete, valid Snowflake View DDL.
    You will be given the original Oracle view (for reference on structure) and the translated Snowflake parts.
    Combine these parts correctly, ensuring proper syntax, order of CTEs, main query, and clauses.
    The final output should be a single `CREATE OR REPLACE VIEW ... AS SELECT ...` statement.
    """),
    ("user", """
    Original Oracle View DDL (for structure reference):
    ```sql
    {original_oracle_view_code}
    ```

    Translated Snowflake View Parts (JSON format):
    ```json
    {translated_snowflake_view_parts}
    ```

    Migration Guidelines:
    ```
    {snowflake_guidelines}
    ```

    Assemble these parts into a complete Snowflake View DDL. Return only the SQL DDL.
    """)
])

# MODIFIED PROMPT for Code Generation (for non-decomposed objects)
CODE_GENERATION_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("user", """
    Here is an Oracle {object_type} to migrate:
    ```sql
    {oracle_object_code}
    ```

    Here are the migration guidelines and best practices:
    ```
    {snowflake_guidelines}
    ```

    Here are sample Snowflake {object_type}s that have been successfully migrated and validated (use these as a learning reference for Snowflake syntax and patterns):
    ```sql
    {sample_snowflake_code}
    ```

    Based on the above, generate the equivalent Snowflake {object_type}.

    If the object type is 'Procedure':
    - Ensure all procedural logic is covered.
    - Use Snowflake Scripting by default unless JavaScript is explicitly better for a specific pattern (e.g., complex JSON parsing).
    - Handle Oracle-specific constructs like cursors, `LOOP`s, `RAISE_APPLICATION_ERROR`, `SYSDATE`, `DUAL` table appropriately.

    If the object type is 'View':
    - Focus strictly on SQL dialect conversion. Do not generate procedural code.
    - Translate Oracle-specific SQL syntax (e.g., `(+)` outer joins, `ROWNUM`, `DECODE`, `CONNECT BY` if present) to Snowflake's equivalent ANSI SQL or optimized Snowflake features (e.g., `QUALIFY` for `ROW_NUMBER()`).
    - Remove `WITH READ ONLY` or other Oracle-specific view options not applicable to Snowflake.

    In all cases, optimize for Snowflake's architecture. Include comments where complex transformations were made, assumptions about Oracle-specific features, or where manual review might be beneficial.
    """)
])

# MODIFIED PROMPT for Optimization/Reflection
OPTIMIZATION_REFLECTION_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("user", """
    Original Oracle {object_type}:
    ```sql
    {original_oracle_object_code}
    ```

    Generated Snowflake {object_type}:
    ```sql
    {generated_snowflake_code}
    ```

    Review the generated Snowflake code against the original Oracle code and the provided guidelines/samples.
    1.  **Logical Equivalence:** Is any logic missed from the Oracle {object_type}?
    2.  **Snowflake Optimization:** Is the code optimized for Snowflake's columnar, MPP architecture? (e.g., using `ARRAY_AGG` instead of `LISTAGG` where appropriate, avoiding row-by-row processing, efficient `MERGE` statements, proper use of `QUALIFY`, set-based operations for views).
    3.  **Adherence to Guidelines:** Does it follow the naming conventions and best practices from the guidelines for a Snowflake {object_type}?
    4.  **Syntax & Idiom:** Is it idiomatic Snowflake SQL/Scripting or JavaScript (for procedures) or pure SQL (for views)?

    Provide your reasoning and suggest specific improvements or corrections.
    If the code is already optimized and accurate, state 'No further optimization required.' and explain why.
    If improvements are needed, indicate 'Improvements required.' and detail them concisely.
    """)
])

# Remaining prompts unchanged
INITIAL_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("user", """
    Analyze the following Oracle {object_type} DDL. Identify its core logic, variables, control flow (if procedure), SQL constructs (if view), cursor usage (if procedure), exception handling (if procedure), and any Oracle-specific features that will require special attention for migration to Snowflake. Provide a summary of the {object_type}'s purpose and a list of identified migration challenges.

    Oracle {object_type}:
    ```sql
    {oracle_object_code}
    ```
    """)
])

RAG_QUERY_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("user", """
    Based on the following analysis of an Oracle {object_type}, identify key terms and concepts that would be most relevant for providing context or examples for its migration. This output is for internal use to guide the agent's focus on provided samples.

    Oracle {object_type} Analysis:
    {oracle_analysis}

    List of relevant keywords/concepts:
    """)
])