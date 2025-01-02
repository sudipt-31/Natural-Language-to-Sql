import sqlite3
import os
from typing import List
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
import tiktoken
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
import pandas as pd
import re
import io
from sqlalchemy import create_engine, text


# Configure Streamlit page
st.set_page_config(
    page_title="Data Analysis App",
    page_icon="🖼️",
    layout="wide"
)

# Load environment variables
load_dotenv()

def generate_dataset_specific_topics(df, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
       
        df = df.head(40)
        # Prepare a more detailed content analysis
        dataset_content = {
            "column_names": list(df.columns),
            "unique_values": {},
            "data_samples": {}
        }
 
        # Collect unique values and samples for each column
        for column in df.columns:
            # Collect unique values
            unique_values = df[column].dropna().unique()
            dataset_content["unique_values"][column] = list(unique_values[:50])  # Limit to first 50 unique values
           
            # Collect data samples
            if pd.api.types.is_numeric_dtype(df[column]):
                dataset_content["data_samples"][column] = {
                    "type": "numeric",
                    "samples": list(df[column].dropna().sample(min(10, len(df[column]))).values)
                }
            elif pd.api.types.is_datetime64_any_dtype(df[column]):
                dataset_content["data_samples"][column] = {
                    "type": "datetime",
                    "samples": list(df[column].dropna().sample(min(10, len(df[column]))).astype(str).values)
                }
            else:
                dataset_content["data_samples"][column] = {
                    "type": "text",
                    "samples": list(df[column].dropna().sample(min(10, len(df[column]))).values)
                }
        
        prompt = f"""
        OBJECTIVE:
        Generate a comprehensive, hierarchical list of ALL topics and subtopics from the provided dataset content using a numbered and sub-numbered format.
 
        DATASET OVERVIEW:
        Columns: {len(df.columns)}
        Rows: {len(df)}
       
        DATASET CONTENT:
        {dataset_content}
 
        OUTPUT FORMAT REQUIREMENTS:
        1. Use numbered main topics (1., 2., 3., etc.)
        2. Use single indented bullet points (-) for subtopics
        3. No asterisks, parentheses, or explanatory notes
        4. No colons after topic names
        5. Clean, consistent hierarchy
        6. No line breaks between topics and their subtopics
        7. One line break between main topic sections
 
        EXAMPLE FORMAT:
        1. Main Topic Name
           - Subtopic one
           - Subtopic two
           - Subtopic three
 
        2. Second Main Topic
           - Subtopic one
           - Subtopic two
           - Subtopic three
 
        CRITICAL REQUIREMENTS:
        - Extract ALL possible topics and subtopics from the dataset
        - Base topics SOLELY on information present in the data
        - Include both explicit and implicit topics
        - Ensure comprehensive coverage
        - Keep topics distinct and non-overlapping
        - Use clear, specific language
        - Group related content logically
        - Maintain consistent formatting
        - No explanatory text or descriptions
        - No duplicate topics or subtopics
        - Include ALL relevant categories from the data
 
        Note: The response should be complete and self-contained, without breaking into multiple parts or requiring continuation.
        """
       
        response = model.generate_content(prompt)
        return response.text
   
    except Exception as e:
        st.error(f"Error: {e}")
        return None
 
 
def generate_subtopic_matching_csv(df, subtopics, output_filename='subtopic_matching_matrix.csv'):
    """
    Generate a CSV file that matches the first column of the dataframe
    against extracted subtopics and removes columns with all zero values.
    Ensures no repeated column names by adding a unique identifier.
   
    Args:
        df (pandas.DataFrame): Original dataframe
        subtopics (list): List of extracted subtopics
        output_filename (str): Name of the output CSV file
   
    Returns:
        pandas.DataFrame: Matching matrix dataframe with zero-value columns removed
    """
    # Ensure we have a copy of the original dataframe
    original_df = df.copy()
   
    # Create a new dataframe with the first column and subtopic columns
    # First column becomes the first column of the new dataframe
    first_column_name = df.columns[0]
    matching_df = pd.DataFrame({first_column_name: original_df[first_column_name]})
   
    # Convert first column to string to ensure consistent matching
    matching_df[first_column_name] = matching_df[first_column_name].astype(str)
   
    # Track column names to avoid duplicates
    used_column_names = {first_column_name}
   
    # For each subtopic, create a column and match against all columns
    for subtopic in subtopics:
        # Generate a unique column name
        base_column_name = subtopic
        counter = 1
        column_name = base_column_name
        while column_name in used_column_names:
            column_name = f"{base_column_name}_{counter}"
            counter += 1
        
        # Add to used column names
        used_column_names.add(column_name)
       
        # Initialize the column with zeros
        matching_df[column_name] = 0
       
        # Check against all columns in the original dataframe
        for column in original_df.columns:
            # Convert column to string and apply case-insensitive matching
            column_data = original_df[column].astype(str).str.lower()
            matching_rows = column_data.str.contains(subtopic.lower(), na=False)
           
            # Update the matching column where matches are found
            matching_df.loc[matching_rows, column_name] = 1
   
    # Remove columns with all zero values (except the first column)
    columns_to_keep = [first_column_name] + [
        col for col in matching_df.columns
        if col != first_column_name and matching_df[col].sum() > 0
    ]
    matching_df = matching_df[columns_to_keep]
   
    # Save to CSV
    matching_df.to_csv(output_filename, index=False)
   
    # Print number of columns removed (if any)
    original_num_columns = len(columns_to_keep)
    print(f"Columns removed due to all zero values: {len(subtopics) + 1 - original_num_columns}")
   
    return matching_df
 

###########################################################

def count_tokens(text):
    """Count tokens using tiktoken"""
    encoder = tiktoken.get_encoding("cl100k_base")  # or another encoding
    return len(encoder.encode(text))

def get_prompt_template():
    """Get the prompt template with dynamic schema"""
    template = """
You are an expert SQL query generator for a dynamic database. Given a natural language question, generate the appropriate SQL query based on the following schema:

DATABASE SCHEMA:
{schema}

IMPORTANT RULES FOR SQL QUERY GENERATION:
1. Return ONLY the SQL query without explanations or comments.
2. Use appropriate JOIN clauses for combining tables.
3. If tables have columns with the same name, use aliases for each table.
4. If two columns are identical across tables, merge them into a single column by selecting only one.
5. Use relevant WHERE clauses for filtering and specify join conditions clearly.
6. Include aggregation functions (COUNT, AVG, SUM) when required.
7. Use GROUP BY for aggregated results and ORDER BY for sorting when applicable.
8. Select only needed columns instead of using *.
9. Always limit results to 10 rows unless asked otherwise.
10. Clearly assign aliases to each table and reference all columns with table aliases.
11. Always check the table schema and column names to ensure correct references.
12. For age calculations use: CAST(CAST(JULIANDAY(CURRENT_TIMESTAMP) - JULIANDAY(CAST(birth_year AS TEXT) || '-01-01') AS INT) / 365 AS INT)
13. Ensure foreign key relationships are correctly used in JOINs.
14. Use aggregation functions with actual columns from the correct table.
15.2. Use table aliases, but **do not use the 'AS' keyword for aliases** (e.g., `uploaded_data ud` instead of `uploaded_data AS ud`).
16. the SQL query should be able to run in SQLite.
17. most Don't use this format  ```sql ```  only give me the sql query.
User Question: {question}

Generate the SQL query that answers this question:
"""
    return PromptTemplate(
        input_variables=["question", "schema"],
        template=template
    )

def setup_llm():
    """Initialize Gemini and chain with dynamic schema"""
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            google_api_key="AIzaSyC5Dqjx0DLbkRXH9YWqWZ1SPTK0w0C4oFY",
            convert_system_message_to_human=True
        )
        prompt = get_prompt_template()
        return prompt, llm
    except Exception as e:
        st.error(f"Error setting up LLM: {str(e)}")
        return None, None

def generate_result_explanation(results_df, user_question, llm):
    """
    Generate a comprehensive natural language explanation of the query results using LLM.
    
    Args:
        results_df (pd.DataFrame): The query results DataFrame
        user_question (str): Original user question
        llm: The language model instance
    
    Returns:
        str: A detailed explanation of the results
    """
    try:
        # Convert DataFrame to a more readable format for analysis
        data_description = []
        
        # Get basic DataFrame info
        total_rows = len(results_df)
        total_cols = len(results_df.columns)
        
        # Analyze each column
        column_insights = []
        for column in results_df.columns:
            col_data = results_df[column]
            
            # Get column type and basic stats
            dtype = str(col_data.dtype)
            non_null_count = col_data.count()
            
            if pd.api.types.is_numeric_dtype(col_data):
                # Numeric column analysis
                stats = {
                    'mean': float(col_data.mean()) if not col_data.empty else 0,
                    'min': float(col_data.min()) if not col_data.empty else 0,
                    'max': float(col_data.max()) if not col_data.empty else 0
                }
                column_insights.append(f"Column '{column}' (numeric):\n"
                                    f"- Range: {stats['min']} to {stats['max']}\n"
                                    f"- Average: {stats['mean']:.2f}")
            else:
                # Categorical column analysis
                unique_values = col_data.nunique()
                most_common = col_data.value_counts().head(3).to_dict()
                column_insights.append(f"Column '{column}' (categorical):\n"
                                    f"- Unique values: {unique_values}\n"
                                    f"- Most common values: {most_common}")

        # Create analysis prompt
        analysis_prompt = f"""Analyze the following data for '{user_question}': {chr(10).join(column_insights)}
                              Sample Data: {results_df.head(3).to_markdown() if not results_df.empty else 'No data available'} 
                              Provide a concise 1-2 sentence answer focusing on key findings and direct response to the question."""

        # Get explanation from LLM
        response = llm.invoke(analysis_prompt)
        explanation = response.content if hasattr(response, 'content') else str(response)
        
        # Format the explanation with markdown
        formatted_explanation = f"""
        ### 📊 Data Analysis Results

        {explanation}

        """
        
        return formatted_explanation

    except Exception as e:
        error_message = f"""
        ### ⚠️ Analysis Error
        
        We encountered an issue while analyzing the results: {str(e)}
        
        Basic Information:
        - Number of records: {len(results_df)}
        - Columns: {', '.join(results_df.columns)}
        """
        print(f"Error in generate_result_explanation: {str(e)}")
        return error_message

################################

def extract_key_phrases(question: str) -> List[str]:
    """
    Extract key phrases from a question using Gemini model analysis.
    Only extracts words that are present in the original question.
    
    Args:
        question (str): The question to analyze
        
    Returns:
        List[str]: List of extracted key phrases from the question
    """
    if not question:
        return []
        
    try:
        genai.configure(api_key="AIzaSyC5Dqjx0DLbkRXH9YWqWZ1SPTK0w0C4oFY")
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""
        OBJECTIVE:
        Extract ONLY the important topic words directly from the given question. Do not add any new words or related topics.
        
        RULES:
        1. Only use words that appear in the original question
        2. Remove common words (articles, prepositions, etc.)
        3. Do not add synonyms or related terms
        4. Do not create new categories or topics
        5. Focus on nouns and key terms
        
        QUESTION:
        {question}
        
        OUTPUT FORMAT:
        - Return only the extracted words, one per line
        - No explanations or additional text
        - No categorization or grouping
        """
        
        response = model.generate_content(prompt)
        
        # Extract topics from response
        topics = [
            line.strip().strip('- ').lower()
            for line in response.text.split('\n')
            if line.strip() and not line.startswith('#') and not line.startswith('OUTPUT')
        ]
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(topics))
        
    except Exception as e:
        print(f"Error in key phrase extraction: {e}")
        return []

###################################


def standardize_dataframe(df):
    """
    Standardize DataFrame while preserving original column names and all data.
    
    Args:
        df: pandas.DataFrame to standardize
        
    Returns:
        pandas.DataFrame: Standardized DataFrame with preserved data
    """
    if df.empty or len(df.columns) == 0:
        return pd.DataFrame()

    # Create a copy of the DataFrame to avoid modifying the original
    df_new = df.copy()
    
    # Clean up the DataFrame
    df_new = df_new.dropna(how='all', axis=0)  # Remove empty rows
    df_new = df_new.dropna(how='all', axis=1)  # Remove empty columns
    
    # Reset index to ensure clean numbering
    df_new = df_new.reset_index(drop=True)
    
    # If DataFrame is empty after cleaning, return empty DataFrame
    if len(df_new.columns) == 0:
        return pd.DataFrame()
        
    return df_new

def preprocess_excel_file(uploaded_file):
    """
    Preprocess Excel file maintaining the structure from the Excel sheet.
    
    Args:
        uploaded_file: Uploaded file object from Streamlit
        
    Returns:
        dict or pandas.DataFrame: Processed DataFrames
    """
    try:
        # Get the file bytes
        file_bytes = uploaded_file.read()
        
        # Handle CSV files
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(io.StringIO(file_bytes.decode('utf-8')))
            df = standardize_dataframe(df)
            return df if not df.empty else None
            
        # Handle Excel files
        else:
            excel_bytes = io.BytesIO(file_bytes)
            excel_file = pd.ExcelFile(excel_bytes)
            sheets = excel_file.sheet_names
            
            if len(sheets) > 1:
                dfs = {}
                for sheet in sheets:
                    df = pd.read_excel(excel_bytes, sheet_name=sheet)
                    df = standardize_dataframe(df)
                    if not df.empty and len(df.columns) > 0:
                        dfs[sheet] = df
                return dfs if dfs else None
            else:
                df = pd.read_excel(excel_bytes)
                df = standardize_dataframe(df)
                return df if not df.empty else None
                
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None
    finally:
        uploaded_file.seek(0)



#######################
def main():
    # Ensure the outputs directory exists
    os.makedirs('outputs', exist_ok=True)

    # Initialize session state variables
    if 'current_db' not in st.session_state:
        st.session_state.current_db = None
    if 'subtopic_matching_file' not in st.session_state:
        st.session_state.subtopic_matching_file = 'subtopic_matching_matrix.csv'  # Fixed filename
    if 'generated_topics' not in st.session_state:
        st.session_state.generated_topics = []
    if 'topics_generated' not in st.session_state:
        st.session_state.topics_generated = False
    if 'original_df' not in st.session_state:
        st.session_state.original_df = None
    if 'matching_df' not in st.session_state:
        st.session_state.matching_df = None
    if 'total_columns_added' not in st.session_state:
        st.session_state.total_columns_added = 0
    if 'columns_tracker' not in st.session_state:
        st.session_state.columns_tracker = []
    if 'all_added_columns' not in st.session_state:
        st.session_state.all_added_columns = set()
    if 'file_processed' not in st.session_state:
        st.session_state.file_processed = False
    if 'llm' not in st.session_state:
        st.session_state.llm = None
    if 'prompt_template' not in st.session_state:
        st.session_state.prompt_template = None
        
    st.title("🔬 SQL Query Analysis Platform")
    st.subheader("SQL-Based Data Exploration")
    
    # Store API key in session state if not already present
    if 'gemini_api_key' not in st.session_state:
        st.session_state.gemini_api_key = "AIzaSyC5Dqjx0DLbkRXH9YWqWZ1SPTK0w0C4oFY"

    # Initialize LLM and prompt template at the start
    if st.session_state.llm is None or st.session_state.prompt_template is None:
        try:
            st.session_state.prompt_template, st.session_state.llm = setup_llm()
            if st.session_state.llm is None:
                st.error("Failed to initialize LLM. Please check your API key and try again.")
                return
        except Exception as e:
            st.error(f"Error initializing LLM: {str(e)}")
            return

    # Display column statistics if available
    if st.session_state.matching_df is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Columns", len(st.session_state.matching_df.columns))
        with col2:
            st.metric("Original Topics", st.session_state.total_columns_added)
        with col3:
            st.metric("Added Topics", len(st.session_state.columns_tracker))

    # File upload section
    uploaded_file = st.file_uploader("Choose a file", type=['xlsx', 'xls', 'csv'])

    if uploaded_file is not None and not st.session_state.file_processed:
        try:
            # Process the uploaded file using the new preprocessing function
            processed_data = preprocess_excel_file(uploaded_file)
            
            if processed_data is None:
                st.error("No valid data found in the uploaded file.")
                return
            
            if isinstance(processed_data, dict):
                # Multiple sheets detected
                valid_sheets = {name: df for name, df in processed_data.items() 
                              if not df.empty and len(df.columns) > 0}
                
                if not valid_sheets:
                    st.error("No valid data found in any sheet of the Excel file.")
                    return
                
                st.write("Found multiple sheets in the Excel file:")
                sheet_names = list(valid_sheets.keys())
                selected_sheet = st.selectbox("Select a sheet to analyze:", sheet_names)
                df_uploaded = valid_sheets[selected_sheet]
                
                # Create database with all sheets
                with st.spinner("Creating database tables for all sheets..."):
                    db_name = 'subtopic_matching_database.db'
                    engine = create_engine(f'sqlite:///{db_name}')
                    
                    # Save each valid sheet as a separate table
                    for sheet_name, sheet_df in valid_sheets.items():
                        if not sheet_df.empty and len(sheet_df.columns) > 0:
                            # Create a valid table name from sheet name
                            table_name = re.sub(r'[^\w]', '_', sheet_name.lower())
                            try:
                                sheet_df.to_sql(table_name, engine, if_exists='replace', index=False)
                                st.success(f"Created table: {table_name}")
                            except Exception as e:
                                st.warning(f"Failed to create table for sheet {sheet_name}: {str(e)}")
                    
                    engine.dispose()
            else:
                # Single sheet/CSV file
                if processed_data.empty or len(processed_data.columns) == 0:
                    st.error("No valid data found in the file.")
                    return
                df_uploaded = processed_data
            
            # Generate topics only once
            if not st.session_state.topics_generated:
                with st.spinner("Extracting dataset topics..."):
                    topics_text = generate_dataset_specific_topics(df_uploaded, st.session_state.gemini_api_key)
                
                if topics_text:
                    subtopics = [
                        line.strip().lstrip('- ').strip()
                        for line in topics_text.split('\n')
                        if line.strip() and line.strip().startswith('-')
                    ]
                    
                    st.session_state.generated_topics = subtopics
                    st.session_state.topics_generated = True
                else:
                    st.error("Could not generate topics from the dataset.")
                    return

            # Generate matching CSV and store DataFrame in session state
            if st.session_state.generated_topics:
                with st.spinner("Creating subtopic matching matrix..."):
                    matching_df = generate_subtopic_matching_csv(
                        df_uploaded, 
                        st.session_state.generated_topics, 
                        st.session_state.subtopic_matching_file
                    )
                    st.session_state.matching_df = matching_df.copy()
                    st.session_state.total_columns_added = len(matching_df.columns) - 1

            # Create database
            if os.path.exists(st.session_state.subtopic_matching_file):
                with st.spinner("Creating database from matching file..."):
                    db_name = 'subtopic_matching_database.db'
                    
                    engine = create_engine(f'sqlite:///{db_name}')
                    st.session_state.matching_df.to_sql('subtopic_matching', engine, if_exists='replace', index=False)
                    engine.dispose()
                    
                    st.session_state.current_db = db_name
                    st.success("Database created successfully!")
                    st.session_state.file_processed = True

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.exception(e)  # This will show the full error traceback
            return

    # Analysis Interface
    st.markdown("### 🔍 SQL Query Analysis")
    user_question = st.text_area("Enter your analysis question", height=100)

    if st.button("Execute Query"):
        if not user_question:
            st.warning("Please enter a question.")
            return

        if not st.session_state.current_db:
            st.error("Please upload a dataset first.")
            return

        with st.spinner("Processing question and updating database..."):
            try:
                # Extract new topics from the question
                key_phrases = extract_key_phrases(user_question)
                new_columns_added = []
                
                # Add new topics to the matching DataFrame
                if st.session_state.matching_df is not None:
                    for phrase in key_phrases:
                        # Clean and standardize column name
                        column_name = re.sub(r'[^\w\s]', '', phrase).replace(' ', '_').lower()
                        
                        # Check if column exists case-insensitively
                        existing_columns = [col.lower() for col in st.session_state.matching_df.columns]
                        if column_name.lower() not in existing_columns:
                            # Create new column with matches
                            st.session_state.matching_df[column_name] = st.session_state.matching_df.apply(
                                lambda row: int(any(phrase.lower() in str(val).lower() for val in row.values)),
                                axis=1
                            )
                            
                            # Add column to tracking set
                            st.session_state.all_added_columns.add(column_name)
                            new_columns_added.append(column_name)
                            st.session_state.columns_tracker.append(column_name)

                    # Update the CSV file with new columns
                    st.session_state.matching_df.to_csv(st.session_state.subtopic_matching_file, index=False)

                # Show information about new columns
                if new_columns_added:
                    st.info(f"Added {len(new_columns_added)} new topics: {', '.join(new_columns_added)}")
                
                # Update database
                engine = create_engine(f'sqlite:///{st.session_state.current_db}')
                with engine.connect() as conn:
                    conn.execute(text('DROP TABLE IF EXISTS subtopic_matching'))
                st.session_state.matching_df.to_sql('subtopic_matching', engine, if_exists='replace', index=False)
                engine.dispose()
                
                # Get updated schema information
                conn = sqlite3.connect(st.session_state.current_db)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                
                schema_str = ""
                for table in tables:
                    table_name = table[0]
                    cursor.execute(f"PRAGMA table_info({table_name});")
                    columns = cursor.fetchall()
                    
                    schema_str += f"Table: {table_name}\n"
                    schema_str += "Columns:\n"
                    for col in columns:
                        schema_str += f"- {col[1]} ({col[2]})\n"
                    schema_str += "\n"
                
                conn.close()

                # Generate and execute SQL query
                response = st.session_state.llm.invoke(st.session_state.prompt_template.format(
                    question=user_question,
                    schema=schema_str
                ))

                sql_query = response.content if hasattr(response, 'content') else str(response)

                # Display the generated SQL query
                with st.expander("View Generated SQL Query"):
                    st.code(sql_query, language="sql")

                # Execute query and display results
                try:
                    conn = sqlite3.connect(st.session_state.current_db)
                    results = pd.read_sql_query(sql_query, conn)
                    conn.close()

                    if not results.empty:
                        st.subheader("Query Results")
                        st.dataframe(results)
                        
                        # Generate explanation
                        explanation = generate_result_explanation(results, user_question, st.session_state.llm)
                        st.markdown("### Analysis Explanation")
                        st.markdown(explanation)

                        # Save results option
                        if st.button("Save Results to CSV"):
                            results_path = os.path.join('outputs', f'query_results_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.csv')
                            results.to_csv(results_path, index=False)
                            st.success(f"Results saved to {results_path}")
                    else:
                        st.warning("No results found for this query.")

                except sqlite3.OperationalError as sql_error:
                    st.error(f"SQL Error: {str(sql_error)}")
                    st.error("Please try rephrasing your question with more specific terms.")
                except Exception as query_error:
                    st.error(f"Error executing query: {str(query_error)}")

            except Exception as e:
                st.error(f"Analysis error: {str(e)}")
                st.error("Please try rephrasing your question.")


if __name__ == "__main__":
    main()
    
    
    
    
    