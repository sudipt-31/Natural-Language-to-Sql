import sqlite3
import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
# from langchain_ollama import ChatOllama
import io
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import speech_recognition as sr
from audio_recorder_streamlit import audio_recorder
import tiktoken
from typing import Dict, Any
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI


# Configure Streamlit page
st.set_page_config(
    page_title="Data Analysis App",
    page_icon="🖼️",
    layout="wide"
)

# Load environment variables
load_dotenv()

# Database initialization function
def init_database():
    try:
        conn = sqlite3.connect('bollywood_analysis.db')
        cursor = conn.cursor()
        
        # Check if data files exist before reading
        required_files = [
    'data/actors.csv', 'data/directors.csv', 'data/genres.csv', 
    'data/movies.csv', 'data/awards.csv', 'data/movie_actors.csv', 
    'data/movie_performance.csv'
]
        
        for file in required_files:
            if not os.path.exists(file):
                st.error(f"Missing required file: {file}")
                return False
        
        # Read CSV files with error handling
        try:
            actors_df = pd.read_csv('data/actors.csv')
            directors_df = pd.read_csv('data/directors.csv')
            genres_df = pd.read_csv('data/genres.csv')
            movies_df = pd.read_csv('data/movies.csv')
            awards_df = pd.read_csv('data/awards.csv')
            movie_actors_df = pd.read_csv('data/movie_actors.csv')
            movie_performance_df = pd.read_csv('data/movie_performance.csv')
        except Exception as e:
            st.error(f"Error reading CSV files: {str(e)}")
            return False
        
        # Save to SQLite with error handling
        try:
            actors_df.to_sql('actors', conn, if_exists='replace', index=False)
            directors_df.to_sql('directors', conn, if_exists='replace', index=False)
            genres_df.to_sql('genres', conn, if_exists='replace', index=False)
            movies_df.to_sql('movies', conn, if_exists='replace', index=False)
            awards_df.to_sql('awards', conn, if_exists='replace', index=False)
            movie_actors_df.to_sql('movie_actors', conn, if_exists='replace', index=False)
            movie_performance_df.to_sql('movie_performance', conn, if_exists='replace', index=False)
        except Exception as e:
            st.error(f"Error creating database tables: {str(e)}")
            return False
        
        conn.close()
        return True
        
    except Exception as e:
        st.error(f"Database initialization error: {str(e)}")
        return False

# Predefined queries dictionary
PREDEFINED_QUERIES = {
    "What are the top 5 highest-grossing movies?": """
        
    """,
    "Show me movies directed by Sanjay Leela Bhansali": """
        
    """,
    "What is the average IMDb rating by genre?": """
        
    """
}

def extract_keywords_from_question(question: str) -> set:
    """Extract meaningful keywords from the question"""
    # Remove common stop words and SQL-related terms
    stop_words = {
        'what', 'which', 'how', 'when', 'where', 'who', 'show', 'list', 'find',
        'get', 'tell', 'give', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'up', 'about', 'into', 'over', 'after'
    }
    
    # Tokenize and clean the question
    words = question.lower().split()
    keywords = {word for word in words if word not in stop_words}
    return keywords

def calculate_table_relevance_scores(keywords: set, schema_info: dict) -> dict:
    """Calculate relevance scores for each table based on keyword matches"""
    relevance_scores = {}
    
    for table_name, columns in schema_info.items():
        score = 0
        # Check table name matches
        table_words = set(table_name.lower().split('_'))
        score += len(keywords.intersection(table_words)) * 2  # Higher weight for table name matches
        
        # Check column matches
        column_words = set()
        for col in columns:
            col_words = set(col['name'].lower().split('_'))
            column_words.update(col_words)
        
        score += len(keywords.intersection(column_words))
        
        # Store score if there's any match
        if score > 0:
            relevance_scores[table_name] = score
            
    return relevance_scores

def get_related_tables(table_name: str, schema_info: dict) -> set:
    """Get related tables through foreign key relationships"""
    related_tables = {table_name}
    
    # Check foreign key relationships
    for col in schema_info[table_name]:
        if 'type' in col and 'FOREIGN KEY' in str(col['type']):
            # Extract referenced table name from foreign key
            referenced_table = col['type'].split('-> ')[1].split('.')[0]
            related_tables.add(referenced_table)
            
    return related_tables

def prune_schema_for_question(question: str, full_schema: dict) -> dict:
    """Prune schema to only include relevant tables for the question"""
    # Extract keywords from question
    keywords = extract_keywords_from_question(question)
    
    # Calculate relevance scores for tables
    relevance_scores = calculate_table_relevance_scores(keywords, full_schema)
    
    # If no direct matches found, return full schema
    if not relevance_scores:
        return full_schema
        
    # Get primary relevant tables and their related tables
    relevant_tables = set()
    for table in relevance_scores:
        relevant_tables.update(get_related_tables(table, full_schema))
        
    # Create pruned schema
    pruned_schema = {
        table: columns 
        for table, columns in full_schema.items() 
        if table in relevant_tables
    }
    
    return pruned_schema

def get_dynamic_schema_for_question(question: str) -> str:
    """Get pruned schema formatted for the prompt."""
    try:
        # Ensure the current database is set
        db_name = st.session_state.current_db
        if not db_name:
            raise ValueError("No database selected.")

        # Connect to the database
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        # Get all tables in the database
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [table[0] for table in cursor.fetchall()]

        if not tables:
            raise ValueError("No tables found in the database.")

        # Retrieve schema information for all tables
        schema_info = []
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table});")
            columns = cursor.fetchall()

            # Format columns into schema
            formatted_columns = [
                f"   - {col[1]} ({col[2]})" for col in columns
            ]
            schema_info.append(f"{table}\n" + "\n".join(formatted_columns))

        conn.close()
        return "\n\n".join(schema_info)

    except Exception as e:
        st.error(f"Error getting dynamic schema: {str(e)}")
        return None
# Modified prompt template that uses dynamic schema
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
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            convert_system_message_to_human=True
        )
        prompt = get_prompt_template()
        return prompt, llm
    except Exception as e:
        st.error(f"Error setting up LLM: {str(e)}")
        return None, None

def create_database_from_excel(uploaded_file):
    """
    Returns tuple of (database_name, schema_info) or (None, None) if there's an error.
    """
    try:
        # Generate a unique name for the database
        db_name = 'shared_database.db'
        
        # Create SQLite connection
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        
        # Read all sheets from the Excel file
        excel_file = pd.ExcelFile(uploaded_file)
        sheet_names = excel_file.sheet_names
        
        schema_info = {}
        
        # Process each sheet
        for sheet_name in sheet_names:
            try:
                # Read the sheet into a DataFrame
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                
                # Clean and format table name
                table_name = (sheet_name.strip()
                            .replace(' ', '_')
                            .replace('-', '_')
                            .replace('.', '_')
                            .lower())
                
                # Clean column names
                df.columns = (df.columns.astype(str)
                            .str.strip()
                            .str.replace(r'[^\w\s]', '_', regex=True)
                            .str.replace(r'\s+', '_', regex=True)
                            .str.lower())
                
                # Remove any duplicate column names
                df.columns = [f"{col}_{i}" if df.columns[:i].tolist().count(col) > 0 
                            else col for i, col in enumerate(df.columns)]
                
                # Create table
                df.to_sql(table_name, conn, if_exists='replace', index=False)
                
                # Get schema information for this table
                cursor.execute(f"PRAGMA table_info([{table_name}])")
                columns = cursor.fetchall()
                schema_info[table_name] = [
                    {'name': col[1], 'type': col[2]} for col in columns
                ]
                
                st.success(f"Successfully created table: {table_name}")
                
            except Exception as sheet_error:
                st.warning(f"Error processing sheet {sheet_name}: {str(sheet_error)}")
                continue
        
        conn.commit()
        conn.close()
        
        if not schema_info:
            raise Exception("No tables were successfully created")
            
        return db_name, schema_info
        
    except Exception as e:
        st.error(f"Error processing uploaded file: {str(e)}")
        if 'conn' in locals() and conn:
            conn.close()
        return None, None




def get_database_stats_from_db(db_name):
    """
    Get statistics from the specified database.
    
    Args:
        db_name: Name of the database file
        
    
    Returns:
        dict: Database statistics
    """
    try:
        conn = sqlite3.connect(db_name)
        # Connect to the database
        cursor = conn.cursor()
        
        # Get table count
        # Get list of all tables in the database
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [table[0] for table in cursor.fetchall()]
        
        # If no tables found, return None
        if not tables:
            st.warning("No tables found in the uploaded database.")
            return None
        
        # Create stats dictionary dynamically
        stats = {}
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            # Get row count for each table
            row_count = cursor.fetchone()[0]
            stats[table] = row_count
        
        conn.close()
        # Close the database connection
        return stats
        
    except Exception as e:
        print(f"Error getting database stats: {e}")
        return None
  


def get_db_connection():
    """
    Returns the shared database connection.
    If no connection exists, it creates a new one based on `st.session_state.current_db`.
    """
    if st.session_state.current_db_connection is None:
        if st.session_state.current_db is None:
            raise ValueError("No database is currently active.")
        st.session_state.current_db_connection = sqlite3.connect(st.session_state.current_db)
    return st.session_state.current_db_connection
def close_db_connection():
    """
    Closes the shared database connection if it exists.
    """
    if st.session_state.current_db_connection is not None:
        st.session_state.current_db_connection.close()
        st.session_state.current_db_connection = None

def execute_query_on_db(db_name, sql_query):
    """
    Executes the provided SQL query on the specified database.
    """
    try:
        # Connect to the database using the provided name
        conn = sqlite3.connect(db_name)
        results = pd.read_sql_query(sql_query, conn)
        conn.close()
        return results
    except sqlite3.Error as sql_error:
        st.error(f"SQL Error: {str(sql_error)}")
        return None
    except Exception as e:
        st.error(f"Error executing query: {str(e)}")
        return None

     
def execute_query(sql_query):
    """Execute SQL query and return results"""
    try:
        conn = sqlite3.connect('bollywood_analysis.db')
        
        # Print the query for debugging
        print("Executing SQL Query:", sql_query)
        
        # Add error handling for SQL syntax
        try:
            results = pd.read_sql_query(sql_query, conn)
            
            # Print result info for debugging
            print("Query Results Shape:", results.shape)
            print("Results Preview:", results.head())
            
            if results.empty:
                st.warning("Query executed successfully but returned no results.")
            
            return results
            
        except sqlite3.Error as sql_error:
            st.error(f"SQL Error: {str(sql_error)}")
            print("SQL Error Details:", sql_error)
            return None
            
        except pd.io.sql.DatabaseError as db_error:
            st.error(f"Database Error: {str(db_error)}")
            print("Database Error Details:", db_error)
            return None
            
    except Exception as e:
        st.error(f"Error executing query: {str(e)}")
        print("General Error Details:", e)
        return None
        
    finally:
        if 'conn' in locals():
            conn.close()


def should_display_graph(results):
    """Determine if the results should be displayed as a graph"""
    # Check if results have more than one row and contain numerical data
    if not results.empty and len(results) > 1 and results.select_dtypes(include=['number']).shape[1] > 0:
        return True
    return False

def display_graph(results):
    """Display a graph based on user selection"""
    # Add a selection box for chart type
    chart_type = st.selectbox(
        "Select Chart Type",
        ["Bar Chart", "Line Chart", "Pie Chart", "Scatter Plot", "Histogram"]
    )
    
    # Create figure with default style and size
    fig, ax = plt.subplots(figsize=(10, 6))  # Increased size for better visibility
    
    try:
        # Set basic style elements
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Plot based on the selected chart type
        if chart_type == "Bar Chart":
            if len(results.columns) >= 2:
                results.set_index(results.columns[0], inplace=True)
                results.plot(kind='bar', ax=ax)
                plt.xticks(rotation=45, ha='right')
            else:
                st.error("Bar chart requires at least two columns")
                return

        elif chart_type == "Line Chart":
            if len(results.columns) >= 2:
                results.set_index(results.columns[0], inplace=True)
                results.plot(kind='line', ax=ax, marker='o')
                plt.xticks(rotation=45, ha='right')
            else:
                st.error("Line chart requires at least two columns")
                return

        elif chart_type == "Pie Chart":
            if results.shape[1] >= 2:
                results.set_index(results.columns[0], inplace=True)
                results.iloc[:, 0].plot(kind='pie', ax=ax, autopct='%1.1f%%')
                plt.axis('equal')  # Equal aspect ratio ensures circular plot
            else:
                st.error("Pie chart requires at least two columns: one for labels and one for values.")
                return

        elif chart_type == "Scatter Plot":
            num_cols = results.select_dtypes(include=['number']).columns
            if len(num_cols) >= 2:
                x_col = st.selectbox("Select X-axis", num_cols)
                y_col = st.selectbox("Select Y-axis", num_cols)
                results.plot(kind='scatter', x=x_col, y=y_col, ax=ax)
                ax.grid(True)
            else:
                st.error("Scatter plot requires at least two numerical columns.")
                return

        elif chart_type == "Histogram":
            num_cols = results.select_dtypes(include=['number']).columns
            if len(num_cols) > 0:
                hist_col = st.selectbox("Select Column for Histogram", num_cols)
                results[hist_col].plot(kind='hist', ax=ax, bins=20, edgecolor='black')
                ax.grid(True)
            else:
                st.error("Histogram requires at least one numerical column.")
                return

        # Add labels and title
        plt.xlabel(results.index.name if results.index.name else 'Index')
        plt.ylabel('Value')
        plt.title(f'{chart_type} of Query Results')

        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Display the plot
        st.pyplot(fig)
        plt.close(fig)  # Clean up the figure

    except Exception as e:
        st.error(f"Error creating graph: {str(e)}")
        plt.close(fig)  # Clean up in case of error
   
def get_table_schema():
    """
    Retrieve comprehensive schema information for all tables in the current database.
    
    Returns:
        dict: Schema information including column details, primary keys, and foreign keys
              for each table in the database
    """
    try:
        # Get database name from session state
        db_name = st.session_state.current_db
        if not db_name:
            raise ValueError("No database selected.")
        
        # Connect to the database
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        
        # Dictionary to store schema information
        schema_info = {}
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [table[0] for table in cursor.fetchall()]
        
        if not tables:
            raise ValueError("No tables found in the database.")
        
        # Get schema details for each table
        for table in tables:
            # Get column information
            cursor.execute(f"PRAGMA table_info({table});")
            columns = cursor.fetchall()
            
            # Get foreign key information
            cursor.execute(f"PRAGMA foreign_key_list({table});")
            fks = cursor.fetchall()
            
            # Process columns with enhanced information
            table_columns = []
            for col in columns:
                column_info = {
                    'name': col[1],
                    'type': col[2],
                    'nullable': not col[3],  # NOT NULL constraint
                    'default_value': col[4],
                    'primary_key': bool(col[5])
                }
                
                # Check if this column is a foreign key
                for fk in fks:
                    if fk[3] == col[1]:
                        column_info['foreign_key'] = {
                            'references_table': fk[2],
                            'references_column': fk[4],
                            'on_delete': fk[5],
                            'on_update': fk[6]
                        }
                
                table_columns.append(column_info)
            
            schema_info[table] = table_columns
        
        conn.close()
        return schema_info
        
    except ValueError as ve:
        st.warning(str(ve))
        return None
    except Exception as e:
        st.error(f"Error retrieving table schema: {str(e)}")
        return None
   
def check_query_cache(question):
    """
    Check if the question exists in the query logs and return the cached SQL query if found.
    Returns None if not found in cache.
    """
    try:
        if os.path.exists('query_logs.xlsx'):
            # Add error handling for file reading
            try:
                logs_df = pd.read_excel('query_logs.xlsx', engine='openpyxl')
                if logs_df.empty:
                    return None
                    
                # Check if question exists in logs (case-insensitive)
                cached_query = logs_df[logs_df['question'].str.lower() == question.lower()]
                if not cached_query.empty:
                    # Return the most recent SQL query for this question
                    return cached_query.iloc[-1]['sql_query']
                    
            except Exception as file_error:
                print(f"Error reading query_logs.xlsx: {str(file_error)}")
                return None
                
        return None
    except Exception as e:
        print(f"Error checking query cache: {str(e)}")
        return None

def log_successful_query(question, sql_query, results=None):
    """
    Log successful queries to Excel with results summary.
    Includes deduplication logic and error handling.
    
    Args:
        question (str): The natural language question
        sql_query (str): The generated SQL query
        results (pd.DataFrame): Query results dataframe
    """
    log_file = 'query_logs.xlsx'
    timestamp_format = '%Y-%m-%d %H:%M:%S'
    
    try:
        # Calculate tokens
        input_tokens = count_tokens(question)
        output_tokens = count_tokens(sql_query)
        total_tokens = input_tokens + output_tokens
        
        # Create results summary if results are provided
        results_summary = None
        if isinstance(results, pd.DataFrame):
            results_summary = f"Rows: {len(results)}, Columns: {len(results.columns)}"
            
        # Create DataFrame for new log entry
        new_log = pd.DataFrame({
            'timestamp': [datetime.now().strftime(timestamp_format)],
            'question': [question],
            'sql_query': [sql_query],
            'input_tokens': [input_tokens],
            'output_tokens': [output_tokens],
            'total_tokens': [total_tokens],
            'results_summary': [results_summary]
        })
        
        # Load existing logs if file exists
        if os.path.exists(log_file):
            try:
                existing_logs = pd.read_excel(log_file, engine='openpyxl')
                
                # Check for duplicate: Same question and SQL query
                if not existing_logs.empty:
                    duplicate_entry = existing_logs[
                        (existing_logs['question'].str.lower() == question.lower()) &
                        (existing_logs['sql_query'] == sql_query)
                    ]
                    if not duplicate_entry.empty:
                        print("Duplicate query found, skipping logging")
                        return
                        
                # Append new log to existing logs
                updated_logs = pd.concat([existing_logs, new_log], ignore_index=True)
            except Exception as read_error:
                print(f"Error reading existing logs: {str(read_error)}")
                updated_logs = new_log
        else:
            updated_logs = new_log
            
        # Write combined logs to the Excel file
        try:
            with pd.ExcelWriter(log_file, engine='openpyxl', mode='w') as writer:
                updated_logs.to_excel(writer, index=False, sheet_name='Logs')
            print(f"Successfully logged query at {datetime.now()}")
        except Exception as write_error:
            print(f"Error writing to log file: {str(write_error)}")
            
    except Exception as e:
        print(f"Error logging query: {str(e)}")

def display_stats_sidebar(stats):
    if stats:
        st.sidebar.header("📊 Database Statistics")
        
        # Dynamically create columns based on the number of tables
        table_names = list(stats.keys())
        num_cols = min(len(table_names), 4)  # Limit to 4 columns
        
        # Create columns
        cols = st.sidebar.columns(num_cols)
        
        # Distribute table stats across columns
        for i, table_name in enumerate(table_names):
            col_index = i % num_cols
            with cols[col_index]:
                st.metric(f"🗃️ {table_name.title()}", stats[table_name])

def display_schema_sidebar(schema_info):
    """Display the database schema in a formatted way in the sidebar"""
    st.sidebar.header("📋 Database Schema")
    
    # Group tables by category
    table_groups = {
        "Core Tables": ["movies", "actors", "directors", "genres"],
        "Relationship Tables": ["movie_actors"],
        "Performance Data": ["movie_performance", "awards"]
    }
    
    for group_name, group_tables in table_groups.items():
        st.sidebar.subheader(f"📑 {group_name}")
        
        for table_name in group_tables:
            if table_name in schema_info:
                with st.sidebar.expander(f"🗃️ {table_name.title()}"):
                    # Create formatted table of columns
                    cols_data = []
                    for col in schema_info[table_name]:
                        # Icons for different column types
                        pk_indicator = "🔑 " if col['primary_key'] else ""
                        fk_indicator = "🔗 " if "FOREIGN KEY" in str(col.get('type', '')) else ""
                        
                        # Format the type information
                        type_info = col['type'].upper()
                        if "FOREIGN KEY" in type_info:
                            type_info = f"FK -> {type_info.split('-> ')[1]}"
                        
                        cols_data.append({
                            "Column": f"{pk_indicator}{fk_indicator}{col['name']}",
                            "Type": type_info
                        })
                    
                    # Display as a clean table
                    df = pd.DataFrame(cols_data)
                    st.table(df)
                    
                    # If it's a relationship table, add a note
                    if table_name == "movie_actors":
                        st.info("Links movies with their cast members")
                    elif table_name == "movie_performance":
                        st.info("Stores box office and rating data")
                    elif table_name == "awards":
                        st.info("Tracks movie awards and nominations")

  
def process_speech_to_text():
    """Function to handle speech input and convert it to text"""
    try:
        # Create audio recorder button
        audio_bytes = audio_recorder(
            text="🎤 Click to ask question",
            recording_color="#e74c3c",
            neutral_color="#3498db",
            icon_size="2x"
        )
        
        if audio_bytes:
            # Save audio bytes to temporary file
            with open("temp_audio.wav", "wb") as f:
                f.write(audio_bytes)
            
            # Initialize recognizer
            recognizer = sr.Recognizer()
            
            # Read the temporary audio file
            with sr.AudioFile("temp_audio.wav") as source:
                audio = recognizer.record(source)
                
            try:
                # Convert speech to text using Google's speech recognition
                text = recognizer.recognize_google(audio)
                return text
            except sr.UnknownValueError:
                st.error("Sorry, I couldn't understand the audio.")
                return None
            except sr.RequestError as e:
                st.error(f"Could not request results from speech recognition service: {e}")
                return None
            finally:
                # Clean up temporary file
                if os.path.exists("temp_audio.wav"):
                    os.remove("temp_audio.wav")
                    
    except Exception as e:
        st.error(f"Error processing speech: {str(e)}")
        return None 
    
 
def count_tokens(text):
    """Count tokens using tiktoken"""
    encoder = tiktoken.get_encoding("cl100k_base")  # or another encoding
    return len(encoder.encode(text))
 

def check_database():
    """Check if database exists and has data"""
    try:
        if not os.path.exists('bollywood_analysis.db'):
            st.error("Database file not found!")
            return False
            
        conn = sqlite3.connect('bollywood_analysis.db')
        cursor = conn.cursor()
        
        # Check if tables exist and have data
        tables = ['movies', 'actors', 'directors', 'genres', 'awards', 'movie_actors', 'movie_performance']
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"Table {table} has {count} records")
            if count == 0:
                st.warning(f"Table {table} is empty!")
                
        conn.close()
        return True
        
    except Exception as e:
        st.error(f"Database check failed: {str(e)}")
        return False
      
def main():
    # Initialize session state variables for database management
    if 'current_db' not in st.session_state:
        st.session_state.current_db = None  # Track the current database name
    if 'current_db_connection' not in st.session_state:
        st.session_state.current_db_connection = None  # Keep a single shared connection
    if 'schema_info' not in st.session_state:
        st.session_state.schema_info = None  # Store the schema information

    try:
        # Custom CSS for better styling
        st.markdown("""
           
        """, unsafe_allow_html=True)

        # Main header
        with st.container():
            st.markdown('<div class="main-header">', unsafe_allow_html=True)
            st.title("🖼️ Data Analysis App")
            st.subheader("Natural Language to SQL Query Converter")
            st.markdown('</div>', unsafe_allow_html=True)

        # File upload section
        st.markdown("### 📤 Upload Your Excel File")
        uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx', 'xls'])
        
        # Process uploaded file if present
        if uploaded_file is not None:
            with st.spinner("Processing Excel file..."):
                db_name, schema_info = create_database_from_excel(uploaded_file)
                if db_name and schema_info:
                    st.session_state.current_db = db_name
                    st.session_state.schema_info = schema_info
                    st.success(f"Database created successfully: {db_name}")
                else:
                    st.error("Failed to create database from uploaded Excel file.")
        # Initialize default database if no custom database is uploaded
        if not st.session_state.current_db:
            if not os.path.exists('bollywood_analysis.db'):
                with st.spinner("Initializing default database..."):
                    if not init_database():
                        st.error("Failed to initialize database. Please check your data files.")
                        return
            st.session_state.current_db = 'bollywood_analysis.db'

        # Setup LLM chain
        prompt_template, llm = setup_llm()
        if not prompt_template or not llm:
            st.error("Failed to initialize LLM. Please check your configuration.")
            return
            
        if not check_database():
            st.error("Please ensure the database is properly initialized with data.")
            return

        # Sidebar content
        with st.sidebar:
            # Get stats based on current database
            if st.session_state.current_db:
                stats = get_database_stats_from_db(st.session_state.current_db)
                if stats:
                    display_stats_sidebar(stats)
                
                # Display schema information
                if st.session_state.schema_info:
                    display_schema_sidebar(st.session_state.schema_info)
                else:
                    schema_info = get_table_schema()
                    if schema_info:
                        display_schema_sidebar(schema_info)

        # Main content layout
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown('<div class="query-section">', unsafe_allow_html=True)
            st.markdown("### 💭 Ask Questions About Your Data")

        examples = list(PREDEFINED_QUERIES.keys())
        selected_question = st.selectbox("📝 Choose from example questions:", [""] + examples, key="example_questions")

        text_tab, voice_tab = st.tabs(["💬 Text Input", "🎤 Voice Input"])
        with text_tab:
            user_question = st.text_input("🔍 Type your question:", value=selected_question, key="user_question", 
                                        placeholder="e.g., What are the top 5 highest-grossing movies?")

        with voice_tab:
            st.write("🎙️ Click the button below and speak your question:")
            spoken_text = process_speech_to_text()
            if spoken_text:
                user_question = spoken_text
                st.success(f"Recognized text: {spoken_text}")

        st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            if user_question:
                st.markdown("### 🤖 Query Processing")
                st.info("Processing your natural language query...")

        if user_question:
            with st.spinner("🔄 Analyzing your question..."):
                try:
                    input_tokens = count_tokens(user_question)
                    start_time = datetime.now()

                    # Get pruned schema for the question
                    pruned_schema = get_dynamic_schema_for_question(user_question)
                    if not pruned_schema:
                        st.error("Failed to analyze database schema.")
                        return

                    # Check cache first
                    cached_sql = check_query_cache(user_question)
                    if cached_sql:
                        sql_query = cached_sql
                        output_tokens = count_tokens(sql_query)
                        st.success("Retrieved from cache! ⚡")
                    else:
                        if user_question in PREDEFINED_QUERIES:
                            sql_query = PREDEFINED_QUERIES[user_question]
                            output_tokens = count_tokens(sql_query)
                        else:
                            response = llm.invoke(prompt_template.format(
                                question=user_question,
                                schema=pruned_schema
                            ))
                            sql_query = response.content if hasattr(response, 'content') else str(response)
                            output_tokens = count_tokens(sql_query)

                    processing_time = (datetime.now() - start_time).total_seconds()

                    # Display query and metrics
                    st.write("Generated SQL Query:")
                    st.code(sql_query, language="sql")
                    st.write(f"⏱️ Processing Time: {processing_time:.2f} seconds")
                    st.write(f"🔢 Input Tokens: {input_tokens}, Output Tokens: {output_tokens}")

                    # Execute query using current database
                    st.write("Executing query...")
                    results = execute_query_on_db(st.session_state.current_db, sql_query)
                    
                    if results is not None:
                        st.write(f"Query returned {len(results)} rows and {len(results.columns)} columns")
        
                        if not results.empty:
                            # Pass results to logging function
                            log_successful_query(user_question, sql_query, results)
                            st.markdown("### 📊 Query Results")
                            
                            # Display statistics
                            result_stats = st.columns(3)
                            with result_stats[0]:
                                st.metric("Rows", len(results))
                            with result_stats[1]:
                                st.metric("Columns", len(results.columns))
                            with result_stats[2]:
                                st.metric("Non-null Values", results.count().sum())

                            # Display results table
                            st.dataframe(results, use_container_width=True, hide_index=True)

                            # Display graph if applicable
                            if should_display_graph(results):
                                st.markdown("### 📊 Graphical Representation")
                                display_graph(results)

                            # Export options
                            st.markdown("### 📥 Export Results")
                            col1, col2 = st.columns(2)
                            with col1:
                                csv = results.to_csv(index=False)
                                st.download_button(
                                    "Download as CSV",
                                    data=csv,
                                    file_name="query_results.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                            with col2:
                                buffer = io.BytesIO()
                                with pd.ExcelWriter(buffer) as writer:
                                    results.to_excel(writer, index=False)
                                st.download_button(
                                    "Download as Excel",
                                    data=buffer.getvalue(),
                                    file_name="query_results.xlsx",
                                    mime="application/vnd.ms-excel",
                                    use_container_width=True
                                )
                        else:
                            st.warning("Query executed successfully but returned no results.")
                    else:
                        st.error("Query execution failed. Please check the error messages above.")

                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
                    print("Error details:", e)
                    st.markdown("Please try rephrasing your question or select from the example queries.")

        # Query History Section
        if os.path.exists('query_logs.xlsx'):
            st.markdown("### 📚 Query History")
            with st.expander("View Previous Queries"):
                logs_df = pd.read_excel('query_logs.xlsx', engine='openpyxl')
                st.dataframe(logs_df, use_container_width=True, hide_index=True)
            with open('query_logs.xlsx', 'rb') as f:
                st.download_button(
                    "Download Query History",
                    data=f.read(),
                    file_name="query_history.xlsx",
                    mime="application/vnd.ms-excel",
                    use_container_width=True
                )

        st.markdown("---")
        st.markdown(
            "<div style='text-align: center; color: #666;'>🎬 Powered by Gemini and SQLite | ❤️ Sudipta</div>",
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        print(f"Unexpected error details: {e}")


if __name__ == "__main__":
    main()

