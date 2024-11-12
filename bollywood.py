import sqlite3
import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_ollama import ChatOllama
import io
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import speech_recognition as sr
from audio_recorder_streamlit import audio_recorder

# Configure Streamlit page
st.set_page_config(
    page_title="Bollywood Movie Analysis",
    page_icon="🎬",
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
            'actors.csv', 'directors.csv', 'genres.csv', 
            'movies.csv', 'awards.csv', 'movie_actors.csv', 
            'movie_performance.csv'
        ]
        
        for file in required_files:
            if not os.path.exists(file):
                st.error(f"Missing required file: {file}")
                return False
        
        # Read CSV files with error handling
        try:
            actors_df = pd.read_csv('actors.csv')
            directors_df = pd.read_csv('directors.csv')
            genres_df = pd.read_csv('genres.csv')
            movies_df = pd.read_csv('movies.csv')
            awards_df = pd.read_csv('awards.csv')
            movie_actors_df = pd.read_csv('movie_actors.csv')
            movie_performance_df = pd.read_csv('movie_performance.csv')
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
        SELECT m.title, m.box_office_millions
        FROM movies m
        ORDER BY m.box_office_millions DESC
        LIMIT 5;
    """,
    "Show me movies directed by Sanjay Leela Bhansali": """
        SELECT m.title, m.release_year
        FROM movies m
        JOIN directors d ON m.director_id = d.director_id
        WHERE d.name = 'Sanjay Leela Bhansali'
        ORDER BY m.release_year;
    """,
    "What is the average IMDb rating by genre?": """
        SELECT g.genre_name, ROUND(AVG(m.imdb_rating), 2) as avg_rating
        FROM movies m
        JOIN genres g ON m.genre_id = g.genre_id
        GROUP BY g.genre_name
        ORDER BY avg_rating DESC;
    """
}


def get_dynamic_schema():
    """Extract schema information from SQLite database and format it for the prompt"""
    try:
        conn = sqlite3.connect('bollywood_analysis.db')
        cursor = conn.cursor()
        
        # Get list of all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        schema_info = []
        for table in tables:
            table_name = table[0]
            # Get column information for each table
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            
            # Format column information
            column_info = []
            for col in columns:
                name = col[1]
                data_type = col[2]
                is_pk = " (PRIMARY KEY)" if col[5] == 1 else ""
                
                # Check if column is a foreign key
                cursor.execute(f"PRAGMA foreign_key_list({table_name});")
                fks = cursor.fetchall()
                is_fk = ""
                for fk in fks:
                    if fk[3] == name:
                        is_fk = f" (FOREIGN KEY -> {fk[2]}.{fk[4]})"
                        break
                
                column_info.append(f"   - {name}{is_pk}{is_fk}")
            
            # Add formatted table schema to list
            schema_info.append(f"{len(schema_info) + 1}. {table_name}\n" + "\n".join(column_info))
        
        conn.close()
        return "\n\n".join(schema_info)
        
    except Exception as e:
        st.error(f"Error getting schema info: {str(e)}")
        return None


# Modified prompt template that uses dynamic schema
def get_prompt_template():
    """Get the prompt template with dynamic schema"""
    schema = get_dynamic_schema()
    template = f"""
You are an expert SQL query generator for a Bollywood movie database. Given a natural language question, generate the appropriate SQL query based on the following schema:

DATABASE SCHEMA:
{schema}

IMPORTANT RULES FOR SQL QUERY GENERATION:
1. Return ONLY the SQL query without explanations or comments.
2. Use appropriate JOIN clauses for combining tables.
3. If tables have columns with the same name, use aliases for each table (e.g., 'a' for actors, 'd' for directors).
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

User Question: {{question}}

Generate the SQL query that answers this question:
"""
    return PromptTemplate(
        input_variables=["question"],
        template=template
    )

def setup_llm():
    """Initialize the LLM and chain with dynamic schema"""
    try:
        llm = ChatOllama(temperature=0, model="llama3.2")
        prompt = get_prompt_template()
        return LLMChain(prompt=prompt, llm=llm)
    except Exception as e:
        st.error(f"Error setting up LLM: {str(e)}")
        return None

def get_database_stats():
    """Get database statistics for the sidebar"""
    try:
        conn = sqlite3.connect('bollywood_analysis.db')
        stats = {}
        
        # Updated list to include ALL tables
        tables = [
            'movies', 
            'actors', 
            'directors', 
            'genres',
            'awards',
            'movie_actors',     # Junction table between movies and actors
            'movie_performance' # Performance metrics table
        ]
        
        for table in tables:
            query = f"SELECT COUNT(*) as count FROM {table}"
            result = pd.read_sql(query, conn)
            stats[table] = result.iloc[0]['count']
            
        conn.close()
        return stats
    except Exception as e:
        st.error(f"Error getting database stats: {str(e)}")
        return None

def execute_query(sql_query):
    """Execute SQL query and return results"""
    try:
        conn = sqlite3.connect('bollywood_analysis.db')
        results = pd.read_sql_query(sql_query, conn)
        conn.close()
        return results
    except Exception as e:
        st.error(f"Error executing query: {str(e)}")
        return None

def should_display_graph(results):
    """Determine if the results should be displayed as a graph"""
    # Example criteria: Check if results have numerical data
    if not results.empty and results.select_dtypes(include=['number']).shape[1] > 0:
        return True
    return False

def display_graph(results):
    """Display a graph based on user selection"""
    # Add a selection box for chart type
    chart_type = st.selectbox(
        "Select Chart Type",
        ["Bar Chart", "Line Chart", "Pie Chart", "Scatter Plot", "Histogram"]
    )

    # Set a smaller figure size
    fig, ax = plt.subplots(figsize=(8, 4))  # Adjust the size as needed

    # Plot based on the selected chart type
    if chart_type == "Bar Chart":
        results.plot(kind='bar', ax=ax)
    elif chart_type == "Line Chart":
        results.plot(kind='line', ax=ax)
    elif chart_type == "Pie Chart":
        # For pie chart, we need to select a single column to plot
        if results.shape[1] >= 2:
            results.set_index(results.columns[0], inplace=True)
            results.iloc[:, 0].plot(kind='pie', ax=ax, autopct='%1.1f%%')
        else:
            st.error("Pie chart requires at least two columns: one for labels and one for values.")
    elif chart_type == "Scatter Plot":
        # For scatter plot, we need at least two numerical columns
        if results.select_dtypes(include=['number']).shape[1] >= 2:
            x_col = st.selectbox("Select X-axis", results.select_dtypes(include=['number']).columns)
            y_col = st.selectbox("Select Y-axis", results.select_dtypes(include=['number']).columns)
            results.plot(kind='scatter', x=x_col, y=y_col, ax=ax)
        else:
            st.error("Scatter plot requires at least two numerical columns.")
    elif chart_type == "Histogram":
        # For histogram, select a single numerical column
        num_cols = results.select_dtypes(include=['number']).columns
        if len(num_cols) > 0:
            hist_col = st.selectbox("Select Column for Histogram", num_cols)
            results[hist_col].plot(kind='hist', ax=ax, bins=10)
        else:
            st.error("Histogram requires at least one numerical column.")

    # Display the plot
    st.pyplot(fig)


def get_table_schema():
    """Get schema information for all tables"""
    try:
        conn = sqlite3.connect('bollywood_analysis.db')
        cursor = conn.cursor()
        
        # Dictionary to store schema for each table
        schema_info = {}
        
        # Get schema for each table
        tables = [
            'actors', 'directors', 'genres', 'movies', 
            'movie_actors', 'awards', 'movie_performance'
        ]
        
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table});")
            columns = cursor.fetchall()
            schema_info[table] = [
                {
                    'name': col[1],
                    'type': col[2],
                    'primary_key': bool(col[5])
                }
                for col in columns
            ]
            
        conn.close()
        return schema_info
    except Exception as e:
        st.error(f"Error getting schema info: {str(e)}")
        return None
    
    ##################


def check_query_cache(question):
    """
    Check if the question exists in the query logs and return the cached SQL query if found.
    Returns None if not found in cache.
    """
    try:
        if os.path.exists('query_logs.xlsx'):
            logs_df = pd.read_excel('query_logs.xlsx', engine='openpyxl')
            # Check if question exists in logs
            cached_query = logs_df[logs_df['question'].str.lower() == question.lower()]
            if not cached_query.empty:
                # Return the most recent SQL query for this question
                return cached_query.iloc[-1]['sql_query']
        return None
    except Exception as e:
        print(f"Error checking query cache: {str(e)}")
        return None

def log_successful_query(question, sql_query):
    """
    Log only successful queries (queries that produced results) to Excel.
    Includes deduplication logic.
    """
    log_file = 'query_logs.xlsx'
    timestamp_format = '%Y-%m-%d %H:%M:%S'
    
    # Create DataFrame for new log entry
    new_log = pd.DataFrame({
        'timestamp': [datetime.now().strftime(timestamp_format)],
        'question': [question],
        'sql_query': [sql_query]
    })

    try:
        # Load existing logs if file exists
        if os.path.exists(log_file):
            existing_logs = pd.read_excel(log_file, engine='openpyxl')
            
            # Check for duplicate: Same question and SQL query
            if not existing_logs.empty:
                duplicate_entry = existing_logs[
                    (existing_logs['question'].str.lower() == question.lower()) &
                    (existing_logs['sql_query'] == sql_query)
                ]
                if not duplicate_entry.empty:
                    return  # Skip logging if duplicate found

            # Append new log to existing logs
            updated_logs = pd.concat([existing_logs, new_log], ignore_index=True)
        else:
            # If no file, start fresh
            updated_logs = new_log

        # Write combined logs to the Excel file
        with pd.ExcelWriter(log_file, engine='openpyxl', mode='w') as writer:
            updated_logs.to_excel(writer, index=False, sheet_name='Logs')
    
    except Exception as e:
        print(f"Error logging query: {str(e)}")

def display_stats_sidebar(stats):
    if stats:
        st.sidebar.header("📊 Database Statistics")
        
        # Main entities
        st.sidebar.subheader("Main Entities")
        cols1 = st.sidebar.columns(2)
        with cols1[0]:
            st.metric("🎥 Movies", stats['movies'])
            st.metric("🎭 Actors", stats['actors'])
        with cols1[1]:
            st.metric("🎬 Directors", stats['directors'])
            st.metric("📽️ Genres", stats['genres'])
            
        # Related data
        st.sidebar.subheader("Related Data")
        cols2 = st.sidebar.columns(2)
        with cols2[0]:
            st.metric("🏆 Awards", stats['awards'])
            st.metric("🤝 Movie-Actor Links", stats['movie_actors'])
        with cols2[1]:
            st.metric("📈 Performance Records", stats['movie_performance'])


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


def get_table_schema():
    """Get enhanced schema information for all tables"""
    try:
        conn = sqlite3.connect('bollywood_analysis.db')
        cursor = conn.cursor()
        
        # Dictionary to store schema for each table
        schema_info = {}
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        for table in tables:
            table_name = table[0]
            
            # Get column information
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            
            # Get foreign key information
            cursor.execute(f"PRAGMA foreign_key_list({table_name});")
            fks = cursor.fetchall()
            
            # Process columns with enhanced information
            table_columns = []
            for col in columns:
                column_info = {
                    'name': col[1],
                    'type': col[2],
                    'primary_key': bool(col[5])
                }
                
                # Check if this column is a foreign key
                for fk in fks:
                    if fk[3] == col[1]:  # if column name matches FK column
                        column_info['type'] = f"FOREIGN KEY -> {fk[2]}.{fk[4]}"
                        break
                
                table_columns.append(column_info)
            
            schema_info[table_name] = table_columns
        
        conn.close()
        return schema_info
        
    except Exception as e:
        st.error(f"Error getting schema info: {str(e)}")
        return None
    
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
    
def clear_text_input():
    """Clear the text input box"""
    st.session_state.user_question = ""
    st.session_state.example_questions = "" 
    
        
def main():
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main-header {
            text-align: center;
            padding: 1rem;
            background-color: #060632;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .stButton>button {
            width: 100%;
        }
        .query-section {
            background-color: #060632;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        </style>
    """, unsafe_allow_html=True)

    # Main header
    with st.container():
        st.markdown('<div class="main-header">', unsafe_allow_html=True)
        st.title("🎬 Bollywood Movie Analysis")
        st.subheader("Natural Language to SQL Query Converter")
        st.markdown('</div>', unsafe_allow_html=True)

    # Initialize database if needed
    if not os.path.exists('bollywood_analysis.db'):
        with st.spinner("Initializing database..."):
            if not init_database():
                st.error("Failed to initialize database. Please check your data files.")
                return

    # Setup LLM chain
    nl_to_sql_chain = setup_llm()
    if not nl_to_sql_chain:
        st.error("Failed to initialize LLM chain. Please check your configuration.")
        return

    # Enhanced sidebar with both statistics and schema
    with st.sidebar:
        # Statistics Section
        # In your main() function:
        stats = get_database_stats()
        if stats:
            display_stats_sidebar(stats)
        # Schema Section
        schema_info = get_table_schema()
        if schema_info:
            display_schema_sidebar(schema_info)

    # Main query interface with enhanced layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="query-section">', unsafe_allow_html=True)
        st.markdown("### 💭 Ask Questions About Bollywood Movies")
    
    # Example questions dropdown with enhanced styling
    examples = list(PREDEFINED_QUERIES.keys())
    selected_question = st.selectbox(
        "📝 Choose from example questions:",
        [""] + examples,
        key="example_questions"
    )
    
    # Add tabs for text/voice input
    text_tab, voice_tab = st.tabs(["💬 Text Input", "🎤 Voice Input"])
    
    with text_tab:
        # Text input field
        user_question = st.text_input(
            "🔍 Type your question:",
            value=selected_question,
            key="user_question",
            placeholder="e.g., What are the top 5 highest-grossing movies?"
        )
    
    
    
    with voice_tab:
        # Voice input using custom function
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

    # Query processing and results display
    # Query processing and results display
    if user_question:
        with st.spinner("🔄 Analyzing your question..."):
            try:
                # First check if query exists in cache
                cached_sql = check_query_cache(user_question)
                
                if cached_sql:
                    sql_query = cached_sql
                    st.success("Retrieved from cache! ⚡")
                else:
                    # Get SQL query
                    if user_question in PREDEFINED_QUERIES:
                        sql_query = PREDEFINED_QUERIES[user_question]
                    else:
                        response = nl_to_sql_chain.invoke({"question": user_question})
                        sql_query = response['text']

                # Show SQL query in a cleaner expander
                with st.expander("🔍 Generated SQL Query", expanded=True):
                    st.code(sql_query, language="sql")

                # Execute query and show results
                results = execute_query(sql_query)
                
                if results is not None and not results.empty:
                    # Only log if query was successful and returned results
                    log_successful_query(user_question, sql_query)
                    
                    # Show result statistics
                    st.markdown("### 📊 Query Results")
                    result_stats = st.columns(3)
                    with result_stats[0]:
                        st.metric("Rows", len(results))
                    with result_stats[1]:
                        st.metric("Columns", len(results.columns))
                    with result_stats[2]:
                        st.metric("Non-null Values", results.count().sum())

                    # Display results in a scrollable container
                    st.dataframe(
                        results,
                        use_container_width=True,
                        hide_index=True
                    )

                    # Check if we should display a graph
                    if should_display_graph(results):
                        st.markdown("### Graphical Representation")
                        display_graph(results)

                    # Download section
                    st.markdown("### 📥 Export Results")
                    col1, col2 = st.columns(2)
                    with col1:
                        csv = results.to_csv(index=False)
                        st.download_button(
                            label="Download as CSV",
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
                            label="Download as Excel",
                            data=buffer.getvalue(),
                            file_name="query_results.xlsx",
                            mime="application/vnd.ms-excel",
                            use_container_width=True
                        )
                else:
                    st.error("Query returned no results. Please try rephrasing your question.")

            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
                st.markdown("Please try rephrasing your question or select from the example queries.")
    
    ######end############
    # And modify the history viewing section in main():
    if os.path.exists('query_logs.xlsx'):
        st.markdown("### 📚 Query History")
        with st.expander("View Previous Queries"):
            logs_df = pd.read_excel('query_logs.xlsx', engine='openpyxl')
            st.dataframe(
                logs_df,
                use_container_width=True,
                hide_index=True
            )
        
        # Download logs button
        with open('query_logs.xlsx', 'rb') as f:
            st.download_button(
                label="Download Query History",
                data=f.read(),
                file_name="query_history.xlsx",
                mime="application/vnd.ms-excel",
                use_container_width=True
            )     

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        🎬 Powered by llama3.2 and SQLite | ❤️ Sudipta
        </div>
        """, 
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
