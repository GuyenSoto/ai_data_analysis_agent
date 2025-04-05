import json
import tempfile
import csv
import streamlit as st
import pandas as pd
import openai
import re
import duckdb

# Function to preprocess and save the uploaded file
def preprocess_and_save(file):
    try:
        # Read the uploaded file into a DataFrame
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, encoding='utf-8', na_values=['NA', 'N/A', 'missing'])
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file, na_values=['NA', 'N/A', 'missing'])
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None, None, None
        
        # Ensure string columns are properly quoted
        for col in df.select_dtypes(include=['object']):
            df[col] = df[col].astype(str).replace({r'"': '""'}, regex=True)
        
        # Parse dates and numeric columns
        for col in df.columns:
            if 'date' in col.lower():
                df[col] = pd.to_datetime(df[col], errors='coerce')
            elif df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    # Keep as is if conversion fails
                    pass
        
        # Create a temporary file to save the preprocessed data
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            temp_path = temp_file.name
            # Save the DataFrame to the temporary CSV file with quotes around string fields
            df.to_csv(temp_path, index=False, quoting=csv.QUOTE_ALL)
        
        return temp_path, df.columns.tolist(), df  # Return the DataFrame as well
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None, None, None

# Function to generate SQL query using OpenAI API directly
def generate_sql_query(api_key, prompt, semantic_model_json):
    """
    Función para generar una consulta SQL usando la API de OpenAI directamente
    """
    # Configurar la API key
    openai.api_key = api_key
    
    try:
        # Crear la solicitud a la API de OpenAI
        response = openai.chat.completions.create(
            model="gpt-4",  # Puedes cambiar a gpt-3.5-turbo si prefieres
            messages=[
                {"role": "system", "content": "You are an expert data analyst. Generate SQL queries to solve the user's query. Return only the SQL query, enclosed in ```sql ``` and give the final answer."},
                {"role": "user", "content": f"""I have a dataset with the following schema:
{semantic_model_json}

Please generate a SQL query to answer the following question:
{prompt}

Return only the SQL query enclosed in ```sql ``` tags."""}
            ],
            temperature=0.2  # Baja temperatura para respuestas consistentes
        )
        
        # Extraer el contenido de la respuesta
        return response.choices[0].message.content
        
    except Exception as e:
        st.error(f"Error al generar la consulta SQL: {e}")
        return f"No se pudo generar una consulta SQL: {str(e)}"

# Function to execute SQL query
def execute_sql_query(sql_query, db_path):
    """
    Función para ejecutar una consulta SQL con DuckDB
    """
    try:
        # Conectar a la base de datos
        conn = duckdb.connect(database=':memory:')
        
        # Cargar los datos
        conn.execute(f"CREATE TABLE uploaded_data AS SELECT * FROM '{db_path}'")
        
        # Ejecutar la consulta
        result = conn.execute(sql_query).fetchdf()
        
        return result
    except Exception as e:
        st.error(f"Error ejecutando la consulta SQL: {e}")
        return None

# Streamlit app
st.title("📊 Data Analyst Agent")

# Sidebar for API keys
with st.sidebar:
    st.header("API Keys")
    openai_key = st.text_input("Enter your OpenAI API key:", type="password")
    if openai_key:
        st.session_state.openai_key = openai_key
        st.success("API key saved!")
    else:
        st.warning("Please enter your OpenAI API key to proceed.")

# File upload widget
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None and "openai_key" in st.session_state:
    # Preprocess and save the uploaded file
    temp_path, columns, df = preprocess_and_save(uploaded_file)
    
    if temp_path and columns and df is not None:
        # Display the uploaded data as a table
        st.write("Uploaded Data:")
        st.dataframe(df)  # Use st.dataframe for an interactive table
        
        # Display the columns of the uploaded data
        st.write("Uploaded columns:", columns)
        
        # Configure the semantic model with the temporary file path
        semantic_model = {
            "tables": [
                {
                    "name": "uploaded_data",
                    "description": "Contains the uploaded dataset.",
                    "path": temp_path,
                    "columns": [{"name": col, "description": f"Column containing {col} data"} for col in columns]
                }
            ]
        }
        
        # Main query input widget
        user_query = st.text_area("Ask a query about the data:")
        
        if st.button("Submit Query"):
            if user_query.strip() == "":
                st.warning("Please enter a query.")
            else:
                try:
                    # Show loading spinner while processing
                    with st.spinner('Processing your query...'):
                        # Generar la consulta SQL utilizando OpenAI directamente
                        response = generate_sql_query(
                            st.session_state.openai_key,
                            user_query,
                            json.dumps(semantic_model)
                        )
                    
                    # Extraer la consulta SQL de la respuesta
                    sql_pattern = r"```sql\s*(.*?)\s*```"
                    sql_match = re.search(sql_pattern, response, re.DOTALL)
                    
                    if sql_match:
                        sql_query = sql_match.group(1).strip()
                        
                        # Mostrar la consulta SQL
                        st.subheader("Generated SQL Query:")
                        st.code(sql_query, language="sql")
                        
                        # Ejecutar la consulta SQL
                        result = execute_sql_query(sql_query, temp_path)
                        
                        if result is not None:
                            # Mostrar los resultados
                            st.subheader("Query Results:")
                            st.dataframe(result)
                            
                            # Ofrecer descarga de resultados
                            csv = result.to_csv(index=False)
                            st.download_button(
                                label="Download results as CSV",
                                data=csv,
                                file_name="query_results.csv",
                                mime="text/csv"
                            )
                    else:
                        # Mostrar la respuesta completa si no se pudo extraer la consulta SQL
                        st.markdown(response)
                    
                except Exception as e:
                    st.error(f"Error during query processing: {e}")
                    st.error("Please try rephrasing your query or check if the data format is correct.")