import streamlit as st
import google.generativeai as genai
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
import logging

# Suppress unnecessary warnings
logging.getLogger('google.generativeai').setLevel(logging.ERROR)

# Configure Gemini API using Streamlit secrets
genai.configure(api_key=st.secrets["general"]["GEN_KEY"])

# Use the model specified in the example
model = genai.GenerativeModel("gemini-2.5-flash")

# Load the CSV files into DataFrames
df_employees = pd.read_csv('employees.csv')
df_departments = pd.read_csv('departments.csv')
df_sales = pd.read_csv('sales.csv')

# Function to get schema-like description from DataFrames
def get_schema():
    schema = ""
    # Employees table
    schema += "Table: employees\n"
    for col, dtype in df_employees.dtypes.items():
        schema += f"  Column: {col} type {dtype}\n"
    # Departments table
    schema += "Table: departments\n"
    for col, dtype in df_departments.dtypes.items():
        schema += f"  Column: {col} type {dtype}\n"
    # Sales table
    schema += "Table: sales\n"
    for col, dtype in df_sales.dtypes.items():
        schema += f"  Column: {col} type {dtype}\n"
    return schema

# Generate Pandas query code using Gemini
def generate_pandas_code(question, schema):
    prompt = f"""You are an expert Pandas code generator for poorly designed datasets with dirty data, bad schema, unnamed-like columns, and vague questions.
    
Schema:
{schema}

Notes on schema (inferred meanings - but use only what's in schema for querying):
- employees: (c1: id, c2: name, c3: dept_id, c4: salary - may be numbers as text or like '60k' or empty)
- departments: (d1: id, d2: name - may have duplicates or inconsistencies)
- sales: (s1: id, s2: emp_id, s3: amount - text like '10000' or '9k', s4: date - mixed formats like '2023-01-01' or '01/02/2023' or 'Jan 3, 2023')

Handle dirty data in Pandas: use .apply() or lambda to clean, e.g., df['c4'].apply(lambda x: float(str(x).replace('k', '000').replace(' thousand ', '000').strip()) if pd.notnull(x) else None) for salaries/amounts.
Handle dates with pd.to_datetime(errors='coerce').
For aggregates, clean data inline. Use merges for joins.

Available DataFrames: df_employees, df_departments, df_sales

Question: {question}

Output ONLY the Python Pandas code snippet that computes the result and assigns it to a DataFrame named 'result_df'. No explanations or imports."""
    response = model.generate_content(prompt)
    code = response.text.strip()
    if code.startswith("```python"):
        code = code.split("\n", 1)[1]
    if code.endswith("```"):
        code = code.rsplit("\n", 1)[0]
    return code.strip()

# Execute Pandas code
def execute_pandas(code):
    try:
        local_env = {'df_employees': df_employees, 'df_departments': df_departments, 'df_sales': df_sales, 'pd': pd}
        exec(code, globals(), local_env)
        result_df = local_env.get('result_df')
        if result_df is None:
            raise ValueError("result_df not defined in the code.")
        return result_df, None
    except Exception as e:
        return None, str(e)

# Fix failed Pandas code using Gemini
def fix_pandas_code(question, schema, bad_code, error):
    prompt = f"""The Pandas code failed: {bad_code}
Error: {error}

Schema: {schema}

Question: {question}

Fix the Python Pandas code, handling dirty data as needed.
Output ONLY the corrected code snippet that assigns to 'result_df'."""
    response = model.generate_content(prompt)
    code = response.text.strip()
    if code.startswith("```python"):
        code = code.split("\n", 1)[1]
    if code.endswith("```"):
        code = code.rsplit("\n", 1)[0]
    return code.strip()

# Generate natural language answer and visualization suggestion
def generate_response(question, df):
    csv_data = df.to_csv(index=False)
    prompt = f"""Question: {question}

Results (CSV):
{csv_data}

Output in this exact format:
Answer: [Natural language explanation of the results, insightful and business-oriented. Include a markdown table if relevant.]

Visualization: [none OR a brief description like 'bar chart of departments by average salary' if a chart would be helpful.]"""
    response = model.generate_content(prompt)
    return response.text

# Parse the response into answer and visualization
def parse_response(text):
    answer = ""
    vis = "none"
    if "Answer:" in text:
        parts = text.split("Answer:", 1)
        if len(parts) > 1:
            answer_part = parts[1].split("Visualization:", 1)[0].strip()
            answer = answer_part
    if "Visualization:" in text:
        parts = text.split("Visualization:", 1)
        if len(parts) > 1:
            vis = parts[1].strip()
    return answer, vis

# Generate matplotlib code for chart
def generate_chart_code(vis, df):
    csv_data = df.to_csv(index=False)
    prompt = f"""Generate a COMPLETE Python script to create the visualization: {vis}

Hardcode the data like this:
from io import StringIO
import pandas as pd
import matplotlib.pyplot as plt

data = '''{csv_data}'''
df = pd.read_csv(StringIO(data))

Then, plot the {vis} using df.
Save the figure to 'chart.png' with plt.savefig('chart.png', bbox_inches='tight')
Close with plt.close()

Output ONLY the code. No explanations."""
    response = model.generate_content(prompt)
    code = response.text.strip()
    if code.startswith("```python"):
        code = code.split("\n", 1)[1]
    if code.endswith("```"):
        code = code.rsplit("\n", 1)[0]
    return code.strip()

# Streamlit app
st.title("Simple Conversational AI Data Agent")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "table" in message:
            st.dataframe(message["table"])
        if "chart" in message:
            st.image(message["chart"])

# Get user input
question = st.chat_input("Ask complex business questions about employees, departments, and sales. Example: 'What is the average salary per department?' or 'Top 10 highest sales in the month of August'")

if question:
    # Append user message to history
    st.session_state.messages.append({"role": "user", "content": question})
    # Display user message
    with st.chat_message("user"):
        st.markdown(question)

    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()  # For streaming-like updates if needed
        schema = get_schema()
        code = generate_pandas_code(question, schema)
        df, error = execute_pandas(code)
        attempts = 1
        while error and attempts < 3:
            st.info(f"Pandas error: {error}. Attempting to fix...")
            code = fix_pandas_code(question, schema, code, error)
            df, error = execute_pandas(code)
            attempts += 1

        if error:
            answer = "Failed to generate a valid query after attempts."
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            # Display raw table if not empty
            if not df.empty:
                st.markdown("**Raw Results Table:**")
                st.dataframe(df)

            response_text = generate_response(question, df)
            answer, vis = parse_response(response_text)
            st.markdown("**Answer:**")
            st.markdown(answer)

            message = {"role": "assistant", "content": answer}
            if not df.empty:
                message["table"] = df

            if vis != "none":
                st.markdown(f"**Visualization:** {vis}")
                code = generate_chart_code(vis, df)
                try:
                    exec(code)
                    st.image('chart.png')
                    message["chart"] = 'chart.png'
                except Exception as e:
                    st.error(f"Chart generation failed: {e}")

            # Append assistant message to history
            st.session_state.messages.append(message)