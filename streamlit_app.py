import streamlit as st
import pandas as pd

st.title("SHL Product Recommendation App")

# Load dataset
try:
    df = pd.read_csv("shl_assessments_clean.csv")
    st.success("Dataset loaded successfully!")
except:
    st.error("Cannot load dataset")
    df = None

# Show dataset preview
if df is not None and st.checkbox("Show dataset preview"):
    st.write(df.head(10))

# User query
query = st.text_input("Enter your query:")

# Show results
if query and df is not None:
    # Search in all text columns
    string_columns = df.select_dtypes(include='object').columns.tolist()
    results = pd.DataFrame()
    for col in string_columns:
        results = pd.concat([results, df[df[col].str.contains(query, case=False, na=False)]])
    results = results.drop_duplicates()
    
    if results.empty:
        st.write("No recommendations found.")
    else:
        st.write("### Recommended Product(s):")
        st.write(results.head(5))