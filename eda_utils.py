import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np

@st.cache_data
def load_data(file):
    """
    Loads data from a CSV file (uploaded or path) into a DataFrame.
    """
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def get_quick_stats(df):
    """
    Returns a dictionary of quick statistics for the dashboard.
    """
    return {
        "rows": df.shape[0],
        "cols": df.shape[1],
        "missing_pct": round(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100, 2),
        "duplicates": df.duplicated().sum()
    }

def detect_column_types(df):
    """
    Separates columns into numerical and categorical/object types.
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    return numerical_cols, categorical_cols

@st.cache_data
def plot_distributions(df, column):
    """
    Generates a distribution plot based on column type using Plotly Express.
    """
    try:
        if pd.api.types.is_numeric_dtype(df[column]):
            # Histogram with a box plot overlay for numerical data
            fig = px.histogram(
                df, 
                x=column, 
                marginal="box", 
                title=f"Distribution of {column}",
                template="plotly_white",
                color_discrete_sequence=["#636EFA"]
            )
        else:
            # Bar chart (Histogram of counts) for categorical data
            # We limit to top 20 mostly for performance/visual clarity
            top_n = df[column].value_counts().nlargest(20).index
            filtered_df = df[df[column].isin(top_n)]
            
            fig = px.histogram(
                filtered_df, 
                x=column, 
                title=f"Count of {column} (Top 20)",
                template="plotly_white",
                color_discrete_sequence=["#EF553B"]
            )
            # Update x-axis to sort by total descending
            fig.update_layout(xaxis={'categoryorder':'total descending'})
        
        fig.update_layout(autosize=True)
        return fig
    except Exception as e:
        st.error(f"Could not plot distribution: {e}")
        return None

def plot_correlation(df):
    """
    Generates a correlation heatmap for numerical columns.
    """
    try:
        numerical_cols = df.select_dtypes(include=[np.number])
        
        if len(numerical_cols.columns) > 1:
            corr = numerical_cols.corr()
            fig = px.imshow(
                corr,
                text_auto=True,
                aspect="auto",
                title="Correlation Heatmap (Numerical Columns)",
                color_continuous_scale="RdBu_r",
                origin='lower'
            )
            fig.update_layout(template="plotly_white")
            return fig
        else:
            return None
    except Exception as e:
        st.error(f"Could not plot correlation: {e}")
        return None
