
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from datetime import datetime, timedelta

from ingestion.data_cleaner import clean_data
from ingestion.mt5_download import download_data
from ml_models.logisticRegression import train_logistic_regression
from ml_models.linear_regression import train_linear_regression
from ml_models.RandomForest import train_random_forest 

def page1():
    st.write(st.session_state.foo)

def page2():
    st.write(st.session_state.bar)

# Widgets shared by all the pages
st.sidebar.selectbox("Foo", ["linear_regression", "logistic_regression", "random_forest"], key="foo")
st.sidebar.checkbox("Bar", key="bar")

pg = st.navigation([page1, page2])
pg.run()