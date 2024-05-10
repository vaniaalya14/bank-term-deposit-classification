import streamlit as st
import pandas as pd
import pickle as pkl
from PIL import Image
import matplotlib.pyplot as plt

def run():
    st.title('Exploratory Data Analysis Term Deposit Subscription')

    # Menampilkan gambar
    title_pict = Image.open('tree-term-depo.jpeg')
    st.image(title_pict, caption='Term Deposit',use_column_width='always')

    # Menampilkan table data
    @st.cache_data
    def fetch_data():
        df = pd.read_csv('P1M2_vania_alya.csv', sep = ';')
        return df

    df = fetch_data()
    st.header('Term Deposit Subscription Data')
    st.write(df)

    # Menampilkan visualisasi data
    st.header('Data Visualization')

    # Visualisasi distribusi data
    st.subheader("Data Distribution")
    feature_dist = st.selectbox("Feature",
                             df.columns)
    st.write("Histogram for feature", feature_dist)
    fig, ax = plt.subplots()
    ax.hist(df[feature_dist])

    st.pyplot(fig)

    # Visualisasi scatter plot
    st.subheader("Scatter Plot")
    feature_1 = st.selectbox("First feature",
                             df.columns)
    feature_2 = st.selectbox("Second feature",
                             df.columns)

    st.write("You have chosen feature:", feature_1, " and ", feature_2)
    st.scatter_chart(df, x=feature_1, y=feature_2,)
