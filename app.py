import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
import preprocessing 

 


st.set_page_config(page_title="KMeans Clustering App", page_icon="")

st.title(" Customer Clustering using KMeans")
st.write("Upload your dataset to predict its clusters using the trained KMeans model.")



preprocessors = joblib.load('preprocessors.pkl')
model = joblib.load('kmeans_model.pkl')

uploaded_file = st.file_uploader(" Upload your test CSV file", type=["csv"])

if uploaded_file is not None:

    test_data = pd.read_csv(uploaded_file)
    st.subheader(" Uploaded Data Preview")
    st.dataframe(test_data.head())

    st.info(" Processing test data...")
    test_processed = preprocessing.preprocessing_test(test_data, preprocessors)
    print(test_data)

    clusters = model.predict(test_processed)
    test_data['Predicted_Cluster'] = clusters


    st.success(" Clustering complete!")
    st.subheader(" Data with Predicted Clusters")
    st.dataframe(test_data.head(20))


    csv = test_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=" Download clustered data as CSV",
        data=csv,
        file_name="clustered_output.csv",
        mime="text/csv"
    )
else:
    st.warning("Please upload a CSV file to start.")
