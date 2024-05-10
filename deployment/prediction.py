import streamlit as st
import pandas as pd
import pickle as pkl
from bank_market_func import binary_mapping, null_imputer
from datetime import datetime


def run():
    st.title('Term Deposit Subscription Prediction')

    # Melakukan loading pickle files
    with open('bank_market_svm.pkl', 'rb') as file_1:
        svm_best_estimator = pkl.load(file_1)

    # Menampilkan data
    @st.cache_data
    def fetch_data():
        df = pd.read_csv('P1M2_vania_alya.csv', sep = ';')
        return df

    df = fetch_data()
    st.header('Term Deposit Subscription Data')
    st.write(df)

    st.header('User Input Features')

    with st.form("User Input Features"):
        age = st.number_input("How old is the client?", min_value=0)
        job = st.text_input('What is the clients job?')
        marital = st.selectbox('What is the client marital status?',
                               ('married', 'single', 'divorced'))
        education = st.selectbox('What is the clients educational level?',
                                 ('primary','secondary','tertiary'))
        default = st.radio('Do client have default credit?', ['yes','no'])
        balance = st.number_input("How much is the client balance?", min_value=0)
        housing = st.radio('Do client have housing loan?', ['yes','no'])
        loan = st.radio('Do client have personal loan?', ['yes','no'])
        temp_contact = st.selectbox('How was the client contacted through?',
                               ('cellular', 'telephone', "I don't know..."))
        date = st.date_input("When did you last contact client?", format="DD/MM/YYYY")
        duration = st.number_input("How long did the call last (in seconds)?", min_value=0)
        campaign = st.number_input("How many campaign did the client get?", min_value=0)
        pdays = st.number_input("How many days have passed since the client last contacted?", min_value=0)
        previous = st.number_input("How many campaigns did the user get?", min_value=0)
        temp_poutcome = st.selectbox('How was the outcome of the last campaign?',
                               ('success', 'failure', "I don't know..."))
        
        sub = st.form_submit_button("Submit user data")

    # Pengecekan input pada poutcome
    if temp_poutcome == "I don't know...":
        poutcome = 'unknown'
    else :
        poutcome = temp_poutcome
    
    # Pengecekan input pada contact
    if temp_contact == "I don't know...":
        contact = 'unknown'
    else :
        contact = temp_contact
    
    data = {
        'age': age,
        'job': job,
        'marital': marital,
        'education': education,
        'default': default,
        'balance': balance,
        'housing': housing,
        'loan': loan,
        'contact': contact,
        'day': date.day,
        'month': date.month,
        'duration': duration,
        'campaign': campaign,
        'pdays': pdays,
        'previous': previous,
        'poutcome': poutcome
    }

    # Menampilkan hasil data user input
    "Hasil input user adalah sebagai berikut :"
    features = pd.DataFrame(data, index=[0])
    st.write(features)

    if sub:
        # Adjustment of the features to be used
        inf_data = features[['duration','previous','balance','poutcome','contact','housing','loan']]

        # Perform imputation by adding the var+_NA column
        # Defining columns to be imputed
        impute_column = ['poutcome']
        # Iterating through the value for imputation
        for var in impute_column:
            null_imputer(inf_data, var, 'unknown')
        
        # Binary encoding
        # Define binary data
        bin_columns = ['housing', 'loan']

        # Generate binary data using map
        for var in bin_columns:
            binary_mapping(inf_data, var)

        # Dropping initial data
        inf_data_enc = inf_data.drop(columns=bin_columns)

        # Melakukan prediksi
        hasil_predict = svm_best_estimator.predict(inf_data_enc)

        if hasil_predict == 1:
            hasil_predict = 'User will subscribe to Term Deposit'
        else:
            hasil_predict = 'User will not subscribe to Term Deposit'

        st.write('Based on user input, the model predicts:')
        st.write(hasil_predict)
