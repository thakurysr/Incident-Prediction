import pandas as pd
import streamlit as st 
from sklearn.ensemble import RandomForestClassifier
import pickle
from pickle import dump
from pickle import load

st.title('Machine Failure Prediction Using Random Forest')

st.sidebar.header('User Input Parameters')

def user_input_features():
    Type = st.sidebar.selectbox('Type',('2','1','0'))
    AT_K = st.sidebar.number_input("Insert the Air Temperature in Kelvin")
    PT_K = st.sidebar.number_input("Insert the Process Temperature in Kelvin")
    Torque = st.sidebar.number_input("Insert Torque in Nm")
    TW_min = st.sidebar.number_input("Insert tool wear in min")
   
    data = {'Type': Type,
            'Air Temp': AT_K,
	    'Process Temp' : PT_K,
            'Torque' : Torque,
            'Tool Wear' : TW_min}
    features = pd.DataFrame(data,index = [0])
    return features 

df = user_input_features()
st.subheader('User Input parameters')
st.write(df)

# load the model from disk
loaded_model = load(open("machine_failure.sav", 'rb'))

prediction = loaded_model.predict(df)
prediction_proba = loaded_model.predict_proba(df)

st.subheader('Predicted Result')
st.write('Yes' if prediction_proba[0][1] > 0.5 else 'No')

st.subheader('Prediction Probability')
st.write(prediction_proba)