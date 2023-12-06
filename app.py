import streamlit as st
import numpy as np
from keras.models import model_from_json

file = open('lstm_model.json', 'r')
loaded  = file.read()
file.close()

lstm_model = model_from_json(loaded)
lstm_model.load_weights("lstm_model.h5")

def main():

     # front end elements of the web page
    html_temp = """ 

    <div style ="background-color:blue;padding:13px"> 
    <h1 style ="color:yellow;text-align:center;">Wind Turbine Gear Oil Temperature Prediction</h1> 
    </div> 
    """
    
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True)
    st.markdown('') 
    st.markdown('**Enter Temperatures of previous 30 minutes to predict the Temperature for next 10 minutes**')

    x1 = st.number_input("Enter Temperature of First 10 min (in °C)", value=0.0)
    x2 = st.number_input("Enter Temperature next 10 min (in °C)", value=0.0)
    x3 = st.number_input("Enter Temperature of Final 10 min (in °C)", value=0.0)

    if st.button("Predict"):
        input_data = np.array([[[x3, x2, x1]]])
        prediction = lstm_model.predict(input_data)
        st.success(f'The predicted Gear Oil Temperature for the next 10 min is: {prediction[0][0]:.2f} °C')

if __name__=='__main__': 
    main()
