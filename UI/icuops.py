import streamlit as st

import pandas as pd
import numpy as np
import json
import base64

import re
import requests
import streamlit.components.v1 as components


pat1_params= [27.0, 25.837205576029483, 1, 2, 0, 158.0, 8, 2, 951, 0, 5, 0.5669565116970012, 64.5, 3.2, 308.0, 1903.01, 1, 0, 0.0, 8.0, 0.9, 0.0, 4.0, 6.0, 0.0, 5.0, 160.0, 106.0, 35.9, 0, 56.0, 0.0, 0.0, 0.0, 0.0, 10.0, 140.0, 36.4, 1278.0, 0, 18.8, 0.0, 0.0, 48.0, 48.0, 48.0, 48.0, 62.0, 62.0, 0.0, 0.0, 59.0, 59.0, 59.0, 59.0, 16.0, 16.0, 100.0, 100.0, 0.0, 0.0, 99.0, 99.0, 99.0, 99.0, 36.4, 36.1, 0.0, 0.0, 83.0, 74.0, 83.0, 74.0, 66.0, 55.0, 0.0, 0.0, 95.0, 88.0, 95.0, 88.0, 14.0, 11.0, 100.0, 99.0, 0.0, 0.0, 133.0, 121.0, 133.0, 121.0, 36.1, 36.1, 3.2, 3.2, 0.0, 0.0, 8.0, 8.0, 8.7, 8.7, 0.9, 0.9, 160.0, 148.0, 25.0, 25.0, 12.1, 11.7, 36.2, 35.9, 0.0, 0.0, 0.0, 0.0, 209.0, 207.0, 4.3, 4.3, 140.0, 140.0, 18.8, 15.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 148.0, 148.0, 0.0, 0.0, 12.1, 12.1, 36.2, 36.2, 0.0, 0.0, 0.0, 0.0, 207.0, 207.0, 0.0, 0.0, 0.0, 0.0, 15.6, 15.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 15.0, 310, 1900, 25, 160, 65, 30, 6.0, 1903.0, 308.0, 1, 8.0, 4831712495883404950, -0.037205576029478486, 1.0450046511627908, 2.3888888888888893, 20, 34, -14, 0.0, 0.0, 0, 0, 0, 0, 0.0, False, False, 0.4000000000000004, 0.0, 1, 0, 0, 1, 0, False, True, 0.0, 0.0, 0, 0, 0, 0, 0.0, False, False, 2.0, 0.0, 0, 1, 0, 1, 0, False, True, 3.200000000000001, 0.0, 0, 1, 0, 1, 0, False, True, 0.0, 0.0, 0, 0, 0, 0, 0.0, False, False, 0.0, 11.0, 0, 0, 1, 0, 0.0, False, False, 0.30000000000000426, 0.0, 1, 0, 0, 1, 0, False, True, 0.0, 9.0, 0, 0, 1, 0, 0.0, False, True, 0.0, 0.0, 0, 0, 0, 0, 0.0, False, False, 0.2999999999999972, 0.0, 0, 1, 0, 1, 0, False, True, 0.0, 0.0, 0, 0, 1, 0, 0.0, True, False, 0.0, 0.0, 0, 0, 1, 0, 0.0, True, False, 0.0, 0.0, 0, 0, 0, 0, 0.0, False, False, 0.0, 9.0, 0, 0, 1, 0, 0.0, False, True, 0.0, 0.0, 0, 0, 0, 0, 0.0, False, False, 0.0, 0.0, 0, 0, 1, 0, 0.0, True, False, 0.0, 3.0, 0, 0, 1, 0, 0.0, False, True, 0.0, 0.0, 0, 0, 1, 0, 0.0, True, False, 0.0, 0.0, 0, 0, 1, 0, 0.0, True, False, 12.0, 0.0, 0, 1, 0, 1, 0, False, True, 0.0, 0.0, 0, 0, 0, 0, 0.0, False, False, 0.0, 0.0, 0, 0, 0, 0, 0.0, False, False, 0.0, 1.0, 1, 0, 1, 0, 0.0, False, False, 0.0, 0.0, 0, 0, 0, 0, 0.0, False, False, 0.0, 7.0, 0, 0, 1, 0, 0.0, False, True, 0.0, 7.0, 0, 0, 1, 0, 0.0, False, True, 0.0, 0.0, 0, 0, 1, 0, 0.0, True, False, 0.0, 0.0, 0, 0, 0, 0, 0.0, False, False, 0.0, 12.0, 0, 0, 1, 0, 0.0, False, True, 0.0, 0.0, 0, 0, 1, 0, 0.0, True, False, 0.0, 12.0, 0, 0, 1, 0, 0.0, False, True, 7, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 13, 0.2711864406779661, 0.16161616161616166, 0.0, 0.6262626262626263, 1.0, 256.0, 100.0, 0, 0, 0, 0.0]

arr = np.array(pat1_params)
pat_param_names=['Age', 'Bmi', 'creatinine_apache', 'albumin_apache']
idx_list = [0, 1, 10, 11]
main_params = arr[idx_list]
patient_data = {"inputs": [pat1_params ]}
db2_srving_url = "http://test-wids-widsdb2.default-tenant.app.mlops1.iguazio-c0.com/"
db2_srving_headers = {
            "Content-Type": "application/json",
            "X-v3io-function": "widsdb2-lightgbm-serving",
            "Authorization": "Basic YXJ1bmFfbGFua2E6cXovQ09gcmA0QiFK"
            }


#Function definitions


def runDb2Prediction():
   payload = patient_data #the ones mentioned in --data-raw and that has inputs
   response = requests.post(db2_srving_url, json=payload, headers=db2_srving_headers)
   return(response.json()) 




def getAnswersforQuery(query):
    url = "http://test-qna-wnlp.default-tenant.app.mlops1.iguazio-c0.com/"
    headers = {
            "Content-Type": "application/json",
            "X-v3io-function": "wnlp-qnaserving",
            "Authorization": "Basic YXJ1bmFfbGFua2E6cXovQ09gcmA0QiFK"
          }
    
    if query !='':
        data = {'query': query}
        r = requests.post(url, data=data)

	#JSON_DATA = {"inputs":[{"query": "what are diabetes mellitus risks ?"}]}
        json_data = {'inputs': [data]}
        payload = json_data #the ones mentioned in --data-raw and that has inputs


        response = requests.post(url, json=payload, headers=headers)
        print(response.text)
        a_json = response.json()
        st.write('Model trained for custom dataset QnA ' , a_json['model_name'])
        #st.write(a_json['outputs'])
        #f = pd.read_json(response.json)
        
        df = pd.DataFrame(a_json['outputs'], columns=['Index', 'Title', 'Text', 'Score'])
        st.table(df)
        #st.table(response.json()['outputs'])
    
        #st.write(response.json())
     



#### Application starts
st.sidebar.image('nav.jpg')
st.sidebar.write(' Welcome to ICUOps prediction of Diabetes Mellitus Application')
selection = st.sidebar.selectbox("Go to page:", [ 'Research Study NLP Analysis', 'QnA Model' , 'Prediction ', 'Model Details'])


#Main Window


st.image('main.png')
st.title('ICU Diabetes Mellitus Prediction Application   ')
if selection == 'QnA Model':
        st.header('Analysis of Diabetes 2 data QnA system ')
        st.write("The  questions on diabetes2 from research publications can be queried here ") 

        query = st.text_input("Enter query here" )
        if st.button('Get Answers '):
           st.write('Retreiving the answers from research papers:')
           getAnswersforQuery(query)
  
elif selection == 'Prediction ':
        st.header('Prediction of Diabetes Mellitus on ICU data')
        st.write('ICU patient parameters are passed to the trained model serving on MLRun platorm to get the prediction' )

        patientList = ('patient1', 'patient2', 'patient3')
        
        patient = st.selectbox(" Select patient Id to send the parameters for model prediction", patientList) 
        id = patientList.index(patient)
        

        patdf = pd.DataFrame([main_params], columns=pat_param_names)
        st.table(patdf)

        if st.button('Run  Model Inference for Diabetes Mellitus Prediction'):
           result = runDb2Prediction()
           db_prb =  result['outputs'][0]
           db_pred = False
           if db_prb >0.5 :
              db_pred = True
           st.write('Prediction of Diabetes Mellitus of Patient from Model ', result['model_name'] ,':',db_pred )
   


elif selection == 'Research Study NLP Analysis':
        st.header(' NLP model visualizations')
        
        st.write(' This NLP dataset is created from research study documents focussing on Diabetes Mellitus conditions, all documents are available on internet ')

        st.write(' Word Cloud on Diabetes Mellitus Research dataset ')
        st.image('wc1.png')

        st.write(' LDA on  Diabetes Mellitus Research dataset ')
        st.image('tm.png')

elif selection == 'Model Details':
        st.header('Prediction Model Details')

        st.write(' Interpretability of model  from SHAP')  
        st.image('shap1.png')
        st.image('shap2.png')

        st.write(' Feature Importance from LGBM Model')  
        st.image('f1.png')





