%%writefile app.py
import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
st.set_option('deprecation.showfileUploaderEncoding', False)
# Load the pickled model
model = pickle.load(open('/content/drive/My Drive/hcclusterassignment.pkl','rb'))   
dataset= pd.read_csv('/content/drive/My Drive/Wholesale customers data.csv')
X = dataset.iloc[:,2:8].values
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
def predict_note_authentication(fresh,milk,grocery,frozen,detergents,delicassen):
  predict= model.fit_predict(sc.transform([[fresh,milk,grocery,frozen,detergents,delicassen]]))  
  return predict
html_temp = """
   <div class="" style="background-color:blue;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
   <center><p style="font-size:30px;color:white;margin-top:10px;">Department of Computer Engineering</p></center> 
   <center><p style="font-size:25px;color:white;margin-top:10px;"Machine Learning End Term</p></center> 
   </div>
   </div>
   </div>
   """
st.markdown(html_temp,unsafe_allow_html=True)
st.header("End Term Disease Project ")
  
  
Gender = st.number_input('Gender',0,1)
Glucose = st.number_input("Glucose",55,500)
BP = st.number_input("BP",3,100000)
SkinThickness = st.number_input("SkinThickness",25,100000)
Insulin = st.number_input("Insulin",3,100000)
BMI = st.number_input("BMI",3,100000)
PedigreeFunction== st.number_input("PedigreeFunction",3,100000)
Age= st.number_input("Age",3,100000)

if st.button("Predict"):
  result=predict_note_authentication(fresh,milk,grocery,frozen,detergents,delicassen)
  st.success('Model has predicted {}'.format(result))
      
if st.button("About"):
  st.subheader("Developed by Suyash Sharma")
  st.subheader("Department of Computer Engineering")
html_temp = """
   <div class="" style="background-color:orange;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:20px;color:white;margin-top:10px;">Machine learning</p></center> 
   </div>
   </div>
   </div>
   """
st.markdown(html_temp,unsafe_allow_html=True)