import streamlit as st
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
import pickle
#title
col=['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']
st.title('California Housing Price Prediction')
st.image('https://miro.medium.com/v2/resize:fit:1024/0*YMZOAO8QE4bZ4_Rk.jpg' )
st.header('A Model of housing prices to predict median house values in California',divider=True)
st.subheader('''User must enter given value to Predict Price:''')
st.sidebar.title('Select House Feature🏠')
st.sidebar.image('https://smartasset.com/wp-content/uploads/sites/2/2013/03/modern-custom-suburban-home-exterior-picture-id1255835529-1.jpg')

#read_data
temp_df=pd.read_csv('california.csv')
random.seed(12)

all_values = []
for i in temp_df[col]:
    min_value, max_value=temp_df[i].agg(['min','max']) 
    var=st.sidebar.slider(f'Select {i} Range',int(min_value),int(max_value),random.randint(int(min_value),int(max_value)))

    all_values.append(var)
    
ss=StandardScaler()
ss.fit(temp_df[col])
final_value= ss.transform([all_values])

with open('house_price_pred_ridge_model.pkl','rb') as f:
    chatgpt=pickle.load(f)

price=chatgpt.predict(final_value)[0]

import time 
st.write(pd.DataFrame(dict(zip(col,all_values)),index=[1]))
progress_bar= st.progress(0)
placeholder=st.empty()
placeholder.write('Predicting Price...')
place=st.empty()
place.image('https://cdn.pixabay.com/animation/2023/06/13/15/12/15-12-51-616_512.gif', width=100)
if price>0:
   
    for i in range(100):
        time.sleep(0.05)
        progress_bar.progress(i+1)
    body=f'Predicted Median House Price:${round(price,2)} Thousand Dollar'
    placeholder.empty()
    place.empty()
    st.success(body)
else:
    body='Invalid House Feature Values'
    st.warning(body)
    