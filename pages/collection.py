import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import datetime


def main():
    def input(id,tag):
        with col[tag].form(key=id):
            st.write(id)
            user_input = st.text_input('Please leave a review')
            
            button = st.form_submit_button(label='Submit')
            
            return user_input, button, id
            
    def add_record(data_dict, input, category, time):
        new_index = len(data_dict) +1
        data_dict[f'{new_index}'] = {'review':input, 'category':category, 'timestamp':time}
    
    st.title('STORE')
    col = st.columns(3)
    
    if 'reviews' not in st.session_state:
        st.session_state.reviews={}
    
    col[0].image(Image.open(r'C:\Users\HP\Documents\NLP\pages\img1.jpeg').resize((313,200)))
    user_input, button, id = input('Trousers',0)
    if button:
        timestamp = datetime.datetime.now()
        add_record(st.session_state.reviews, user_input, id, timestamp)
        
    col[1].image(Image.open(r'C:\Users\HP\Documents\NLP\pages\img2.jpeg').resize((313,200)))
    user_input, button, id = input('Skirts',1)
    if button:
        timestamp = datetime.datetime.now()
        add_record(st.session_state.reviews, user_input, id, timestamp)
    col[2].image(Image.open(r'C:\Users\HP\Documents\NLP\pages\img1.jpeg').resize((313,200)))
    user_input, button, id = input('Shoe',2)
    if button:
        timestamp = datetime.datetime.now()
        add_record(st.session_state.reviews, user_input, id, timestamp)
    col[1].image(Image.open(r'C:\Users\HP\Documents\NLP\pages\img2.jpeg').resize((313,200)))
    user_input, button, id = input('Shirt',1)
    if button:
        timestamp = datetime.datetime.now()
        add_record(st.session_state.reviews, user_input, id, timestamp)
    
    
    st.write(pd.DataFrame.from_dict(st.session_state.reviews, orient='index'))
    
if __name__ == '__main__':
    main()