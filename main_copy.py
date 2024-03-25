import pandas as pd
import streamlit as st
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns


def main():
    # function to classify the sentiment
    def remark(com):
        if com > 0.25:
            return 'positive'
        if com < -0.25:
            return 'negative'
        else:
            return 'neutral'
        
    
    # Function to color text
    def text_color(value):
        if value == 'negative':
            return "red"  # Green color for positive values
        elif value == 'positive':
            return "lightgreen"    # Red color for negative values
        else:
            return "yellow"
    
    # creating the dictionary of reviews
    if 'list_of_reviews' not in st.session_state:
        st.session_state.list_of_reviews={}
        
    # Disply structure
    st.title('ReviewScope')
    col = st.columns(2)

    # input section
    with col[0].form(key='input form'):
        user_input = st.text_input('Please leave a review')
        
        button = st.form_submit_button(label='Submit')
        
        if button:
            
                # sentiment analyzer
            sia = SentimentIntensityAnalyzer()
            intensity = sia.polarity_scores(user_input)  
            
            keys = list(intensity.keys())
            values = list(intensity.values()) 
            
            new_sentiment = remark(intensity['compound'])
            st.session_state.list_of_reviews[user_input] = new_sentiment

    st.write(f'Review Count: {len(st.session_state.list_of_reviews)}')
    
    
    #reviews section
    st.subheader('REVIEWS')   
    for i,j in st.session_state.list_of_reviews.items():
        
        color = text_color(j)   
        colored_text = f'<span style="color:{color}">{j}</span>'
        res = f'* {i} \[{colored_text}\]'
        st.write(res, unsafe_allow_html=True)

    st.set_option('deprecation.showPyplotGlobalUse', False)   
    # Plotting the bar chart
    remark_list = list(st.session_state.list_of_reviews.values())

    value_counts = pd.Series(remark_list).value_counts().reset_index()
    value_counts.columns = ['Customer Remark', 'Count']
    # Plot bar chart
    plt.figure(figsize=(8,6))
    sns.barplot(x='Customer Remark', y='Count', data=value_counts)
    plt.xlabel('Count')
    plt.ylabel('Customer Remark')
    plt.title('Value Counts of Customer remark')
    plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    col[1].pyplot()

if __name__ == "__main__":
    main()