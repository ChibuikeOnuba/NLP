import streamlit as st
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

def main():
    
    def remark(com):
        if com > 0.25:
            return 'positive'
        if com < -0.25:
            return 'negative'
        else:
            return 'neutral'
    
    if 'list_of_reviews' not in st.session_state:
        st.session_state.list_of_reviews={}
    # Disply structure
    st.title('TEXT SENTIMENT ANALYZER')
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
        st.write('*',i,'[',j,']')   
         


    #if col[0].button('Submit'):
    #    list_of_reviews.append(input)
    #    st.write(f'Review Count: {len(list_of_reviews)}')
         



    # Plotting the bar chart



    # COMMENT
    st.text('Note: The compound value shows the sentiment of the comment on a scale of -1 and 1.')
    st.text('values closer to 1 are positive reviews while values closer to -1 are negative reviews')

if __name__ == "__main__":
    main()