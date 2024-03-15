import streamlit as st
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

def main():
    
    # Creating a list of reviews (serves as the database in this use case)
    list_of_reviews = []
    
    # Disply structure
    st.title('TEXT SENTIMENT ANALYZER')
    col = st.columns(2)

    # input section
    with col[0].form(key='input form'):
        user_input = st.text_input('Review product')
        
        button = st.form_submit_button(label='Submit')
        
        if button:
            list_of_reviews.append(user_input)

    st.write(f'Review Count: {len(list_of_reviews)}')
    
    #reviews section
    st.subheader('REVIEWS')   
    for i in list_of_reviews:
        st.write('*',i)   
         
    # sentiment analyzer
    sia = SentimentIntensityAnalyzer()
    output = sia.polarity_scores(user_input)
    st.write(output)

    #if col[0].button('Submit'):
    #    list_of_reviews.append(input)
    #    st.write(f'Review Count: {len(list_of_reviews)}')
         

    
    

    keys = list(output.keys())
    values = list(output.values())


    # Plotting the bar chart

    colors = ['red' if value <0 else 'blue' for value in values]
    fig, ax = plt.subplots(figsize=(5,4))
    ax.bar(keys, values, color=colors)
    ax.axhline(0, color='gray',linestyle='--', linewidth=0.5)
    ax.set_ylim(-1,1)
    col[1].pyplot(fig)

    # COMMENT
    st.text('Note: The compound value shows the sentiment of the comment on a scale of -1 and 1.')
    st.text('values closer to 1 are positive reviews while values closer to -1 are negative reviews')

if __name__ == "__main__":
    main()