import pandas as pd
import streamlit as st
from PIL import Image
import os

st.set_page_config(layout='centered', initial_sidebar_state='expanded')
st.sidebar.image('Data/App_icon.png')
st.markdown("<h1 style='text-align: center;'>Restaurants</h1>", unsafe_allow_html=True)

# Your existing code for restaurant selection and details here...

# Collect User Feedback
st.markdown("## Rate Your Experience")

# Create placeholders for feedback input
feedback_placeholder = st.empty()

with feedback_placeholder.form(key='feedback_form'):
    rating = st.slider('Rate this restaurant (1-5)', 1, 5)
    feedback_comment = st.text_area('Your Feedback')
    submit_button = st.form_submit_button(label='Submit Feedback')

    if submit_button:
        feedback_file = 'Data/feedback.csv'

        # Create the CSV file if it doesn't exist
        if not os.path.isfile(feedback_file):
            feedback_df = pd.DataFrame(columns=['Reviews', 'Comments'])
            feedback_df.to_csv(feedback_file, index=False)

        # Load existing feedback data
        feedback_df = pd.read_csv(feedback_file)

        # Append new feedback
        new_feedback = pd.DataFrame([{'Reviews': f'{rating} of 5 bubbles', 'Comments': feedback_comment}])
        feedback_df = pd.concat([feedback_df, new_feedback], ignore_index=True)
        feedback_df.to_csv(feedback_file, index=False)

        st.success('Thanks for your feedback!')

        # Clear the feedback form
        feedback_placeholder.empty()
