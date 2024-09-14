import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from PIL import Image
import os

# Load the dataset
@st.cache
def load_data():
    return pd.read_csv("./Data/TripAdvisor_RestauarantRecommendation.csv")

df = load_data()

# Data preprocessing
df["Location"] = df["Street Address"] + ', ' + df["Location"]
df = df.drop(['Street Address'], axis=1)
df = df[df['Comments'].notna()]
df = df.drop_duplicates(subset='Name')
df = df.reset_index(drop=True)

# Display the first few rows of the dataset
st.write(df.head())

# Function to recommend restaurants based on 'Comments'
def recom(dataframe, name):
    dataframe = dataframe.drop(["Trip_advisor Url", "Menu"], axis=1)

    # Creating recommendations using TF-IDF on Comments
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(dataframe.Comments)
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    # Get restaurant index by name
    indices = pd.Series(dataframe.index, index=dataframe.Name).drop_duplicates()
    if name not in indices:
        st.write("Restaurant not found!")
        return
    
    idx = indices[name]
    if isinstance(idx, pd.Series):
        idx = idx[0]

    # Get similarity scores and recommend top 10
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    restaurant_indices = [i[0] for i in sim_scores]

    recommended = list(dataframe['Name'].iloc[restaurant_indices])

    st.write("Top 10 Restaurants you might like:")
    for restaurant in recommended:
        st.write(f"- {restaurant}")

    # User input to select a recommended restaurant
    title = st.selectbox('Select a restaurant from the recommended list:', recommended)

    if title in dataframe['Name'].values:
        reviews = dataframe.at[dataframe['Name'].eq(title).idxmax(), 'Reviews']
        st.write(f"Restaurant Rating: {reviews}")

        comment = dataframe.at[dataframe['Name'].eq(title).idxmax(), 'Comments']
        if comment != "No Comments":
            st.write(f"Comments: {comment}")

        rest_type = dataframe.at[dataframe['Name'].eq(title).idxmax(), 'Type']
        st.write(f"Restaurant Category: {rest_type}")

        location = dataframe.at[dataframe['Name'].eq(title).idxmax(), 'Location']
        st.write(f"The Address: {location}")

        contact_no = dataframe.at[dataframe['Name'].eq(title).idxmax(), 'Contact Number']
        if contact_no != "Not Available":
            st.write(f"Contact Details: Phone: {contact_no}")

    # Display an image of the restaurant (if applicable, else you can skip or handle differently)
    image_path = '/path/to/restaurant_image.jpg'
    if os.path.isfile(image_path):
        try:
            image = Image.open(image_path)
            st.image(image, caption='Restaurant Image')
        except Exception as e:
            st.write(f"Error loading image: {str(e)}")
    else:
        st.write("Image not found.")

# Collect User Feedback
st.write("Rate Your Experience")
rating = st.slider('Rate this restaurant (1-5)', 1, 5)
feedback_comment = st.text_area('Your Feedback')

if st.button('Submit Feedback'):
    # Save the feedback to a CSV file
    feedback_file = 'feedback.csv'
    
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
