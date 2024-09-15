import pandas as pd
import streamlit as st
import os
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load the dataset
df = pd.read_csv("./Data/TripAdvisor_RestauarantRecommendation.csv")

# Combine 'Street Address' and 'Location' into one 'Location' column and clean the data
df["Location"] = df["Street Address"] + ', ' + df["Location"]
df = df.drop(['Street Address'], axis=1)
df = df[df['Type'].notna()]  # Only consider rows where 'Type' is not missing
df = df.drop_duplicates(subset='Name')
df = df.reset_index(drop=True)

# Streamlit configuration
st.set_page_config(layout='centered', initial_sidebar_state='expanded')
st.sidebar.image('Data/App_icon.png')

# Main page title and introduction
st.markdown("<h1 style='text-align: center;'>Recommended</h1>", unsafe_allow_html=True)

st.markdown("""
### Welcome to Restaurant Recommender!

Looking for the perfect place to dine? Look no further! Our Restaurant Recommender is here to help you discover the finest dining experiences tailored to your taste.

### How It Works:

1. **Select Your Favorite Restaurant:**
   Choose from a list of renowned restaurants that pique your interest.

2. **Explore Similar Gems:**
   Our advanced recommendation system analyzes customer reviews and ratings to suggest similar restaurants you might love.

3. **Discover Your Next Culinary Adventure:**
   Dive into detailed information about each recommended restaurant, including ratings, reviews, cuisine types, locations, and contact details.

4. **Enjoy Your Meal:**
   With our recommendations in hand, savor a delightful dining experience at your chosen restaurant!

### Start Your Culinary Journey Now!

Begin exploring the diverse culinary landscape and uncover hidden gastronomic treasures with Restaurant Recommender.
↓
""")

image = Image.open('Data/food_cover.jpg')
st.image(image, use_column_width=True)

st.markdown("### Select Restaurant")

# User input to select a restaurant
name = st.selectbox('Select the Restaurant you like', list(df['Name'].unique()))

def recom(dataframe, name):
    # Drop unnecessary columns and filter out rows with missing 'Comments'
    dataframe = dataframe.drop(["Trip_advisor Url", "Menu"], axis=1)
    dataframe = dataframe[dataframe['Comments'].notna() & (dataframe['Comments'] != "No Comments")]

    # TF-IDF Vectorization based on 'Type'
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(dataframe['Type'])  # Using 'Type' for recommendations
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Mapping restaurant names to their indices
    indices = pd.Series(dataframe.index, index=dataframe['Name']).drop_duplicates()

    # Find the index of the restaurant selected by the user
    idx = indices.get(name)
    if idx is None:
        st.warning("Selected restaurant is not found in the dataset.")
        return

    # Get similarity scores for all restaurants
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Top 10 most similar restaurants, excluding the selected one
    restaurant_indices = [i[0] for i in sim_scores]

    # Get the names and ratings of the top 10 recommended restaurants
    recommended = dataframe.iloc[restaurant_indices]
    recommended = recommended[['Name', 'Ratings']]

    # Sort recommended restaurants by their ratings
    recommended = recommended.sort_values(by='Ratings', ascending=False)

    st.markdown("## Top 10 Restaurants you might like:")

    # User selects from the list of recommended restaurants
    title = st.selectbox('Restaurants most similar [Based on Type]', recommended['Name'])
    if title in dataframe['Name'].values:
        details = dataframe[dataframe['Name'] == title].iloc[0]
        reviews = details['Reviews']

        st.markdown("### Restaurant Rating:")

        # Display reviews as images
        review_images = {
            '4.5 of 5 bubbles': 'Data/Ratings/Img4.5.png',
            '4 of 5 bubbles': 'Data/Ratings/Img4.0.png',
            '5 of 5 bubbles': 'Data/Ratings/Img5.0.png'
        }
        review_image_path = review_images.get(reviews)
        if review_image_path:
            image = Image.open(review_image_path)
            st.image(image, use_column_width=True)
        
        # Display comments
        comment = details['Comments']
        if comment != "No Comments":
            st.markdown("### Comments:")
            st.warning(comment)
        
        # Display type of restaurant
        rest_type = details['Type']
        st.markdown("### Restaurant Category:")
        st.error(rest_type)

        # Display location
        location = details['Location']
        st.markdown("### The Address:")
        st.success(location)

        # Display contact details
        contact_no = details['Contact Number']
        if contact_no != "Not Available":
            st.markdown("### Contact Details:")
            st.info('Phone: ' + contact_no)

    st.text("")
    image = Image.open('Data/food_2.jpg')
    st.image(image, use_column_width=True)

# Call the recommendation function
recom(df, name)
