import pandas as pd
import streamlit as st
import os
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load the dataset
df = pd.read_csv("./Data/TripAdvisor_RestauarantRecommendation1.csv")

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
st.markdown("<h1 style='text-align: center;'>Recommended Restaurants</h1>", unsafe_allow_html=True)
st.markdown("""### Find the best restaurants based on your taste!""")

# User inputs for filtering by rating and type
st.markdown("### Filter Restaurants by Rating and Type")
rating_filter = st.slider('Minimum Rating', 1.0, 5.0, step=0.5)
type_filter = st.multiselect('Select Restaurant Type', df['Type'].unique(), default=df['Type'].unique())

# Filter the dataframe based on user input
filtered_df = df[(df['Ratings'] >= rating_filter) & (df['Type'].isin(type_filter))]

# Display the filtered list of restaurants
st.markdown(f"### Showing {len(filtered_df)} Restaurants Matching Your Filters")
name = st.selectbox('Select the Restaurant you like', filtered_df['Name'].unique())

# Function to recommend restaurants based on the 'Type' of the restaurant
def recom(dataframe, name):
    dataframe = dataframe.drop(["Trip_advisor Url", "Menu"], axis=1)
    
    # Creating recommendations based on 'Type'
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(dataframe['Type'])  # Using 'Type' for recommendations
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Mapping restaurant names to their indices
    indices = pd.Series(dataframe.index, index=dataframe.Name).drop_duplicates()

    # Find the index of the restaurant selected by the user
    idx = indices[name]
    if isinstance(idx, pd.Series):
        idx = idx[0]

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
    title = st.selectbox('Restaurants most similar [Based on user ratings(collaborative)]', recommended['Name'])
    if title in dataframe['Name'].values:
        details = dataframe[dataframe['Name'] == title].iloc[0]
        reviews = details['Reviews']

        st.markdown("### Restaurant Rating:")
        st.write(f"**{reviews}**")

        # Display comments
        if 'Comments' in dataframe.columns:
            comment = details['Comments']
            if comment != "No Comments":
                st.markdown("### Comments:")
                st.warning(comment)
            else:
                pass

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

# Call the recommendation function
recom(filtered_df, name)

# Collect User Feedback
st.markdown("## Rate Your Experience")
rating = st.slider('Rate this restaurant (1-5)', 1, 5)
feedback_comment = st.text_area('Your Feedback')

if st.button('Submit Feedback'):
    # Save the feedback to a CSV file
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
    
    # Clear the fields after submission
    st.session_state.rating = None
    st.session_state.feedback_comment = ''
    
    st.success('Thanks for your feedback!')
