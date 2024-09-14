import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from PIL import Image

# Load the dataset
df = pd.read_csv("/content/TripAdvisor_RestauarantRecommendation.csv")

df["Location"] = df["Street Address"] + ', ' + df["Location"]
df = df.drop(['Street Address'], axis=1)
df = df[df['Comments'].notna()]
df = df.drop_duplicates(subset='Name')
df = df.reset_index(drop=True)

# Display the first few rows of the dataset
print(df.head())

# Function to recommend restaurants based on 'Comments'
def recom(dataframe, name):
    dataframe = dataframe.drop(["Trip_advisor Url", "Menu"], axis=1)

    # Creating recommendations using TF-IDF on Comments
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(dataframe.Comments)
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    # Get restaurant index by name
    indices = pd.Series(dataframe.index, index=dataframe.Name).drop_duplicates()
    idx = indices[name]
    if isinstance(idx, pd.Series):
        idx = idx[0]

    # Get similarity scores and recommend top 10
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    restaurant_indices = [i[0] for i in sim_scores]

    recommended = list(dataframe['Name'].iloc[restaurant_indices])

    print("\nTop 10 Restaurants you might like:")
    for restaurant in recommended:
        print(f"- {restaurant}")

    # User input to select a recommended restaurant
    title = input("\nSelect a restaurant from the recommended list: ")

    if title in dataframe['Name'].values:
        reviews = dataframe.at[dataframe['Name'].eq(title).idxmax(), 'Reviews']
        print(f"\nRestaurant Rating: {reviews}")

        comment = dataframe.at[dataframe['Name'].eq(title).idxmax(), 'Comments']
        if comment != "No Comments":
            print(f"\nComments: {comment}")

        rest_type = dataframe.at[dataframe['Name'].eq(title).idxmax(), 'Type']
        print(f"\nRestaurant Category: {rest_type}")

        location = dataframe.at[dataframe['Name'].eq(title).idxmax(), 'Location']
        print(f"\nThe Address: {location}")

        contact_no = dataframe.at[dataframe['Name'].eq(title).idxmax(), 'Contact Number']
        if contact_no != "Not Available":
            print(f"\nContact Details: Phone: {contact_no}")
    
    # Display an image of the restaurant (if applicable, else you can skip or handle differently)
    # For now, we will simulate displaying an image with a placeholder
    # Note: Replace '/path/to/restaurant_image.jpg' with actual image paths if available.
    image_path = '/path/to/restaurant_image.jpg'
    try:
        image = Image.open(image_path)
        image.show()  # Open image with default viewer
    except Exception as e:
        print(f"Image not found: {str(e)}")

# User input to select a restaurant
name = input("Select the restaurant you like: ")
recom(df, name)

# Collect User Feedback
print("\nRate Your Experience")
rating = input("Rate this restaurant (1-5): ")
feedback_comment = input("Your Feedback: ")

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

print("Thanks for your feedback!")
