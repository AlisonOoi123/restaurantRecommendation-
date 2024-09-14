import streamlit as st
import seaborn as sns
import itertools
import pandas as pd
import matplotlib.pyplot as plt

# Set page layout and sidebar
st.set_page_config(layout='wide', initial_sidebar_state='expanded')
st.sidebar.image('Data/App_icon.png')

# Main page title
st.markdown("""
# Welcome to our restaurant insights page!
Discover fascinating trends and data-driven analysis of the culinary landscape. From popular cuisine types to the best states and cities for food lovers, we've got you covered.
""")

# Load data
df = pd.read_csv("./Data/TripAdvisor_RestauarantRecommendation.csv")
df = df.drop(['Contact Number', 'Trip_advisor Url', 'Menu'], axis=1)
df = df.drop([1744, 2866])
df = df.reset_index(drop=True)
df.Comments = df.Comments.fillna('')
df.Type = df.Type.fillna(df.Type.value_counts().index[0])

# Sidebar and main content layout
col1, col2 = st.columns([1, 2])

# Visualization for popular cuisine types
types = list(itertools.chain(*[t.split(",") for t in df.Type if isinstance(t, str)]))
types_counts = pd.Series(types).value_counts()[:10]
fig, ax = plt.subplots()
fig.set_facecolor('#121212') 
ax.set_facecolor('#121212')

pie = types_counts.plot(kind='pie', shadow=True, cmap=plt.get_cmap('Spectral'), ax=ax)
for text in pie.texts:
    text.set_color('white')
  
    
ax.set_ylabel('')
ax.tick_params(colors='white')
ax.title.set_color('white')


with col2:
    st.markdown("""
    ### 10 Most Popular Types of Cuisines
    Ever wondered what cuisines people are loving the most? Dive into our interactive visualization to explore the top 10 most popular types of cuisines based on our data. From Italian to Japanese, uncover the culinary delights that are capturing diners' hearts.
    """)
    plt.tight_layout()

    st.pyplot(fig)

# Visualization for number of restaurants per state
df['State'] = [i.split(",")[-1].split(" ")[1] for i in df.Location]
df = df.drop(df[df.State == ''].index[0])
state_counts = df['State'].value_counts()
fig, ax = plt.subplots()
sns.barplot(x=state_counts.index, y=state_counts, palette="rocket", ax=ax)

fig.set_facecolor('#121212') 
ax.set_facecolor('#121212')
ax.set_ylabel('No of Restaurants', color='white')
ax.set_xlabel('State', color='white')
ax.tick_params(color='white')
ax.title.set_color('white')
plt.xticks(rotation=45, color='white')  
plt.yticks(rotation=45, color='white')  




for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
plt.gcf().set_size_inches(7, 5)
with col1:
    st.markdown("""
    ## No of Restaurants per State
    Curious about which states boast the highest number of restaurants? Our bar chart breaks down the restaurant scene across different states, giving you insights into where culinary diversity thrives.
    """)
    plt.tight_layout()

    st.pyplot(fig)

# Convert 'Reviews' column to float, handle non-numeric values by coercing errors to NaN
df['Reviews'] = pd.to_numeric(df['Reviews'].str.split(" ").str[0], errors='coerce')

# Convert 'No of Reviews' column to int, handle non-numeric values by coercing errors to NaN
df['No of Reviews'] = pd.to_numeric(df['No of Reviews'].str.split(" ").str[0].replace(",", ""), errors='coerce')

# Drop rows where 'Reviews' or 'No of Reviews' are NaN, since we can't use them for calculations
df = df.dropna(subset=['Reviews', 'No of Reviews'])

# Calculate the weighted ratings
df['weighted_ratings'] = df['Reviews'] * df['No of Reviews']

# Group by state to calculate maximum weighted ratings
state_avg_ratings = df.groupby('State')['weighted_ratings'].max().reset_index()

# Plot the state with the best restaurant
with col1:
    st.markdown("""
    ## State with the Best Restaurant
    """)
    fig, ax = plt.subplots()
    fig.set_facecolor('#121212') 
    ax.set_facecolor('#121212')
    sns.barplot(x='State', y="weighted_ratings", data=state_avg_ratings, palette="PuOr", ax=ax)
    ax.set_ylabel('Weighted Average Ratings', color='white')
    ax.set_xlabel('State', color='white')
    ax.tick_params(colors='white')
    ax.title.set_color('white')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)


# Best state for food
state_total_ratings = df.groupby('State')['weighted_ratings'].sum().reset_index()
with col2:
    st.markdown("""
    ## Best State For Food
    Looking for the ultimate foodie destination? Explore our findings on the best state for food based on total weighted ratings. Whether you're craving gourmet cuisine or down-home cooking, this state promises a gastronomic adventure.
    """)
    fig, ax = plt.subplots()
    fig.set_facecolor('#121212') 
    ax.set_facecolor('#121212')
    sns.barplot(x='State', y="weighted_ratings", data=state_total_ratings, palette="mako", ax=ax)
    ax.set_ylabel('Total Weighted Ratings', color='white')
    ax.set_xlabel('State', color='white')
    ax.tick_params(colors='white')
    ax.title.set_color('white')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

# Top 5 cities for food
df['City'] = [",".join(i.split(",")[:-1]) for i in df.Location]
city_total_ratings = df.groupby('City')['weighted_ratings'].sum().reset_index().sort_values(by='weighted_ratings', ascending=False).head(5)
with col2:
    st.markdown("""
    ## Top 5 Cities For Food
    Discover the top 5 cities that are culinary hotspots. Our analysis reveals the cities where food lovers can indulge in the finest dining experiences, from bustling metropolises to charming culinary gems.
    """)
    fig, ax = plt.subplots()
    fig.set_facecolor('#121212') 
    ax.set_facecolor('#121212')
    sns.barplot(x='City', y="weighted_ratings", data=city_total_ratings, palette="flare", ax=ax)
    ax.set_ylabel('Total Weighted Ratings', color='white')
    ax.set_xlabel('City', color='white')
    ax.tick_params(colors='white')
    ax.title.set_color('white')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
