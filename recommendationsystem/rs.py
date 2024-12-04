import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample dataset
data = {
    'MovieID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Title': [
        'The Matrix', 'Inception', 'The Godfather', 'Avengers: Endgame', 'Titanic',
        'Blade Runner', 'Interstellar', 'Pulp Fiction', 'The Dark Knight', 'Forrest Gump'
    ],
    'Genre': [
        'Sci-Fi, Action', 'Sci-Fi, Thriller', 'Crime, Drama', 'Action, Adventure', 'Romance, Drama',
        'Sci-Fi, Drama', 'Sci-Fi, Adventure', 'Crime, Drama', 'Action, Crime', 'Drama, Romance'
    ]
}

# Convert dataset to DataFrame
df = pd.DataFrame(data)

# Step 1: Feature extraction
tfidf = TfidfVectorizer(stop_words='english')
genre_matrix = tfidf.fit_transform(df['Genre'])

# Step 2: Compute similarity
cosine_sim = cosine_similarity(genre_matrix, genre_matrix)

# Function to recommend movies based on genre
def recommend_movies_by_genre(selected_genre, num_recommendations=3):
    # Filter movies by selected genre
    filtered_movies = df[df['Genre'].str.contains(selected_genre, case=False)]
    
    if filtered_movies.empty:
        return f"No movies found for the genre: {selected_genre}"
    
    # Get similarity scores for filtered movies
    indices = filtered_movies.index.tolist()
    sim_scores = list(enumerate(cosine_sim[indices].mean(axis=0)))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get top N recommendations
    sim_scores = sim_scores[:num_recommendations]
    movie_indices = [i[0] for i in sim_scores]
    
    return df['Title'].iloc[movie_indices]

# User input for genre
user_selected_genre = input("Please enter a genre (e.g., Sci-Fi, Action, Drama , Romance , Thriller , Adventure , Crime): ")
print(f"Recommendations for genre '{user_selected_genre}':")
print(recommend_movies_by_genre(user_selected_genre))