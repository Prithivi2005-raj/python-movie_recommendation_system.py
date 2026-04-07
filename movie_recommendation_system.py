# movie_recommendation_system.py

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample movie dataset (you can expand this or use a CSV file)
data = {
    'title': [
        'The Dark Knight', 
        'Inception', 
        'Interstellar', 
        'The Matrix', 
        'Avengers: Endgame',
        'Iron Man',
        'The Lion King',
        'Titanic'
    ],
    'genre': [
        'Action Crime Drama',
        'Action Adventure Sci-Fi',
        'Adventure Drama Sci-Fi',
        'Action Sci-Fi',
        'Action Adventure Sci-Fi',
        'Action Sci-Fi',
        'Animation Adventure Drama',
        'Drama Romance'
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Vectorize genres
vectorizer = CountVectorizer()
genre_matrix = vectorizer.fit_transform(df['genre'])

# Compute similarity
similarity = cosine_similarity(genre_matrix)

# Recommendation function
def recommend(movie_title):
    if movie_title not in df['title'].values:
        return ["Movie not found in database. Try another title."]
    
    idx = df[df['title'] == movie_title].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    
    recommendations = []
    for i, score in scores[1:6]:  # top 5 similar movies
        recommendations.append(df.iloc[i]['title'])
    return recommendations


# Example run
if __name__ == "__main__":
    movie = input("Enter a movie name: ")
    recs = recommend(movie)
    print("\nRecommended movies:")
    for r in recs:
        print("- " + r)
