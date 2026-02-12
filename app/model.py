import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from app.utils import preprocess_text
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "spotify_millsongdata.csv")

df = pd.read_csv(DATA_PATH)

# Reduce dataset size for deployment
df = df.sample(5000, random_state=42)

# Drop unnecessary columns
df = df.drop("link", axis=1).reset_index(drop=True)

df["cleaned_text"] = df["text"].apply(preprocess_text)


tfidf_vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf_vectorizer.fit_transform(df["cleaned_text"])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend_songs(song_name, top_n=5):
    matches = df[df["song"].str.lower().str.contains(song_name.lower(), na=False)]

    if matches.empty:
        return None

    idx = matches.index[0]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]

    song_indices = [i[0] for i in sim_scores]

    return df[["artist", "song"]].iloc[song_indices].to_dict(orient="records")

