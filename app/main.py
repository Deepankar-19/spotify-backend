from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from app.model import recommend_songs

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for development only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Spotify ML Recommender API Running"}

@app.get("/recommend/{song_name}")
def get_recommendations(song_name: str):
    results = recommend_songs(song_name)
    if results is None:
        return {"error": "Song not found"}
    return {"recommendations": results}
