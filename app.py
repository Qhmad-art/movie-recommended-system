"""
app.py  –  Movie Recommendation System (Streamlit)
----------------------------------------------------
A premium-styled Streamlit app that loads the pre-built
recommendation model and displays the top-5 similar movies
with TMDB poster images.
"""

import pickle

import pandas as pd
import requests
import streamlit as st

# ──────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎬 CineMatch – Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────
# Custom CSS – dark cinema theme with glassmorphism
# ──────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 40%, #24243e 100%);
}

/* ── Header ── */
.hero-title {
    text-align: center;
    font-size: 3.2rem;
    font-weight: 900;
    background: linear-gradient(135deg, #e94560 0%, #f5af19 50%, #c471ed 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.1rem;
    letter-spacing: -1px;
}

.hero-sub {
    text-align: center;
    color: #9b9bb4;
    font-size: 1.1rem;
    margin-bottom: 2.5rem;
    font-weight: 400;
}

/* ── Glass card ── */
.glass-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    margin-bottom: 2rem;
}

/* ── Movie card ── */
.movie-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    overflow: hidden;
    transition: transform 0.35s cubic-bezier(.25,.8,.25,1),
                box-shadow 0.35s cubic-bezier(.25,.8,.25,1);
    height: 100%;
}

.movie-card:hover {
    transform: translateY(-8px) scale(1.03);
    box-shadow: 0 20px 40px rgba(233, 69, 96, 0.25);
}

.movie-card img {
    width: 100%;
    aspect-ratio: 2/3;
    object-fit: cover;
    display: block;
}

.movie-card .card-body {
    padding: 1rem 1rem 1.2rem;
}

.movie-card .card-title {
    color: #ffffff;
    font-weight: 700;
    font-size: 1rem;
    line-height: 1.3;
    margin: 0;
    text-align: center;
}

/* ── Recommend button ── */
div.stButton > button {
    background: linear-gradient(135deg, #e94560, #c471ed);
    color: #fff;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 2.5rem;
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    cursor: pointer;
    transition: all 0.3s ease;
    width: 100%;
}

div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(233, 69, 96, 0.4);
}

/* ── Selectbox label ── */
.stSelectbox label {
    color: #d1d1e0 !important;
    font-weight: 600;
    font-size: 1rem;
}

/* ── Section heading ── */
.section-heading {
    color: #f0f0f8;
    font-size: 1.6rem;
    font-weight: 800;
    margin-bottom: 1.5rem;
    padding-left: 0.5rem;
    border-left: 4px solid #e94560;
}

/* ── Divider ── */
.custom-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(233,69,96,0.4), transparent);
    margin: 2rem 0;
}

/* ── Footer ── */
.footer {
    text-align: center;
    color: #6b6b80;
    font-size: 0.85rem;
    padding: 2rem 0 1rem;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────
# Load model artefacts
# ──────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    movie_dict = pickle.load(open("movie_dict.pkl", "rb"))
    movies = pd.DataFrame(movie_dict)
    similarity = pickle.load(open("similarity.pkl", "rb"))
    return movies, similarity

movies, similarity = load_data()

# ──────────────────────────────────────────────────────────────
# TMDB poster fetcher
# ──────────────────────────────────────────────────────────────
TMDB_API_KEY = "8265bd1679663a7ea12ac168da84d2e8"  # Public demo key

def fetch_poster(movie_id):
    """Fetch the poster URL for a given TMDB movie_id."""
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
        response = requests.get(url, timeout=8)
        data = response.json()
        poster_path = data.get("poster_path")
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except Exception:
        pass
    # Fallback placeholder
    return "https://via.placeholder.com/500x750?text=No+Poster"

# ──────────────────────────────────────────────────────────────
# Recommendation logic
# ──────────────────────────────────────────────────────────────

def recommend(movie):
    """Return lists of (names, posters) for the top-5 similar movies."""
    movie_index = movies[movies["title"] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1],
    )[1:6]

    names = []
    posters = []
    for idx, _ in movie_list:
        mid = movies.iloc[idx].movie_id
        names.append(movies.iloc[idx].title)
        posters.append(fetch_poster(mid))
    return names, posters

# ──────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────

# Hero
st.markdown('<h1 class="hero-title">🎬 CineMatch</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Discover movies you\'ll love — powered by content-based filtering</p>',
    unsafe_allow_html=True,
)

# Search card
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
col_select, col_btn = st.columns([4, 1], gap="large")

with col_select:
    selected_movie = st.selectbox(
        "Pick a movie you like",
        movies["title"].values,
        index=None,
        placeholder="Start typing a movie name…",
    )

with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)  # spacer to align with selectbox
    show = st.button("✨ Recommend")

st.markdown("</div>", unsafe_allow_html=True)

# Results
if show and selected_movie:
    with st.spinner("Finding the best matches for you…"):
        names, posters = recommend(selected_movie)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-heading">Movies You\'ll Love</p>', unsafe_allow_html=True)

    cols = st.columns(5, gap="medium")
    for i, col in enumerate(cols):
        with col:
            st.markdown(
                f"""
                <div class="movie-card">
                    <img src="{posters[i]}" alt="{names[i]}">
                    <div class="card-body">
                        <p class="card-title">{names[i]}</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

elif show and not selected_movie:
    st.warning("⚡ Please select a movie first!")

# Footer
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.markdown(
    '<p class="footer">Built with ❤️ using Streamlit &amp; TMDB API  •  Content-Based Recommendation Engine</p>',
    unsafe_allow_html=True,
)
