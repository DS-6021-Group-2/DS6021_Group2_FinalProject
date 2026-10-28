"""Small TMDB client for scripts and notebooks; credentials stay in .env."""

import os
from pathlib import Path

import requests
from dotenv import load_dotenv


def tmdb_get(endpoint, **params):
    """Fetch a v3 endpoint, e.g. tmdb_get('search/movie', query='Inception')."""
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    token = os.getenv("token", "").strip()
    key = os.getenv("TMDB_API_KEY", "").strip()
    if not token and not key:
        raise RuntimeError("Add TMDB_API_KEY or token to the project's .env file.")

    endpoint = endpoint.lstrip("/")
    if not endpoint or ":" in endpoint or "?" in endpoint or "#" in endpoint or ".." in endpoint:
        raise ValueError("Use a TMDB endpoint path such as 'movie/popular'.")
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    else:
        params["api_key"] = key
    try:
        response = requests.get(
            f"https://api.themoviedb.org/3/{endpoint}",
            params=params, headers=headers, timeout=30,
        )
    except requests.RequestException:
        # Request exceptions can include the URL and its API key.
        raise RuntimeError("TMDB request failed. Check your network connection.") from None
    if not response.ok:
        raise RuntimeError(
            f"TMDB returned HTTP {response.status_code}. "
            "Check your credentials for 401/403; retry later for 429."
        )
    return response.json()


if __name__ == "__main__":
    try:
        movie = tmdb_get("movie/550")
        print(f"TMDB connection successful: {movie['title']} ({movie['release_date']})")
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
