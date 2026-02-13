# Part 3: Recommender Design
"""
Part 3: Recommender Design

Design and implement two different recommendation algorithms for the music system.
"""

from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-12


# -----------------------------------------------------------------------------
# Part 1 utilities: P(5star | feature=value) with smoothing
# -----------------------------------------------------------------------------
def smoothed_p_5star(is_5star_series: pd.Series, alpha: float = 1.0) -> float:
    """
    Laplace smoothing for a Bernoulli event (is_5star).
    """
    n_5 = float(is_5star_series.sum())
    n_total = float(len(is_5star_series))
    return (n_5 + alpha) / (n_total + 2.0 * alpha)


def compute_p5_given_feature(
    merged: pd.DataFrame,
    feature: str,
    alpha: float = 1.0,
    min_count: int = 5,
) -> pd.DataFrame:
    """
    Build a table of conditional probabilities for one feature:
        P(5star | feature=value)

    Inputs:
    - merged: ratings joined with tracks metadata (must include 'is_5star' and 'feature')
    - feature: column name for which we compute conditionals
    - alpha: smoothing strength
    - min_count: drop rare feature values with too few samples

    Output:
    - DataFrame with columns:
        feature, P5_given_<feature>, n_ratings
    """
    tmp = merged.dropna(subset=[feature]).copy()

    grouped = (
        tmp.groupby(feature, observed=True)["is_5star"]
        .apply(lambda s: smoothed_p_5star(s, alpha=alpha))
        .reset_index(name=f"P5_given_{feature}")
    )

    counts = tmp[feature].value_counts().rename("n_ratings").reset_index()
    counts = counts.rename(columns={"index": feature})

    out = grouped.merge(counts, on=feature, how="left")
    out = out[out["n_ratings"] >= min_count].copy()
    return out


def build_p5_tables(
    ratings: pd.DataFrame,
    tracks: pd.DataFrame,
    features=("primary_artist_name", "ab_genre_dortmund_value"),
    alpha: float = 1.0,
    min_count: int = 5,
) -> dict:
    """
    Build p5_tables for selected features.
    Intended structure:
        p5_tables[feature][value] = P(5star | feature=value)

    NOTE: This keeps your original logic as-is. (It returns DataFrames per feature,
    not dicts. If you need dicts, do it at call-site like you already do elsewhere.)
    """
    merged = ratings.merge(
        tracks,
        left_on="song_id",
        right_on="track_id",
        how="left",
    )

    merged["is_5star"] = (merged["rating"] == 5).astype(int)

    p5_tables = {}
    for f in features:
        if f not in merged.columns:
            p5_tables[f] = {}
            continue

        # Keep original behavior (even though compute_p5_given_feature does not accept observed=)
        p5_tables[f] = compute_p5_given_feature(
            merged,
            feature=f,
            alpha=alpha,
            min_count=min_count,
        )

    return p5_tables


# -----------------------------------------------------------------------------
# Part 2 utilities: Tu and patience normalization
# -----------------------------------------------------------------------------
def build_Tu_table(ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Tu per user:
        Tu = first round_idx where the user gives a 5-star rating.
    """
    Tu_table = (
        ratings.loc[ratings["rating"].eq(5), ["user_id", "round_idx"]]
        .groupby("user_id", as_index=False)["round_idx"]
        .min()
        .rename(columns={"round_idx": "Tu"})
    )
    return Tu_table


def build_patience_norm_fn(Tu_table: pd.DataFrame):
    """
    Convert Tu into a normalized patience score in [0, 1].

    - patience_norm(user) ~ 0 means low Tu  -> fast 5star -> more selective / impatient
    - patience_norm(user) ~ 1 means high Tu -> slow 5star -> more patient
    """
    Tu = Tu_table["Tu"].dropna().astype(float).values

    if len(Tu) == 0:

        def patience_norm(_user_id):
            return 0.5

        return patience_norm

    low, high = np.percentile(Tu, 10), np.percentile(Tu, 90)
    tu_map = Tu_table.set_index("user_id")["Tu"].to_dict()

    def patience_norm(user_id):
        tu = float(tu_map.get(user_id, np.median(Tu)))
        return float(np.clip((tu - low) / (high - low + EPS), 0.0, 1.0))

    return patience_norm


def build_popularity_bin(track_popularity: pd.Series) -> pd.Series:
    """
    (0,30]   -> Low
    (30,60]  -> Medium
    (60,100] -> High
    """
    return pd.cut(
        track_popularity,
        bins=[0, 30, 60, 100],
        labels=["Low", "Medium", "High"],
        include_lowest=False,
    )


# -----------------------------------------------------------------------------
# Profile construction
# -----------------------------------------------------------------------------
def build_user_profile(user_history_merged: pd.DataFrame, like_threshold: int = 4):
    """
    Build a simple "taste profile" for the user based on their liked songs.
    """
    liked = user_history_merged[user_history_merged["rating"] >= like_threshold].copy()
    if liked.empty:
        liked = user_history_merged.copy()

    def top_values(col, k=3):
        if col not in liked.columns:
            return []
        vc = liked[col].dropna().value_counts()
        return vc.head(k).index.tolist()

    profile = {
        "top_artists": top_values("primary_artist_name", k=5),
        "top_genres": top_values("ab_genre_dortmund_value", k=5),
        "top_years": top_values("album_release_year", k=5),
        "top_pop_bins": top_values("popularity_bin", k=2),
    }

    optional_features = [
        "ab_danceability_value",
        "ab_mood_happy_value",
        "ab_mood_relaxed_value",
        "ab_mood_party_value",
        "ab_mood_sad_value",
        "ab_mood_electronic_value",
        "ab_voice_instrumental_value",
        "ab_timbre_value",
    ]
    for f in optional_features:
        profile[f] = top_values(f, k=2)

    return profile


# -----------------------------------------------------------------------------
# Model A: Simple Conditional Filtering (artist/genre priority)
# -----------------------------------------------------------------------------
def conditional_filtering_recommender(
    user_id: str,
    ratings: pd.DataFrame,
    tracks: pd.DataFrame,
    p5_tables: dict,  # feature -> {value: P5}
    top_k: int = 5,
):
    """
    Simple conditional filtering recommender.

    Philosophy:
    - Hard filter based on user's 5star preferences
    - Prioritize artist and genre
    - Light ranking using Part 1 conditional probabilities
    """
    merged = ratings.merge(tracks, left_on="song_id", right_on="track_id", how="left")

    user_hist = merged[merged["user_id"] == user_id].copy()
    liked = user_hist[user_hist["rating"] == 5].copy()
    print(user_hist.head(5))

    if liked.empty:
        print("User did not like any track.")
        return pd.DataFrame(columns=["track_id", "track_name", "reason"])

    top_artists = (
        liked["primary_artist_name"].dropna().value_counts().head(5).index.tolist()
    )
    top_genres = (
        liked["ab_genre_dortmund_value"].dropna().value_counts().head(5).index.tolist()
    )

    seen = set(user_hist["track_id"].dropna().unique())
    candidates = tracks[~tracks["track_id"].isin(seen)].copy()

    def passes_filter(row):
        return (
            row.get("primary_artist_name") in top_artists
            or row.get("ab_genre_dortmund_value") in top_genres
        )

    candidates = candidates[candidates.apply(passes_filter, axis=1)].copy()
    if candidates.empty:
        print("There are no recommendations for this user.")
        return pd.DataFrame(columns=["track_id", "track_name", "reason"])

    def score(row):
        """
        Simple priority-based score:
        - artist match > genre match
        - break ties using Part 1 P(5star | feature)
        """
        s = 0.0

        artist = row.get("primary_artist_name")
        genre = row.get("ab_genre_dortmund_value")

        if artist in top_artists:
            s += 2.0
            s += p5_tables.get("primary_artist_name", {}).get(artist, 0.0)

        if genre in top_genres:
            s += 1.0
            s += p5_tables.get("ab_genre_dortmund_value", {}).get(genre, 0.0)

        return s

    candidates["score"] = candidates.apply(score, axis=1)

    recs = candidates.sort_values("score", ascending=False).head(top_k).copy()

    def reason(row):
        reasons = []
        if row.get("primary_artist_name") in top_artists:
            reasons.append("artist match")
        if row.get("ab_genre_dortmund_value") in top_genres:
            reasons.append("genre match")
        return ", ".join(reasons)

    recs["reason"] = recs.apply(reason, axis=1)

    return recs[["track_id", "track_name", "reason"]].reset_index(drop=True)

# -----------------------------------------------------------------------------
# Model B: Utility-Based Sampling (probabilistic; exploration/exploitation via Tu)
# -----------------------------------------------------------------------------
def utility_based_recommender(
    user_id: str,
    ratings: pd.DataFrame,
    tracks: pd.DataFrame,
    top_k: int = 5,
    like_threshold: int = 4,
    alpha_smooth: float = 1.0,
    min_count: int = 5,
    seed: int = 0,
    temp_min: float = 0.35,
    temp_max: float = 1.50,
):
    """
    Utility-Based Sampling recommender.

    - Assign each candidate track i a user-specific utility U(i), proportional to an
      estimated hit probability P(5star | i) derived from Part 1 conditional models.
    - Sample tracks proportionally to U(i), which blends:
        * exploitation (high-probability hits have higher utility)
        * exploration (sampling can still pick lower-utility items sometimes)
    - Incorporate user patience (with Tu):
        * patient users (higher expected Tu) -> higher temperature -> more exploratory sampling
        * impatient users -> lower temperature -> more exploitative sampling

    Output:
    - DataFrame with sampled recommendations [track_id, track_name, utility, prob]
      (prob is the sampling probability under the final distribution).
    """

    rng = np.random.default_rng(seed)

    # Join ratings with track metadata so we can compute feature-based utilities
    merged = ratings.merge(tracks, left_on="song_id", right_on="track_id", how="left")
    merged["is_5star"] = (merged["rating"] == 5).astype(int)

    # derived feature of popularity
    if "track_popularity" in merged.columns:
        merged["popularity_bin"] = build_popularity_bin(merged["track_popularity"])
    else:
        merged["popularity_bin"] = np.nan

    # compute patience from Tu and map it to a sampling temperature
    Tu_table = build_Tu_table(ratings)
    patience_norm = build_patience_norm_fn(Tu_table)
    patience = float(patience_norm(user_id))  # 0 .. 1; higher => more patient

    # Temperature controls exploration:
    # - low temp: exploit top items
    # - high temp: explore more
    temp = temp_min + patience * (temp_max - temp_min)

    # Build conditional probability lookup tables:
    #    p5_tables[feature][value] = P(5star | feature=value)
    feature_list = [
        "primary_artist_name",
        "album_release_year",
        "popularity_bin",
        "ab_genre_dortmund_value",
        "ab_danceability_value",
        "ab_mood_happy_value",
        "ab_mood_relaxed_value",
        "ab_mood_party_value",
        "ab_mood_sad_value",
        "ab_mood_electronic_value",
        "ab_voice_instrumental_value",
        "ab_timbre_value",
    ]

    p5_tables = {}
    for f in feature_list:
        if f in merged.columns:
            p5_tables[f] = (
                compute_p5_given_feature(
                    merged, feature=f, alpha=alpha_smooth, min_count=min_count
                )
                .set_index(f)[f"P5_given_{f}"]
                .to_dict()
            )
        else:
            p5_tables[f] = {}

    # Global fallback probability when feature value is unseen/rare
    p5_global = float(merged["is_5star"].mean())

    def get_p5(f, val):
        return float(p5_tables.get(f, {}).get(val, p5_global))

    # Build candidate set: all unseen tracks (broad pool for exploration)
    user_hist = merged[merged["user_id"] == user_id].copy()
    seen = set(user_hist["track_id"].dropna().unique().tolist())

    candidates = tracks[~tracks["track_id"].isin(seen)].copy()
    if candidates.empty:
        return pd.DataFrame(columns=["track_id", "track_name", "utility", "prob"])

    # Add derived popularity_bin to candidates if possible
    if "track_popularity" in candidates.columns:
        candidates["popularity_bin"] = build_popularity_bin(candidates["track_popularity"])

    # Compute utility U(i) ==> P(5star | i) using conditional models from Part 1.
    #
    # We combine feature conditionals as independent evidence in log-space:
    #   log U(i) = sum_f w_f * log(P(5star | feature=value))
    #
    # Utility is then:
    #   U(i) = exp(log U(i))
    #
    # Notes:
    # - U is a positive number (not necessarily <= 1).
    # - We will sample proportional to U after temperature scaling.
    w_base = {
        "primary_artist_name": 1.2,
        "ab_genre_dortmund_value": 1.0,
        "album_release_year": 0.5,
        "popularity_bin": 0.6,
        "ab_timbre_value": 0.3,
        "ab_danceability_value": 0.3,
        "ab_mood_party_value": 0.2,
        "ab_mood_happy_value": 0.2,
        "ab_mood_relaxed_value": 0.1,
        "ab_mood_sad_value": 0.1,
        "ab_mood_electronic_value": 0.2,
        "ab_voice_instrumental_value": 0.2,
    }

    # Small popularity stabilizer
    pop_prior_weight = 0.1

    def log_utility_row(row) -> float:
        lu = 0.0

        # Feature-based evidence for 5star
        for f, wf in w_base.items():
            if f not in row:
                continue
            val = row.get(f, None)
            p5 = get_p5(f, val)
            lu += wf * np.log(p5 + EPS)

        # Mild popularity prior to reduce very risky recommendations
        pop = float(row.get("track_popularity", 0.0) or 0.0)
        lu += pop_prior_weight * np.log(pop + 1.0)

        return float(lu)

    candidates["logU"] = candidates.apply(log_utility_row, axis=1)

    # Convert logU -> positive utility
    # Shift by max for numerical stability (does not change proportional sampling)
    max_logU = float(candidates["logU"].max())
    candidates["utility"] = np.exp(candidates["logU"] - max_logU)

    # safety against all-zero utilities
    candidates["utility"] = candidates["utility"].clip(lower=EPS)

    # Sample proportionally to utility with temperature
    #
    # We sample using:
    #   prob(i) <==> exp(logU(i) / temp)
    z = (candidates["logU"].values - max_logU) / (temp + EPS)
    probs = np.exp(z)
    probs = probs / probs.sum()

    k = min(top_k, len(candidates))
    chosen_idx = rng.choice(len(candidates), size=k, replace=False, p=probs)

    recs = candidates.iloc[chosen_idx].copy()
    recs["prob"] = probs[chosen_idx]

    # sort final output by utility
    recs = recs.sort_values("utility", ascending=False)

    return recs[["track_id", "track_name", "utility", "prob"]].reset_index(drop=True)


# -----------------------------------------------------------------------------
def main():
    """
    Part 3: Recommender Design (Conditional Filtering)

    High-level idea:
    - Build global conditional probabilities from Part 1:
        P(5star | feature=value) for several track features (artist, genre, mood bins, etc.).
    - Build a user "taste profile" from their own high ratings.
    - Use Part 2 "patience" (Tu = round of first 5star) to control how strict filtering is:
        * small Tu  -> user is selective / impatient -> stricter filtering (less exploration)
        * large Tu  -> user is patient              -> looser filtering (more exploration)
    - Score unseen tracks using a weighted sum of log-probabilities and return top 5.
    """
    print("Part 3: Recommender Design")
    print("TODO: Implement recommendation models")
    print("- Design conditional filtering approach")

    DATA_DIR = Path(__file__).resolve().parent.parent / "data"
    ratings = pd.read_csv(DATA_DIR / "ratings.csv")
    tracks = pd.read_csv(DATA_DIR / "tracks.csv")

    user_id = ratings["user_id"].iloc[1]
    user_id = "U0002"

    recs = utility_based_recommender(user_id, ratings, tracks, top_k=10)
    print(recs)

    # users = ratings["user_id"].unique()[:50]
    # all_scores = []
    # for u in users:
    #     recs_u = recommend_conditional_filtering(u, ratings, tracks, top_k=200)
    #     all_scores.append(recs_u["score"].values)
    # all_scores = np.concatenate(all_scores)
    # print(all_scores.min(), all_scores.max())

    p5_tables = build_p5_tables(
        ratings,
        tracks,
        features=["primary_artist_name", "ab_genre_dortmund_value"],  # add more later
        alpha=1.0,
        min_count=5,
    )

    print(conditional_filtering_recommender(user_id, ratings, tracks, p5_tables, top_k=5))

    print("- Design popularity-biased approach")
    print("- Test models on sample data")
    print("- Prepare for Tune Duel competition")


if __name__ == "__main__":
    main()

