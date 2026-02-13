# src/part4.py
"""
Part 4: Monte Carlo Evaluation (fixed integration)

Integrates Part 3 recommenders (conditional_filtering_recommender and utility_based_recommender)
and runs Monte Carlo comparisons using the p5_lookup.csv produced by Part 1.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from part3 import conditional_filtering_recommender, utility_based_recommender


THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR.parent / "data"
P5_LOOKUP_PATH = DATA_DIR / "p5_lookup.csv"


def simulate_rating(p_5star, rng):
    """Simulate a user rating given Bernoulli p_5star for 5★."""
    if rng.random() < p_5star:
        return 5
    else:
        return rng.choice([1, 2, 3, 4], p=[0.1, 0.2, 0.4, 0.3])


def simulate_session(recommender_fn, user_id, k, p5_lookup, p5_global, rng):
    """
    Simulate one user session for a given recommender function.

    NOTE: recommender_fn(user_id, top_k) must return a DataFrame containing at least
          a 'track_id' column (and optionally 'track_name').
    """
    recs = recommender_fn(user_id, top_k=k)

    ratings = []
    for _, row in recs.iterrows():
        # Use track_id (not track_name) to lookup p5
        track_id = row["track_id"]
        p5 = p5_lookup.get(track_id, p5_global)
        r = simulate_rating(p5, rng)
        ratings.append(r)

    return ratings


def compute_metrics(ratings):
    ratings = np.asarray(ratings)
    hit = int(np.any(ratings == 5))
    avg_rating = float(ratings.mean()) if ratings.size > 0 else 0.0
    if hit:
        Tu = int(np.argmax(ratings == 5) + 1)
    else:
        Tu = np.inf
    return hit, avg_rating, Tu


def monte_carlo_evaluate(
    recommender_A,
    recommender_B,
    users,
    p5_lookup,
    p5_global,
    k=5,
    n_trials=2000,
    seed=42
):
    rng = np.random.default_rng(seed)
    results = {"hit_diff": [], "avg_rating_diff": [], "Tu_diff": []}

    # Use paired design: same user for both recommenders
    for _ in range(n_trials):
        user = rng.choice(users)

        ratings_A = simulate_session(recommender_A, user, k, p5_lookup, p5_global, rng)
        ratings_B = simulate_session(recommender_B, user, k, p5_lookup, p5_global, rng)

        hit_A, avg_A, Tu_A = compute_metrics(ratings_A)
        hit_B, avg_B, Tu_B = compute_metrics(ratings_B)

        results["hit_diff"].append(hit_A - hit_B)
        results["avg_rating_diff"].append(avg_A - avg_B)
        # Tu can be infinite; handle by mapping inf -> large number for differences
        Tu_A_safe = Tu_A if np.isfinite(Tu_A) else (k + 1)
        Tu_B_safe = Tu_B if np.isfinite(Tu_B) else (k + 1)
        results["Tu_diff"].append(Tu_A_safe - Tu_B_safe)

    return results


def confidence_interval(samples, alpha=0.05):
    samples = np.asarray(samples, dtype=float)
    mean = float(samples.mean())
    std = float(samples.std(ddof=1))
    n = len(samples)
    z = 1.96  # approx for 95%
    half_width = z * std / np.sqrt(n)
    return mean, mean - half_width, mean + half_width


def main():
    print("Part 4: Monte Carlo Evaluation (integrated)")

    # Load ratings and tracks so wrappers can call Part 3 functions
    ratings = pd.read_csv(DATA_DIR / "ratings.csv")
    tracks = pd.read_csv(DATA_DIR / "tracks.csv")

    # Load p5_lookup
    p5_df = pd.read_csv(P5_LOOKUP_PATH)
    # p5_lookup is keyed by track_id (ensure same dtype as tracks.track_id)
    # enforce types (try int if possible)
    try:
        p5_df["track_id"] = p5_df["track_id"].astype(tracks["track_id"].dtype)
    except Exception:
        pass
    p5_lookup = p5_df.set_index("track_id")["p_final"].to_dict()
    p5_global = float(p5_df["p_global"].iloc[0]) if "p_global" in p5_df.columns else float(ratings["rating"].eq(5).mean())

    print(f"Loaded p5_lookup for {len(p5_lookup)} tracks, global p5={p5_global:.4f}")

    # Build the users list from real data
    users = ratings["user_id"].unique()

    # -------------------------
    # Wrapper adapters (use local ratings/tracks)
    # -------------------------
    def recommender_A(user_id, top_k=5):
        # conditional_filtering_recommender signature:
        # conditional_filtering_recommender(user_id, ratings, tracks, p5_tables, top_k)
        # Build p5_tables minimally for the recommender (cheap).
        # If you have a global p5_tables saved from Part1, load & reuse that instead.
        p5_tables = {}  # we can pass empty dict; your recommender handles missing keys
        return conditional_filtering_recommender(user_id=user_id, ratings=ratings, tracks=tracks, p5_tables=p5_tables, top_k=top_k)

    def recommender_B(user_id, top_k=5):
        # utility_based_recommender signature:
        # utility_based_recommender(user_id, ratings, tracks, top_k=5, ...)
        return utility_based_recommender(user_id=user_id, ratings=ratings, tracks=tracks, top_k=top_k)

    # Run Monte Carlo
    results = monte_carlo_evaluate(
        recommender_A,
        recommender_B,
        users,
        p5_lookup,
        p5_global,
        k=5,
        n_trials=2000,
        seed=123
    )

    # Compute confidence Intervals
    hit_m, hit_lo, hit_hi = confidence_interval(results["hit_diff"])
    avg_m, avg_lo, avg_hi = confidence_interval(results["avg_rating_diff"])
    Tu_m, Tu_lo, Tu_hi = confidence_interval(results["Tu_diff"])

    print("\n=== MODEL COMPARISON (A − B) ===")
    print(f"Hit@5 diff:    {hit_m:.4f} [{hit_lo:.4f}, {hit_hi:.4f}]")
    print(f"AvgRating diff: {avg_m:.4f} [{avg_lo:.4f}, {avg_hi:.4f}]")
    print(f"Tu diff (A−B): {Tu_m:.4f} [{Tu_lo:.4f}, {Tu_hi:.4f}]")


if __name__ == "__main__":
    main()

