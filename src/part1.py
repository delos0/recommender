# Part 1: Conditional Probability Modeling

"""
Part 1: Conditional Probability Modeling

Estimate how likely a song is to receive a 5★ rating using conditional probabilities.
"""
import pandas as pd
import numpy as np
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent

# Go up one level to project root, then into data/
DATA_DIR = THIS_DIR.parent / "data"

# Build full paths
tracks_path = DATA_DIR / "tracks.csv"
ratings_path = DATA_DIR / "ratings.csv"
kml_session_path = DATA_DIR / "kml_session.csv"
def main():
    print("Part 1: Conditional Probability Modeling")
    print("TODO: Implement conditional probability calculations")
    print("- Load tracks.csv and ratings.csv")

    tracks = pd.read_csv(tracks_path)
    ratings = pd.read_csv(ratings_path)

    # Merge on song names
    data = ratings.merge(
        tracks,
        left_on="song_id",
        right_on="track_id",
        how="left"
    )
    # Renaming the the track_name column for disambiguation
    data = data.rename(columns={
    "track_name_x": "track_name_rating",
    "track_name_y": "track_name"
    })


    # Create a binary variable for 5-star ratings
    data["is_5star"] = (data["rating"] == 5).astype(int)

    def section(title):
        print("\n" + "=" * 70)
        print(f" {title}")
        print("=" * 70 + "\n")
    section("DATA PREVIEW (Merged Ratings + Tracks)")
    print(data[["user_id", "track_name", "primary_artist_name", "album_release_year", "rating", "is_5star"]].head())

    print("- Compute P(5★ | Artist), P(5★ | Year), P(5★ | Popularity)")
    def compute_conditional(df, group_cols, name, min_count=5):
        """
        Computes P(5* | group_cols) with Laplace smoothing
        """
        p = (
            df
            .groupby(group_cols, observed=True)["is_5star"]
            .apply(smoothed_p_5star)
            .reset_index(name=name)
        )

        counts = df.groupby(group_cols, observed=True).size().reset_index(name="n")

        p = p.merge(counts, on=group_cols)
        p = p[p["n"] >= min_count]

        return p

    # This functions is to be applied to each conditional probability space examined
    # We don't smooth the whole P(5*) since Laplace smoothing regularizes individual
    # conditional probability estimates, each of which corresponds to a distinct
    # distribution with its own sample size and uncertainty.

    def smoothed_p_5star(series, alpha=1):
        n_5 = series.sum()
        n_total = len(series)
        return (n_5 + alpha) / (n_total + 2 * alpha)


    # Computes P(5⋆ | Artist = A)
    p_5_given_artist = (
        data
        .groupby("primary_artist_name")["is_5star"]
        .apply(smoothed_p_5star)
        .reset_index(name="P_5star_given_artist")
    )

    # Add rating count for reliability
    p_5_given_artist["n_ratings"] = data.groupby("primary_artist_name").size().values

    # Filter the artist with low counts
    p_5_given_artist = p_5_given_artist[p_5_given_artist["n_ratings"] >= 5]

    p_5_given_artist = p_5_given_artist.sort_values("P_5star_given_artist", ascending=False)

    section("GLOBAL P(5★ | Artist)")
    print(p_5_given_artist.head())



    #Computes P(5⋆ | Year = Y )
    p_5_given_year = compute_conditional(
        data,
        ["album_release_year"],
        "P_5star_given_year",
        min_count=5
    )

    section("GLOBAL P(5★ | Track Release Year)")
    print(p_5_given_year.sort_values("P_5star_given_year", ascending=False).head())



    data["popularity_bin"] = pd.cut(
        data["track_popularity"],
        bins=[0, 30, 60, 100],
        labels=["Low", "Medium", "High"]
    )

    p_5_given_popularity = compute_conditional(
        data,
        ["popularity_bin"],
        "P_5star_given_popularity",
        min_count=5
    )

    section("GLOBAL P(5★ | Track Popularity)")
    print(p_5_given_popularity)


    # Computes P(5⋆∣Artist=A,Genre=G)
    p_5_artist_genre = compute_conditional(
        data,
        ["primary_artist_name", "ab_genre_dortmund_value"],
        "P_5star",
        min_count=5
    ).sort_values("P_5star", ascending=False)

    section("GLOBAL P(5★ | Artist, Genre)")
    print(p_5_artist_genre.head(10))
    


    p_5_mood_timbre = compute_conditional(
        data,
        ["ab_mood_happy_value", "ab_timbre_value"],
        "P_5star"
    ).sort_values("P_5star", ascending=False)

    section("GLOBAL P(5★ | Mood, Timbre)")
    print(p_5_mood_timbre)

    p_5_dance_party = compute_conditional(
        data,
        ["ab_danceability_value", "ab_mood_party_value"],
        "P_5star"
    ).sort_values("P_5star", ascending=False)

    section("GLOBAL P(5★ | Danceability, Party Mood)")
    print(p_5_dance_party) 


    p_5_instrumental_electronic = compute_conditional(
        data,
        ["ab_voice_instrumental_value", "ab_mood_electronic_value"],
        "P_5star"
    ).sort_values("P_5star", ascending=False)

    section("GLOBAL P(5★ | Instrumental, Electronic)")
    print(p_5_instrumental_electronic)



    feature_cols = [
        "ab_danceability_value",
        "ab_mood_happy_value",
        "ab_mood_relaxed_value",
        "ab_mood_party_value",
        "ab_mood_sad_value",
        "ab_mood_electronic_value",
        "ab_voice_instrumental_value",
        "ab_timbre_value"
    ]

    rows = []

    for feature in feature_cols:
        grouped = (
            data
            .dropna(subset=[feature])
            .groupby(feature, observed=True)["is_5star"]
            .apply(smoothed_p_5star)
        )

        counts = data[feature].value_counts()

        for category, p in grouped.items():
            if counts.get(category, 0) >= 20:
                rows.append({
                    "feature": feature,
                    "category": category,
                    "P_5star": p
                })


    feature_effects = pd.DataFrame(rows)
    feature_effects = feature_effects.sort_values("P_5star", ascending=False)

    section("GLOBAL FEATURE EFFECTS ON P(5★)")
    print(feature_effects)




    # P(5*)
    p_5_global = data["is_5star"].mean()

    # P(Artist)
    artist_prior = data["primary_artist_name"].value_counts(normalize=True)

    bayes_artist = p_5_given_artist.copy()
    bayes_artist["P_artist"] = bayes_artist["primary_artist_name"].map(artist_prior)
    bayes_artist["P_5star"] = p_5_global

    bayes_artist["P_artist_given_5star"] = (
        bayes_artist["P_5star_given_artist"] *
        bayes_artist["P_artist"] /
        bayes_artist["P_5star"]
    )

    bayes_artist = bayes_artist.sort_values("P_artist_given_5star", ascending=False)

    section("BAYESIAN INFERENCE: P(Artist | 5★)")
    print(bayes_artist.head(10))



    from glob import glob
    session_files = glob(str(DATA_DIR / "*_session.csv"))

    per_user_results = {
        "year": [],
        "popularity": [],
        "artist_genre": [],
        "mood_timbre": [],
        "dance_party": [],
        "instrumental_electronic": []
    }

    for path in session_files:
        session = pd.read_csv(path)

        merged = session.merge(
            tracks,
            left_on="song_id",
            right_on="track_id",
            how="left"
        )

        merged["is_5star"] = (merged["rating"] == 5).astype(int)

        # ---- Year ----
        per_user_results["year"].append(
            compute_conditional(
                merged,
                ["album_release_year"],
                "P_5star"
            )
            .set_index("album_release_year")["P_5star"]
        )

        # ---- Popularity ----
        merged["popularity_bin"] = pd.cut(
            merged["track_popularity"],
            bins=[0, 30, 60, 100],
            labels=["Low", "Medium", "High"]
        )

        per_user_results["popularity"].append(
            compute_conditional(
                merged,
                ["popularity_bin"],
                "P_5star",
                min_count=3
            )
            .set_index("popularity_bin")["P_5star"]
        )

        # ---- Artist & Genre ----
        per_user_results["artist_genre"].append(
            compute_conditional(
                merged,
                ["primary_artist_name", "ab_genre_dortmund_value"],
                "P_5star",
                min_count=3
            )
            .set_index(["primary_artist_name", "ab_genre_dortmund_value"])["P_5star"]
        )

        # ---- Mood & Timbre ----
        per_user_results["mood_timbre"].append(
            compute_conditional(
                merged,
                ["ab_mood_happy_value", "ab_timbre_value"],
                "P_5star"
            )
            .set_index(["ab_mood_happy_value", "ab_timbre_value"])["P_5star"]
        )

        # ---- Dance & Party ----
        per_user_results["dance_party"].append(
            compute_conditional(
                merged,
                ["ab_danceability_value", "ab_mood_party_value"],
                "P_5star"
            )
            .set_index(["ab_danceability_value", "ab_mood_party_value"])["P_5star"]
        )

        # ---- Instrumental & Electronic ----
        per_user_results["instrumental_electronic"].append(
            compute_conditional(
                merged,
                ["ab_voice_instrumental_value", "ab_mood_electronic_value"],
                "P_5star"
            )
            .set_index(["ab_voice_instrumental_value", "ab_mood_electronic_value"])["P_5star"]
        )
    group_results = {}

    for key, series_list in per_user_results.items():
        if not series_list:
            continue

        df = pd.concat(series_list, axis=1)
        group_results[key] = df.mean(axis=1).reset_index(name="P_5star_group")
    print("\n=== GROUP-LEVEL CONDITIONAL PROBABILITIES ===\n")

    for key, df in group_results.items():
        print(f"\n--- Group P(5★ | {key.replace('_', ' ').title()}) ---")
        print(df.sort_values("P_5star_group", ascending=False).head(10))

    def build_and_save_p5_lookup(
        data,
        tracks,
        p_5_given_artist,
        p_5_given_year,
        p_5_given_popularity,
        p_5_artist_genre,
        p_5_mood_timbre,
        p_5_dance_party,
        p_5_instrumental_electronic,
        feature_effects,
        out_path,
        min_track_empirical_count=10,
        smoothing_tau=5.0,
        alpha_empirical=1.0
    ):
        """
        Build a per-track P(5★) lookup DataFrame and save to CSV.

        Strategy:
        - Use empirical per-track P(5★) (Laplace-smoothed) if the track has
          at least `min_track_empirical_count` ratings, so it is reliable
        - Otherwise, aggregate conditional estimates from features / combos:
            * artist, year, popularity_bin,
            * (artist, genre), (mood, timbre), (danceability, party), (instrumental, electronic)
          Each conditional table includes a support count 'n' (except feature_effects).
        - Weighted-average feature estimates by their sample sizes (n).
        - Shrink the aggregated estimate toward the global P(5★) using `smoothing_tau`.
        - Save CSV with detailed per-track columns for transparency.
        """

        # Convert conditional tables into lookup dicts with support counts where possible.
        # Artist
        artist_map = {}
        if "primary_artist_name" in p_5_given_artist.columns:
            idx = p_5_given_artist.set_index("primary_artist_name")
            for name, row in idx.iterrows():
                artist_map[name] = (float(row["P_5star_given_artist"]), int(row.get("n_ratings", row.get("n", 0))))

        # Year
        year_map = {}
        if "album_release_year" in p_5_given_year.columns:
            idx = p_5_given_year.set_index("album_release_year")
            for year, row in idx.iterrows():
                year_map[year] = (float(row["P_5star_given_year"]), int(row.get("n", 0)))

        # Popularity bin
        pop_map = {}
        if "popularity_bin" in p_5_given_popularity.columns:
            idx = p_5_given_popularity.set_index("popularity_bin")
            for cat, row in idx.iterrows():
                pop_map[cat] = (float(row["P_5star_given_popularity"]), int(row.get("n", 0)))

        # Artist + Genre
        ag_map = {}
        if set(["primary_artist_name", "ab_genre_dortmund_value"]).issubset(p_5_artist_genre.columns):
            idx = p_5_artist_genre.set_index(["primary_artist_name", "ab_genre_dortmund_value"])
            for keys, row in idx.iterrows():
                ag_map[keys] = (float(row["P_5star"]), int(row.get("n", 0)))

        # Mood + Timbre
        mt_map = {}
        if set(["ab_mood_happy_value", "ab_timbre_value"]).issubset(p_5_mood_timbre.columns):
            idx = p_5_mood_timbre.set_index(["ab_mood_happy_value", "ab_timbre_value"])
            for keys, row in idx.iterrows():
                mt_map[keys] = (float(row["P_5star"]), int(row.get("n", 0)))

        # Dance + Party
        dp_map = {}
        if set(["ab_danceability_value", "ab_mood_party_value"]).issubset(p_5_dance_party.columns):
            idx = p_5_dance_party.set_index(["ab_danceability_value", "ab_mood_party_value"])
            for keys, row in idx.iterrows():
                dp_map[keys] = (float(row["P_5star"]), int(row.get("n", 0)))

        # Instrumental + Electronic
        ie_map = {}
        if set(["ab_voice_instrumental_value", "ab_mood_electronic_value"]).issubset(p_5_instrumental_electronic.columns):
            idx = p_5_instrumental_electronic.set_index(["ab_voice_instrumental_value", "ab_mood_electronic_value"])
            for keys, row in idx.iterrows():
                ie_map[keys] = (float(row["P_5star"]), int(row.get("n", 0)))

        # Feature-level single-feature map (feature_effects contains feature,category,P_5star)
        feat_map = {}
        if feature_effects is not None and not feature_effects.empty:
            for _, row in feature_effects.iterrows():
                feat_map[(row["feature"], row["category"])] = float(row["P_5star"])

        # Global fallback
        p_5_global = float(data["is_5star"].mean())

        # Empirical per-track stats (smoothed)
        track_stats = (
            data
            .groupby("track_id", observed=True)["is_5star"]
            .agg(n_ratings="size", n_5="sum")
            .reset_index()
        )
        # Laplace-smoothed empirical P(5★) for tracks
        track_stats["p_empirical"] = (track_stats["n_5"] + alpha_empirical) / (track_stats["n_ratings"] + 2.0 * alpha_empirical)

        # Reindex for fast lookup
        track_stats_index = track_stats.set_index("track_id").to_dict(orient="index")

        rows = []
        for _, track in tracks.iterrows():
            tid = track["track_id"]
            tname = track.get("track_name", "")
            artist = track.get("primary_artist_name", None)
            year = track.get("album_release_year", None)
            pop = track.get("track_popularity", None)
            # derive popularity bin same way as earlier
            pop_bin = None
            try:
                # Use same bins: (0,30], (30,60], (60,100]
                if pd.notna(pop):
                    if pop <= 30:
                        pop_bin = "Low"
                    elif pop <= 60:
                        pop_bin = "Medium"
                    else:
                        pop_bin = "High"
            except Exception:
                pop_bin = None

            genre = track.get("ab_genre_dortmund_value", None)
            mood_happy = track.get("ab_mood_happy_value", None)
            timbre = track.get("ab_timbre_value", None)
            dance = track.get("ab_danceability_value", None)
            mood_party = track.get("ab_mood_party_value", None)
            voice_inst = track.get("ab_voice_instrumental_value", None)
            mood_elec = track.get("ab_mood_electronic_value", None)

            # Start collecting estimates
            estimates = []
            supports = []

            # Artist
            if artist in artist_map:
                p, n = artist_map[artist]
                estimates.append(p); supports.append(n)

            # Year
            if year in year_map:
                p, n = year_map[year]
                estimates.append(p); supports.append(n)

            # Popularity bin
            if pop_bin in pop_map:
                p, n = pop_map[pop_bin]
                estimates.append(p); supports.append(n)

            # Artist + Genre
            key_ag = (artist, genre)
            if key_ag in ag_map:
                p, n = ag_map[key_ag]
                estimates.append(p); supports.append(n)

            # Mood + Timbre
            key_mt = (mood_happy, timbre)
            if key_mt in mt_map:
                p, n = mt_map[key_mt]
                estimates.append(p); supports.append(n)

            # Dance + Party
            key_dp = (dance, mood_party)
            if key_dp in dp_map:
                p, n = dp_map[key_dp]
                estimates.append(p); supports.append(n)

            # Instrumental + Electronic
            key_ie = (voice_inst, mood_elec)
            if key_ie in ie_map:
                p, n = ie_map[key_ie]
                estimates.append(p); supports.append(n)

            # Single-feature effects fallback (if any)
            # e.g., top categories for ab_mood_happy_value etc.
            # We'll include up to two single-feature estimates if present (not weighted by support)
            sf_estimates = []
            # Check a few single features that exist in feature_effects
            for feat_key in [
                ("ab_danceability_value", dance),
                ("ab_mood_happy_value", mood_happy),
                ("ab_mood_relaxed_value", track.get("ab_mood_relaxed_value")),
                ("ab_mood_party_value", mood_party),
                ("ab_mood_sad_value", track.get("ab_mood_sad_value")),
                ("ab_mood_electronic_value", mood_elec),
                ("ab_voice_instrumental_value", voice_inst),
                ("ab_timbre_value", timbre)
            ]:
                if feat_key in feat_map:
                    sf_estimates.append(feat_map[feat_key])
            # add up to 2 single-feature estimates (no support info available)
            for p_sf in sf_estimates[:2]:
                estimates.append(p_sf); supports.append(0)  # support unknown

            # Empirical track-level
            track_emp = track_stats_index.get(tid, None)
            if track_emp is not None:
                n_track = int(track_emp["n_ratings"])
                p_emp = float(track_emp["p_empirical"])
            else:
                n_track = 0
                p_emp = None

            # Decide final combined probability:
            # If we have a sufficiently large empirical sample, trust it directly.
            if n_track >= min_track_empirical_count:
                p_final = p_emp
                method = "empirical"
            else:
                # Weighted average by supports (supports with value 0 are treated as small weight)
                weights = np.array(supports, dtype=float)
                # Replace zeros (unknown support) with a small epsilon so they still contribute a little
                weights[weights == 0] = 1.0
                if weights.size > 0 and np.sum(weights) > 0:
                    p_feat = float(np.dot(estimates, weights) / np.sum(weights))
                    total_support = float(np.sum(weights))
                else:
                    p_feat = float(p_5_global)
                    total_support = 0.0

                # Shrink toward global by pseudo-count smoothing
                p_final = (total_support * p_feat + smoothing_tau * p_5_global) / (total_support + smoothing_tau)
                method = "feature_agg"

                # Optionally fuse with empirical (if small n_track exists)
                if p_emp is not None and n_track > 0:
                    # weight empirical by n_track vs features total_support
                    p_final = (n_track * p_emp + total_support * p_feat + smoothing_tau * p_5_global) / (
                        n_track + total_support + smoothing_tau
                    )
                    method = "mixed"

            # Build row record
            row = {
                "track_id": tid,
                "track_name": tname,
                "primary_artist_name": artist,
                "album_release_year": year,
                "track_popularity": pop,
                "popularity_bin": pop_bin,
                # empirical
                "n_ratings_track": n_track,
                "p_empirical": p_emp if p_emp is not None else np.nan,
                # component estimates (fill with NaN if missing)
                "p_artist": artist_map.get(artist, (np.nan, 0))[0] if artist is not None else np.nan,
                "n_artist": artist_map.get(artist, (np.nan, 0))[1] if artist is not None else 0,
                "p_year": year_map.get(year, (np.nan, 0))[0],
                "n_year": year_map.get(year, (np.nan, 0))[1],
                "p_popularity": pop_map.get(pop_bin, (np.nan, 0))[0],
                "n_popularity": pop_map.get(pop_bin, (np.nan, 0))[1],
                "p_artist_genre": ag_map.get(key_ag, (np.nan, 0))[0],
                "n_artist_genre": ag_map.get(key_ag, (np.nan, 0))[1],
                "p_mood_timbre": mt_map.get(key_mt, (np.nan, 0))[0],
                "n_mood_timbre": mt_map.get(key_mt, (np.nan, 0))[1],
                "p_dance_party": dp_map.get(key_dp, (np.nan, 0))[0],
                "n_dance_party": dp_map.get(key_dp, (np.nan, 0))[1],
                "p_instrumental_electronic": ie_map.get(key_ie, (np.nan, 0))[0],
                "n_instrumental_electronic": ie_map.get(key_ie, (np.nan, 0))[1],
                # aggregated
                "n_feature_support": int(np.sum(supports)) if len(supports) > 0 else 0,
                "p_feature_weighted": float(p_feat) if 'p_feat' in locals() else p_5_global,
                "p_final": float(p_final),
                "p_global": float(p_5_global),
                "method": method
            }
            rows.append(row)

        p5_df = pd.DataFrame(rows)

        # Ensure sensible bounds
        p5_df["p_final"] = p5_df["p_final"].clip(0.0, 1.0)

        # Save
        out_dir = Path(out_path).resolve().parent
        out_dir.mkdir(parents=True, exist_ok=True)
        p5_df.to_csv(out_path, index=False)

        return p5_df

    # ------------------------------------------------------------
    # Call the builder at the end of main() (adjust path as needed)
    # ------------------------------------------------------------
    from pathlib import Path
    OUT_PATH = DATA_DIR / "p5_lookup.csv"

    p5_df = build_and_save_p5_lookup(
        data=data,
        tracks=tracks,
        p_5_given_artist=p_5_given_artist,
        p_5_given_year=p_5_given_year,
        p_5_given_popularity=p_5_given_popularity,
        p_5_artist_genre=p_5_artist_genre,
        p_5_mood_timbre=p_5_mood_timbre,
        p_5_dance_party=p_5_dance_party,
        p_5_instrumental_electronic=p_5_instrumental_electronic,
        feature_effects=feature_effects,
        out_path=str(OUT_PATH),
        min_track_empirical_count=10,
        smoothing_tau=5.0,
        alpha_empirical=1.0
    )

    print(f"\nSaved p5 lookup to: {OUT_PATH}")




if __name__ == "__main__":
    main()
