# Part 2: User Variability Modeling

"""
Part 2: User Variability Modeling

Model how many recommendations it takes for users to rate a song 5★ using 
geometric and Beta-geometric distributions.
"""

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from pathlib import Path
import matplotlib.pyplot as plt


THIS_DIR = Path(__file__).resolve().parent

# Go up one level to project root, then into data/
DATA_DIR = THIS_DIR.parent / "data"

# Build full paths
tracks_path = DATA_DIR / "tracks.csv"
ratings_path = DATA_DIR / "ratings.csv"

def main():
    print("Part 2: User Variability Modeling")
    print("TODO: Implement user variability analysis")
    print("- Calculate time-to-5★ for each user")

    ratings = pd.read_csv(ratings_path)
    ratings = ratings.sort_values(["user_id", "round_idx"], kind="mergesort") # sort just to make sure

    print("\n\n ------------- Ratings table ----------------")
    print(ratings[["user_id", "round_idx", "track_name", "rating"]].head())


    print("\n\n ------------- Time-to-5★ for each user ----------------")
    # caclulate time to 5 star per user, dropping users with no 5 star
    Tu_table = (
        ratings.loc[ratings["rating"].eq(5), ["user_id", "round_idx"]]
        .groupby("user_id", as_index=False)["round_idx"]
        .min()
        .rename(columns={"round_idx": "Tu"})
    )
    print(Tu_table.head())

    print("\n\n- Fit geometric distribution")

    Tu_values = Tu_table["Tu"]
    Tu_mean = Tu_values.mean()
    p = 1.0 / Tu_mean

    print("\n\nSample mean of Tu distribution =", Tu_mean)
    print("Probability p of being a 5 star hit for any user =", p)

    # real pmf = pmf computed from ratings.csv
    pmf = (
        Tu_values
        .value_counts(normalize=True)
        .sort_index()
        .rename("real_pmf")
        .reset_index()
        .rename(columns={"index": "Tu"})
    )

    # geometric distribution is computed with the appropriate formula
    pmf["geometric_pmf"] = (
            (1 - p) ** (pmf["Tu"] - 1) * p
    )

    print("\n\n ------------- PMF table ----------------")
    print(pmf)

    plt.figure()

    # real distribution (bars)
    plt.bar(
        pmf["Tu"],
        pmf["real_pmf"],
        alpha=0.6,
        label="Real PMF"
    )

    # geometric model (line + points)
    plt.plot(
        pmf["Tu"],
        pmf["geometric_pmf"],
        marker="o",
        label="Geometric model"
    )

    plt.xlabel(r"$T_u$ (round of first 5★)")
    plt.ylabel("Probability")
    plt.title(r"Real vs Geometric Distribution of $T_u$")
    plt.legend()
    plt.tight_layout()
    plt.savefig("tu_geometric_fit.png", dpi=300)
    plt.close()



    print("- Fit Beta-geometric distribution")

    import numpy as np
    import math

    def betaln(a, b):
        """log Beta(a,b) computed via log-gamma."""
        return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)

    def log_pmf_beta_geometric(t, alpha, beta):
        """
        Log PMF of Beta-Geometric:
          P(T=t) = B(alpha+1, beta+t-1) / B(alpha, beta),   t=1,2,3,...

        Returns log P(T=t). Supports scalar or array-like t.
        """

        # Vectorize betaln with loop
        base = betaln(alpha, beta)
        out = np.empty_like(t, dtype=float)
        for i, ti in np.ndenumerate(t):
            out[i] = betaln(alpha + 1.0, beta + float(ti) - 1.0) - base
        return out

    def neg_log_likelihood(Tu, alpha, beta):
        Tu = np.asarray(Tu)
        return -float(np.sum(log_pmf_beta_geometric(Tu, alpha, beta)))

    def fit_beta_geometric_mle(Tu, init_alpha=1.0, init_beta=1.0,
                               step0=1.0, shrink=0.5, max_iter=200, tol=1e-6):
        """
        MLE using coordinate search in log-space:
        optimize over x=log(alpha), y=log(beta) to keep alpha,beta>0

        Returns alpha_hat, beta_hat, nll_hat.
        """
        Tu = np.asarray(Tu, dtype=float)

        x = math.log(init_alpha)
        y = math.log(init_beta)
        step = step0

        def nll_xy(xy):
            a = math.exp(xy[0])
            b = math.exp(xy[1])
            return neg_log_likelihood(Tu, a, b)

        best = nll_xy((x, y))

        for _ in range(max_iter):
            improved = False

            # Try moving in +/- directions for x then y
            for dx, dy in [(step, 0.0), (-step, 0.0), (0.0, step), (0.0, -step)]:
                cand = nll_xy((x + dx, y + dy))
                if cand < best:
                    best = cand
                    x += dx
                    y += dy
                    improved = True

            if not improved:
                step *= shrink
                if step < tol:
                    break

        alpha_hat = math.exp(x)
        beta_hat = math.exp(y)
        return alpha_hat, beta_hat, best

    Tu = Tu_values.to_numpy()
    a_hat, b_hat, nll = fit_beta_geometric_mle(Tu, init_alpha=1.0, init_beta=5.0)
    print("\n\nValue of alpha", a_hat)
    print("Value of beta", b_hat)
    print("Negative log likelihood value", nll)

    # draw the  beta geometric distribution
    t_min, t_max = int(Tu.min()), int(Tu.max())
    t_grid = np.arange(t_min, t_max + 1)

    # Compute beta-geometric model PMF
    log_p = log_pmf_beta_geometric(t_grid, a_hat, b_hat)
    p_model = np.exp(log_p)

    # Optional: renormalize over the plotted range (only needed if you want the curve
    # to sum to 1 over [t_min, t_max] rather than over 1..infinity)
    # p_model = p_model / p_model.sum()

    plt.figure()

    plt.bar(
        pmf["Tu"],
        pmf["real_pmf"],
        alpha=0.6,
        label="Real PMF"
    )

    plt.plot(
        t_grid,
        p_model,
        marker="o",
        label=f"Beta-geometric (α={a_hat:.3g}, β={b_hat:.3g})"
    )

    plt.xlabel(r"$T_u$ (round of first 5★)")
    plt.ylabel("Probability")
    plt.title(r"Real PMF vs Beta-geometric Distribution of $T_u$")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("- Perform hypothesis testing between user groups")

    def group_users_by_popularity_and_test(ratings, Tu_table, top_pct=0.5):
        """
        Group users into 'popular' vs 'less_popular' based on the track that gave their first 5★,
        then perform Mann-Whitney U test to compare Tu between groups.

        Returns:
            results dict with descriptive stats, test results, and effect sizes.
        """

        if "track_name" in ratings.columns:
            track_col = "track_name"
        else:
            raise ValueError("ratings DataFrame has no 'track_name' column.")

        # Find the track that caused each user's first 5-star
        # Merge Tu_table (user, Tu) with ratings where rating==5 & round_idx == Tu
        first5 = ratings.loc[ratings["rating"].eq(5)].merge(
            Tu_table, left_on=["user_id", "round_idx"], right_on=["user_id", "Tu"], how="inner"
        )
        # Keep only user and track
        first5 = first5[["user_id", track_col]].drop_duplicates(subset=["user_id"])

        # Compute track popularity (number of unique users who rated the track)
        # Make the users unique to avoid exploitation against multiple ratings by same user,
        # although this have probably been handled in the aggreagation of the rating.csv 
        track_pop = ratings.groupby(track_col)["user_id"].nunique().rename("popularity").reset_index()

        # attach popularity to first5s, merging on track_name
        first5 = first5.merge(track_pop, on=track_col, how="left")

        # If some tracks have NaN popularity, drop those users, for robustness and clarity
        first5 = first5.dropna(subset=["popularity"])

        # Define the split and group the users
        # Here we split at the median as the default(= top_pct=0.5):
        split = track_pop["popularity"].quantile(1.0 - top_pct) if top_pct != 0.5 else track_pop["popularity"].median()
        first5["popular"] = first5["popularity"] >= split

        # Merge back to Tu_table to get Tu along with popularity label ---
        data = Tu_table.merge(first5[["user_id", "popular", "popularity"]], on="user_id", how="inner")
        # Drop any potential NaNs again
        data = data.dropna(subset=["Tu", "popular"])

        # Group arrays
        A = data.loc[data["popular"], "Tu"].values.astype(int)   # popular users
        B = data.loc[~data["popular"], "Tu"].values.astype(int)  # less popular users

        nA, nB = len(A), len(B)
        if nA == 0 or nB == 0:
            raise ValueError("One of the groups is empty after splitting. There is a problem.")

        # Meaningful descriptive statistics
        def describe(arr):
            return {
                "n": len(arr),
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "std": float(np.std(arr, ddof=1)),
                "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
                "min": int(np.min(arr)),
                "max": int(np.max(arr))
            }
        descA = describe(A)
        descB = describe(B)

        # Visual checks (hist, boxplot, ECDF)
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.hist([A, B], bins=range(1, int(max(A.max(), B.max()))+2), label=['popular','less_popular'], alpha=0.6)
        plt.xlabel("Tu (round of first 5★)")
        plt.ylabel("Count")
        plt.legend()
        plt.title("Histogram of Tu by group")

        plt.subplot(1,2,2)
        plt.boxplot([A, B], tick_labels=['popular', 'less_popular'], showfliers=False)
        plt.ylabel("Tu")
        plt.title("Boxplot of Tu by group")
        plt.tight_layout()
        plt.savefig("boxplot_Tu_by_group.png", dpi=300)
        plt.close()

        def ecdf(x):
            x = np.sort(x)
            y = np.arange(1, len(x)+1) / len(x)
            return x, y
        xA, yA = ecdf(A)
        xB, yB = ecdf(B)
        plt.step(xA, yA, where="post", label=f'popular (n={nA})')
        plt.step(xB, yB, where="post", label=f'less_popular (n={nB})')
        plt.xlabel("Tu")
        plt.ylabel("ECDF")
        plt.legend()
        plt.title("ECDF of Tu by group")
        plt.savefig("ecdf_Tu_by_group.png", dpi=300)
        plt.close()


        # Run the test: Mann-Whitney U (nonparametric)
        u_stat, p_value = mannwhitneyu(A, B, alternative='two-sided')

        # Effect sizes (nonparametric, interpretable)
        # Common-language effect (probability A < B)
        # P(A < B) = (#pairs where Ai < Bj) / (nA * nB)
        # P(A > B) similarly. Cliff's delta = P(A < B) - P(A > B)
        pairs_less = np.sum(A[:, None] < B)
        pairs_greater = np.sum(A[:, None] > B)
        total_pairs = nA * nB
        p_A_less_B = pairs_less / total_pairs
        p_A_greater_B = pairs_greater / total_pairs
        cliffs_delta = p_A_less_B - p_A_greater_B

        # Also compute mean difference
        mean_diff = descA["mean"] - descB["mean"]
        median_diff = descA["median"] - descB["median"]

        results = {
            "nA": nA, "nB": nB,
            "descA": descA, "descB": descB,
            "mannwhitney": {"U": u_stat, "pvalue": p_value},
            "effect_sizes": {
                "P(A < B)": p_A_less_B,
                "P(A > B)": p_A_greater_B,
                "Cliffs_delta": cliffs_delta,
                "mean_diff": mean_diff,
                "median_diff": median_diff
            },
            "split_cutoff": split,
            "data": data
        }

        return results

    results = group_users_by_popularity_and_test(ratings, Tu_table)
    import pprint
    pprint.pprint(results['descA'])
    pprint.pprint(results['descB'])
    print("Mann-Whitney U:", results['mannwhitney']['U'])
    print("p-value:", results['mannwhitney']['pvalue'])
    print("P(A < B):", results['effect_sizes']['P(A < B)'])
    print("Mean-diff:", results['effect_sizes']['mean_diff'])
    print("Cliff's delta:", results['effect_sizes']['Cliffs_delta'])
if __name__ == "__main__":
    main()
