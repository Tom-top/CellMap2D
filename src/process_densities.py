import os
import numpy as np
import pandas as pd
import tifffile
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")


def load_kde(path):
    return tifffile.imread(path).astype(np.float32)


def rank_subclasses_difference(test_kde, subclass_kde_dir, subclass_threshold=0.05):
    results = []

    for subclass_file in sorted(os.listdir(subclass_kde_dir)):
        subclass_path = os.path.join(subclass_kde_dir, subclass_file)
        subclass_kde = load_kde(subclass_path)
        subclass_name = subclass_file.replace("_approx_kde.tif", "")

        subclass_mask = subclass_kde > (subclass_threshold * subclass_kde.max())
        if subclass_mask.sum() == 0:
            continue

        diff = (subclass_kde - test_kde) * subclass_mask
        mse = np.mean(diff[subclass_mask] ** 2)
        sum_abs_diff = np.sum(np.abs(diff[subclass_mask]))
        false_positive_mask = subclass_mask & (test_kde < (0.01 * test_kde.max()))
        penalty = np.sum(subclass_kde[false_positive_mask]) / (np.sum(subclass_kde[subclass_mask]) + 1e-8)

        final_score = mse + penalty
        match_score = 1 / (final_score + 1e-8)  # Inverted score for interpretability

        results.append({
            'subclass': subclass_name,
            'mse': mse,
            'sum_abs_diff': sum_abs_diff,
            'penalty': penalty,
            'final_score': final_score,
            'match_score': match_score
        })

    results_df = pd.DataFrame(results)
    return results_df.sort_values(by='match_score', ascending=False)  # Higher is better


def process_sample(sample_id, data_dir, subclass_kde_dir, subclass_threshold):
    print(f"\n🎯 Ranking subclasses for sample: {sample_id}")

    sample_dir = os.path.join(data_dir, f"{sample_id}.zarr")
    test_kde_path = os.path.join(sample_dir, f"{sample_id}_kde.tif")
    if not os.path.exists(test_kde_path):
        print(f"❌ Missing KDE for {sample_id}, skipping.")
        return

    test_kde = load_kde(test_kde_path)
    results_df = rank_subclasses_difference(test_kde, subclass_kde_dir, subclass_threshold)

    # Save results
    result_csv_path = os.path.join(sample_dir, f"{sample_id}_subclass_ranking.csv")
    results_df.to_csv(result_csv_path, index=False)
    print(f"✅ Ranking saved to: {result_csv_path}")

    # Plot Top 20 by match score
    top_20 = results_df.head(20)

    plt.figure(figsize=(8, 10))
    sns.barplot(
        x="match_score",
        y="subclass",
        data=top_20,
        palette="viridis"
    )

    plt.xlabel("Match Score", fontsize=12)
    plt.ylabel("Subclass", fontsize=12)
    plt.title(f"Top 20 Subclass Matches for Sample {sample_id}", fontsize=14)
    plt.tight_layout()

    plot_path = os.path.join(sample_dir, f"{sample_id}_top20_subclass_matches.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"📌 Bar plot saved to: {plot_path}")
