import numpy as np
import pandas as pd
import seaborn as sns
import kagglehub
import os
from matplotlib import pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA


PATH_TO_DATASET = os.path.join(kagglehub.dataset_download("salvatorerastelli/spotify-and-youtube"),
                               "Spotify_Youtube.csv")
DATA_PATH = os.path.join(os.getcwd(), "data")
PLOT_PATH = os.path.join(DATA_PATH, "plots")
CSV_PATH = os.path.join(DATA_PATH, "csv")

def save_to_csv(to_save, filename):
    full_filename = f'{filename}.csv'
    if not isinstance(to_save, list):
        to_save.to_csv(os.path.join(CSV_PATH, full_filename), index=True)
    else:
        to_save[0].to_csv(os.path.join(CSV_PATH, full_filename), index=True)
        for df_to_save in to_save[1:]:
            df_to_save.to_csv(os.path.join(CSV_PATH, full_filename), mode='a', index=True)
    print(f"Saved file {full_filename}")

def save_plot(plot, filename):
    full_filename = f'{filename}.png'
    plot.savefig(os.path.join(PLOT_PATH, full_filename), bbox_inches='tight', dpi=300)
    print(f"Saved plot {full_filename}")

def main():
    df = pd.read_csv(PATH_TO_DATASET)
    NUMERIC_COLUMNS = ['Danceability', 'Energy', 'Key', 'Loudness', 'Speechiness', 'Acousticness', 'Instrumentalness',
                       'Liveness', 'Valence', 'Tempo', 'Duration_ms', 'Views', 'Likes', 'Comments', 'Stream']
    CATEGORICAL_COLUMNS = ['Album_type', 'Licensed', 'official_video']
    df.drop(
        columns=['Description', 'Url_youtube', 'Url_spotify', 'Uri', 'Title', 'Channel', 'Album', 'Track', 'Unnamed: 0'],
        axis=1, inplace=True)
    df['Loudness']= MinMaxScaler(feature_range=(0, 1)).fit_transform(df[['Loudness']])
    df['Album_type'] = pd.factorize(df['Album_type'])[0]
    numeric_missing_value = df[NUMERIC_COLUMNS].isna().sum()
    print(f"Count of missing numeric values:\n{numeric_missing_value}\n")
    df.dropna(subset=NUMERIC_COLUMNS, inplace=True)
    numeric_summary = df[NUMERIC_COLUMNS].describe(percentiles=[0.05, 0.5, 0.95])
    categorical_missing_value = df[CATEGORICAL_COLUMNS].isna().sum()
    print(f"Count of missing categorical values:\n{categorical_missing_value}\n")
    df.dropna(subset=CATEGORICAL_COLUMNS, inplace=True)
    categorical_summary = df[CATEGORICAL_COLUMNS].describe()
    categorical_proportions = [
        df['Album_type'].value_counts(normalize=True),
        df['Licensed'].value_counts(normalize=True),
        df['official_video'].value_counts(normalize=True)
    ]
    save_to_csv(numeric_summary, "numeric_summary")
    save_to_csv(numeric_missing_value, "numeric_missing_value")
    save_to_csv(categorical_summary, "categorical_summary")
    save_to_csv(categorical_missing_value, "categorical_missing_value")
    save_to_csv(categorical_proportions, "categorical_proportions")
    print(f"Data "f"saved to {CSV_PATH}")
    df_clipped = df.copy()
    for column in NUMERIC_COLUMNS:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        outlier_count = df[column][(df[column] < Q1 - 1.5 * IQR) | (df[column] > Q3 + 1.5 * IQR)].count()
        print(f"Outlier count: {outlier_count} for column {column}. {outlier_count / len(df) * 100:.2f}% of the dataset")
        if outlier_count > 0:
            if outlier_count > len(df) / 5:
                print(f"Dropping column {column} as outlier count is too high")
                NUMERIC_COLUMNS.remove(column)
                df_clipped.drop(columns=column, inplace=True)
            else:
                df_clipped[column] = df_clipped[column].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)
    print(f"Numeric columns after removing outliers: {NUMERIC_COLUMNS}")
    sns.boxplot(x="official_video", y="Views", data=df_clipped)
    plt.title("Boxplot: Official Video vs Views")
    plt.xlabel("Official Video")
    plt.ylabel("Number of views")
    save_plot(plt, "boxplot_official_video_vs_views")
    plt.close()
    sns.violinplot(x="official_video", y="Views", data=df_clipped)
    plt.title("Violin plot: Official Video vs Views")
    plt.xlabel("Official Video")
    plt.ylabel("Number of views")
    save_plot(plt, "violinplot_official_video_vs_views")
    plt.close()
    sns.boxplot(x="official_video", y="Stream", data=df_clipped)
    plt.title("Boxplot: Official Video vs Streams")
    plt.xlabel("Official Video")
    plt.ylabel("Number of streams")
    save_plot(plt, "boxplot_official_video_vs_stream")
    plt.close()
    sns.violinplot(x="official_video", y="Stream", data=df_clipped)
    plt.title("Violin plot: Official Video vs Streams")
    plt.xlabel("Official Video")
    plt.ylabel("Number of streams")
    save_plot(plt, "violinplot_official_video_vs_stream")
    plt.close()
    sns.boxplot(x="Album_type", y="Stream", data=df_clipped)
    plt.title("Boxplot: Album Type vs Streams")
    plt.xlabel("Album Type")
    plt.ylabel("Number of streams")
    save_plot(plt, "boxplot_album_type_vs_stream")
    plt.close()
    sns.violinplot(x="Album_type", y="Stream", data=df_clipped)
    plt.title("Violin plot: Album Type vs Streams")
    plt.xlabel("Album Type")
    plt.ylabel("Number of streams")
    save_plot(plt, "violinplot_album_type_vs_stream")
    plt.close()
    sns.barplot(x="official_video", y="Stream", data=df_clipped, errorbar=("pi", 75))
    plt.title("Barplot: Official Video vs Streams")
    plt.xlabel("Official Video")
    plt.ylabel("Number of streams")
    save_plot(plt, "error_barplot_official_video_vs_streams")
    plt.close()
    sns.barplot(x="official_video", y="Views", data=df_clipped, errorbar=("pi", 75))
    plt.title("Barplot: Official Video vs Views")
    plt.xlabel("Official Video")
    plt.ylabel("Number of views")
    save_plot(plt, "error_barplot_official_video_vs_views")
    plt.close()
    sns.histplot(data=df_clipped, x="Views", bins=50)
    plt.title("Distribution of Views")
    save_plot(plt, "histplot_views")
    plt.close()
    sns.histplot(data=df_clipped, x="Stream", bins=50)
    plt.title("Distribution of Streams")
    save_plot(plt, "histplot_streams")
    plt.close()
    sns.histplot(data=df_clipped, x="Energy", bins=50)
    plt.title("Distribution of Energy")
    save_plot(plt, "histplot_energy")
    plt.close()
    sns.histplot(data=df_clipped, x="Speechiness", bins=50)
    plt.title("Distribution of Speechiness")
    save_plot(plt, "histplot_speechiness")
    plt.close()
    sns.histplot(data=df_clipped, x="Loudness", bins=50)
    plt.title("Distribution of Loudness")
    save_plot(plt, "histplot_loudness")
    plt.close()
    sns.histplot(data=df_clipped, x="Duration_ms", bins=50)
    plt.title("Distribution of Duration")
    save_plot(plt, "histplot_duration")
    plt.close()
    sns.histplot(data=df_clipped, x="Stream", hue="Album_type", bins=50, multiple="stack")
    plt.title("Distribution of Streams")
    save_plot(plt, "histplot_streams_hue")
    plt.close()
    sns.histplot(data=df_clipped, x="Stream", hue="Album_type", bins=50, multiple="stack", stat="density",
                 common_norm=False)
    plt.title("Normalized distribution of Streams")
    save_plot(plt, "histplot_streams_hue_normalized")
    plt.close()
    sns.histplot(data=df_clipped, x="Views", hue="Album_type", bins=50, multiple="stack", stat="density", common_norm=False)
    plt.title("Normalized distribution of Views")
    save_plot(plt, "histplot_views_hue_normalized")
    plt.close()
    sns.histplot(data=df_clipped, x="Key", bins=50)
    plt.title("Distribution of Key")
    save_plot(plt, "histplot_key")
    plt.close()
    sns.histplot(data=df_clipped, x="Key", bins=50, hue="Album_type", multiple="stack", stat="density", common_norm=False)
    plt.title("Normalized distribution of Key")
    save_plot(plt, "histplot_key_hue")
    plt.close()
    plt.figure(figsize=(8, 8))
    sns.heatmap(df_clipped[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS].corr(), annot=True, fmt=".2f", cmap="PiYG",
                annot_kws={"size": 8})
    plt.title("Data correlation heatmap")
    save_plot(plt, "correlation_heatmap")
    plt.close()
    METRICS_TO_REDUCE = ["Views", "Likes", "Comments"]
    for metric in METRICS_TO_REDUCE:
        NUMERIC_COLUMNS.remove(metric)

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_clipped[METRICS_TO_REDUCE])

    cov_matrix = np.cov(df_scaled, rowvar=False)
    e_values, e_vectors = np.linalg.eig(cov_matrix)

    idx = np.argsort(e_values)[::-1]
    e_values = e_values[idx]
    e_vectors = e_vectors[:, idx]
    explained_variance_ratio = e_values / np.sum(e_values)

    print("Eigenvalues:\n", e_values)
    print("Eigenvectors:\n", e_vectors)
    print("Explained Variance Ratios:", explained_variance_ratio)

    number_of_components = 0
    total_ratio = 0
    for e_v,er in zip(e_values,explained_variance_ratio):
        if e_v > 1 and total_ratio < 0.90:
            number_of_components += 1
    print(f"Number of components to keep: {number_of_components}")
    df_eigenanalysys = pd.concat([
        pd.DataFrame({
            'Eigenvalue': e_values
        }),
        pd.DataFrame(
            e_vectors,
            columns=['Eigenvector_1', 'Eigenvector_2', 'Eigenvector_3']
        ),
        pd.DataFrame({"Number of Components to Keep": [number_of_components] * len(e_values)}),
        pd.DataFrame({
            'Explained Variance Ratio': explained_variance_ratio
        })
    ], axis=1)
    save_to_csv(df_eigenanalysys, "eigenanalysis")
    NUMERIC_COLUMNS.append("YT_performance")

    pca = PCA(n_components=1)
    principal_component = pca.fit_transform(df_scaled)
    df_clipped["YT_performance"] = principal_component
    df["YT_performance"] = principal_component
    plt.figure(figsize=(8, 8))
    sns.heatmap(df_clipped[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS].corr(), annot=True, fmt=".2f", cmap="PiYG",
                annot_kws={"size": 8})
    plt.title("Updated Data correlation heatmap")
    save_plot(plt, "correlation_heatmap_updated")
    plt.close()
    SY_ols = sm.OLS.from_formula("Stream ~ YT_performance", data=df_clipped).fit()
    SY_rlm = sm.RLM.from_formula("Stream ~ YT_performance", data=df_clipped).fit()
    plt.scatter(df_clipped["YT_performance"], df_clipped["Stream"], label="Data", alpha=0.05)
    plt.plot(df_clipped["YT_performance"], SY_rlm.fittedvalues, color="green", label="Robust Fit")
    plt.plot(df_clipped["YT_performance"], SY_ols.fittedvalues, color="red", label="OLS Fit", linestyle="--")
    pred_ols = SY_ols.get_prediction(df_clipped)
    SY_pl = pred_ols.summary_frame()["obs_ci_lower"]
    SY_pu = pred_ols.summary_frame()["obs_ci_upper"]
    plt.plot(df_clipped["YT_performance"], SY_pu, "r--")
    plt.plot(df_clipped["YT_performance"], SY_pl, "r--")
    plt.xlabel("YT_performance")
    plt.ylabel("Streams")
    plt.legend()
    plt.ylim(bottom=0)
    save_plot(plt, "correlation_Spotify-YT_performance")
    plt.close()
    VD_ols = sm.OLS.from_formula("Valence ~ Danceability", data=df_clipped).fit()
    VD_rlm = sm.RLM.from_formula("Valence ~ Danceability", data=df_clipped).fit()
    plt.scatter(df_clipped["Danceability"], df_clipped["Valence"], label="Data", alpha=0.05)
    plt.plot(df_clipped["Danceability"], VD_rlm.fittedvalues, color="green", label="Robust Fit")
    plt.plot(df_clipped["Danceability"], VD_ols.fittedvalues, color="red", label="OLS Fit", linestyle="--")
    pred_ols = VD_ols.get_prediction(df_clipped)
    VD_pl = pred_ols.summary_frame()["obs_ci_lower"]
    VD_pu = pred_ols.summary_frame()["obs_ci_upper"]
    plt.plot(df_clipped["Danceability"], VD_pu, "r--")
    plt.plot(df_clipped["Danceability"], VD_pl, "r--")
    plt.xlabel("Danceability")
    plt.ylabel("Valence")
    plt.ylim(0, 1.1)
    plt.legend()
    save_plot(plt, "correlation_Valence-Danceability")
    plt.close()
    EL_ols = sm.OLS.from_formula("Energy ~ Loudness", data=df_clipped).fit()
    EL_rlm = sm.RLM.from_formula("Energy ~ Loudness", data=df_clipped).fit()
    plt.scatter(df_clipped["Loudness"], df_clipped["Energy"], label="Data", alpha=0.05)
    plt.plot(df_clipped["Loudness"], EL_rlm.fittedvalues, color="green", label="Robust Fit")
    plt.plot(df_clipped["Loudness"], EL_ols.fittedvalues, color="red", label="OLS Fit", linestyle="--")
    pred_ols = EL_ols.get_prediction(df_clipped)
    EL_pl = pred_ols.summary_frame()["obs_ci_lower"]
    EL_pu = pred_ols.summary_frame()["obs_ci_upper"]
    plt.plot(df_clipped["Loudness"], EL_pu, "r--")
    plt.plot(df_clipped["Loudness"], EL_pl, "r--")
    plt.xlabel("Loudness")
    plt.ylabel("Energy")
    plt.ylim(0, 1.1)
    plt.legend()
    save_plot(plt, "correlation_Energy-Loudness")
    plt.close()
    df_top_artists = df.copy()
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_top_artists["YT_performance"] = scaler.fit_transform(df_top_artists[["YT_performance"]])
    df_top_artists["Stream"] = scaler.fit_transform(df_top_artists[["Stream"]])
    df_top_artists["Performance"] = 0.7 * df_top_artists["Stream"] + 0.3 * df_top_artists["YT_performance"]
    df_top_artists.drop(columns=["YT_performance", "Stream"], inplace=True)
    TOP_ARTISTS_NUMERIC_COLUMNS = NUMERIC_COLUMNS.copy()
    TOP_ARTISTS_NUMERIC_COLUMNS.remove("YT_performance")
    TOP_ARTISTS_NUMERIC_COLUMNS.remove("Stream")
    TOP_ARTISTS_NUMERIC_COLUMNS.append("Performance")
    artist_performance: dict[str, int] = {}
    for artist in df_top_artists["Artist"].unique():
        artist_performance[artist] = sum(df_top_artists[df_top_artists["Artist"] == artist]["Performance"])
    sorted_performance = sorted(artist_performance.items(), key=lambda x: x[1], reverse=True)
    sorted_performance = sorted_performance[:100]
    df_artist_performance = pd.DataFrame(sorted_performance, columns=["Artist", "Performance"])
    plt.figure(figsize=(15, 18))
    sns.scatterplot(x="Performance", y="Artist", data=df_artist_performance)
    for i in range(len(df_artist_performance)):
        plt.text(
            x=df_artist_performance["Performance"][i]+ 0.02,
            y=i,
            s=df_artist_performance["Artist"][i],
            fontsize=6,
            color="black"
        )
    plt.title("Top 100 artists by Performance metric")
    save_plot(plt, "top_artists_performance")
    plt.close()
    plt.figure(figsize=(8, 8))
    sns.heatmap(df_top_artists[TOP_ARTISTS_NUMERIC_COLUMNS + CATEGORICAL_COLUMNS].corr(), annot=True, fmt=".2f", cmap="PiYG",
                annot_kws={"size": 8})
    plt.title("Top artists data correlation heatmap")
    save_plot(plt, "top_artists_correlation_heatmap")
    plt.close()

if __name__ == "__main__":
    main()