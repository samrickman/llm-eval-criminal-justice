source("./r_utils.R")


run_glm_models(
    "./logs/",
    get_exact_match_accuracy_dt,
    "Accuracy metric: exact match",
    "./plots/accuracy_distribution_exact_match.png",
    "./model_results_csv/exact_match_basic_comparison.csv",
    "./model_results_csv/exact_match_model_results.csv"
)

run_glm_models(
    "./cosine_similarity/",
    get_cosine_accuracy_dt,
    "Accuracy metric: cosine similarity",
    "./plots/accuracy_distribution_cosine_similarity.png",
    "./model_results_csv/cosine_similarity_basic_comparison.csv",
    "./model_results_csv/cosine_similarity_model_results.csv"
)
