import os, time, warnings
import random
import numpy as np
import torch


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    # For additional reproducibility on GPU:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


from master_bo2 import run_BO_scenario, identify_pareto_front
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


warnings.filterwarnings("ignore")




# Define parameter grids for the scenarios
model_types = ["logistic","random_forest","decision_tree","nn", 'random_forest2']
# Use the union of candidate generation methods from both cells:
cand_gens = ["dycors_org", "apex", "sobol"]
# Candidate selection methods
cand_sels = ["pareto","ehvi","parego"]
# Example batch sizes
batch_sizes = [1,3,5]  

cand_size = 2500
total_budget = 150  
output_dir = "./fairpilot_results"

# Number of repetitions per scenario
N = 10  # or more if desired

# Containers for aggregated results and AUC histories
all_results = []      # For aggregated Pareto metrics (e.g. avg error, fairness, duration)
auc_histories = []    # For aggregated AUC histories per scenario

# Create output folders for plots if they don't exist
plots_dir = os.path.join(output_dir, "plots")
auc_plots_dir = os.path.join(output_dir, "auc_plots")
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(auc_plots_dir, exist_ok=True)

# Loop over each scenario
for model in model_types:
    for cand_gen in cand_gens:
        for cand_sel in cand_sels:
            for batch in batch_sizes:
                scenario_reps = []
                scenario_auc_histories = []  # store auc_history from each rep
                print(f"Running: model={model}, cand_gen={cand_gen}, cand_sel={cand_sel}, batch_size={batch}")
                for rep in range(N):
                    # Run the BO scenario (this returns a dict with final_X, final_Y, duration, pareto_idx, and auc_history)
                    res = run_BO_scenario(model, cand_gen, cand_sel, batch, cand_size, total_budget, output_dir)
                    final_Y = res["final_Y"]
                    avg_error = np.mean(final_Y[:, 0])
                    avg_fairness = np.mean(final_Y[:, 1])
                    scenario_reps.append({
                        "rep": rep,
                        "avg_error": avg_error,
                        "avg_fairness": avg_fairness,
                        "duration": res["duration"]
                    })
                    # Store AUC history (assumed to be returned in res)
                    if "auc_history" in res:
                        scenario_auc_histories.append(res["auc_history"])
                    else:
                        scenario_auc_histories.append([])
                

                    # Save evaluated points and AUC DataFrames immediately for this repetition
                    points_filename = os.path.join(output_dir, f"points_{model}_{cand_gen}_{cand_sel}_batch{batch}_rep{rep}.csv")
                    res["points_df"].to_csv(points_filename, index=False)
                    print(f"Saved evaluated points (with Pareto labels) to {points_filename}")
                    
                    auc_filename = os.path.join(output_dir, f"auc_{model}_{cand_gen}_{cand_sel}_batch{batch}_rep{rep}.csv")
                    res["auc_df"].to_csv(auc_filename, index=False)
                    print(f"Saved AUC history to {auc_filename}")

                # Aggregate Pareto metrics for this scenario 
                df_rep = pd.DataFrame(scenario_reps)
                agg_metrics = df_rep.mean(numeric_only=True).to_dict()
                agg_metrics.update({
                    "model": model,
                    "cand_gen": cand_gen,
                    "cand_sel": cand_sel,
                    "batch_size": batch
                })
                all_results.append(agg_metrics)
                
                # Aggregate AUC histories (if N > 1, average elementwise)
                if len(scenario_auc_histories) > 0 and len(scenario_auc_histories[0]) > 0:
                    avg_auc_history = np.mean(scenario_auc_histories, axis=0)
                else:
                    avg_auc_history = []
                auc_histories.append({
                    "model": model,
                    "cand_gen": cand_gen,
                    "cand_sel": cand_sel,
                    "batch_size": batch,
                    "auc_history": avg_auc_history
                })
                
                # # Save individual repetition results to CSV for this scenario
                # df_rep.to_csv(os.path.join(output_dir, f"results_{model}_{cand_gen}_{cand_sel}_batch{batch}.csv"), index=False)
                
                # --- Produce a scatter plot for one representative run (rep 0) ---
                # (This plot shows the Pareto frontier from the final evaluated points)
                rep0_final_Y = res["final_Y"]
                pareto_idx = identify_pareto_front(rep0_final_Y)
                pareto_Y = rep0_final_Y[pareto_idx]
                
                plt.figure(figsize=(10, 7))
                sns.scatterplot(x=rep0_final_Y[:, 0], y=rep0_final_Y[:, 1], label='All Points', alpha=0.6)
                sns.scatterplot(x=pareto_Y[:, 0], y=pareto_Y[:, 1], color='red', label='Pareto Front', alpha=0.8)
                plt.xlabel('Error Rate (1 - F1 Score)')
                plt.ylabel('Statistical Parity Difference')
                plt.title(f'Pareto Front - {model.capitalize()}, {cand_gen}, {cand_sel}, batch_size={batch}')
                plt.xlim(0.3, 0.5)
                plt.ylim(0.05, 0.3)
                plt.xticks(np.linspace(0.3, 0.5, 11))
                plt.yticks(np.linspace(0.05, 0.3, 11))
                plt.legend()
                plt.grid(True)
                scatter_filename = os.path.join(plots_dir, f"Pareto_Front_{model}_{cand_gen}_{cand_sel}_batch{batch}.png")
                plt.savefig(scatter_filename)
                plt.close()
                print(f"Scatter plot saved to {scatter_filename}")
                
                # --- Produce an AUC evolution plot for this scenario (if available) ---
                if len(avg_auc_history) > 0:
                    iterations = np.arange(1, len(avg_auc_history) + 1)
                    plt.figure(figsize=(10, 6))
                    sns.lineplot(x=iterations, y=avg_auc_history, marker="o")
                    plt.xlabel("Iteration")
                    plt.ylabel("Pareto Frontier AUC")
                    plt.title(f"AUC Evolution - {model.capitalize()}, {cand_gen}, {cand_sel}, batch_size={batch}")
                    # Use fixed axis limits: assume x-axis from 0 to total_budget (or maximum iterations) and y-axis from 0 to 1
                    plt.xlim(0, total_budget)
                    plt.ylim(0, 0.01)
                    plt.xticks(np.linspace(0, total_budget, num=11))
                    plt.yticks(np.linspace(0, 0.01, num=11))
                    plt.grid(True)
                    auc_filename = os.path.join(auc_plots_dir, f"AUC_{model}_{cand_gen}_{cand_sel}_batch{batch}.png")
                    plt.savefig(auc_filename)
                    plt.close()
                    print(f"AUC plot saved to {auc_filename}")
                else:
                    print(f"No AUC history available for scenario: {model}, {cand_gen}, {cand_sel}, batch_size={batch}")

# Create aggregated results DataFrames and save
df_all = pd.DataFrame(all_results)
df_all.to_csv(os.path.join(output_dir, "aggregated_results.csv"), index=False)
print("Aggregated Pareto frontier results:")
display(df_all)

auc_data = []
for item in auc_histories:
    if len(item["auc_history"]) > 0:
        final_auc = item["auc_history"][-1]
        num_iter = len(item["auc_history"])
    else:
        final_auc = np.nan
        num_iter = 0
    auc_data.append({
        "model": item["model"],
        "cand_gen": item["cand_gen"],
        "cand_sel": item["cand_sel"],
        "batch_size": item["batch_size"],
        "final_auc": final_auc,
        "num_iterations": num_iter
    })
df_auc = pd.DataFrame(auc_data)
df_auc.to_csv(os.path.join(output_dir, "aggregated_auc_results.csv"), index=False)
print("Aggregated AUC results:")
display(df_auc)
