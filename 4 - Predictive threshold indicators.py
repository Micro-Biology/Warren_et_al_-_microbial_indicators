# Databricks notebook source
# MAGIC %pip install plotly --upgrade
# MAGIC %pip install shap==0.46 xgboost==3.0.0

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Final results
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import chi2_contingency
from statsmodels.stats.proportion import proportions_ztest
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from matplotlib.colors import ListedColormap
import pickle
from scipy import stats

model = "XGB" #"RF"
n = "nitrate as n_mean"
ton = "nitrogen total oxidised as n_mean"

n_titan = pd.read_csv(f"{os.getcwd()}/Output/TITAN/{n}/titan_tax.csv").rename(columns={"Unnamed: 0":"Genus", "zenv.cp":"zenv_n", "filter":"group_n"})[["Genus","zenv_n","group_n"]]
ton_titan = pd.read_csv(f"{os.getcwd()}/Output/TITAN/{ton}/titan_tax.csv").rename(columns={"Unnamed: 0":"Genus", "zenv.cp":"zenv_ton", "filter":"group_ton"})[["Genus","zenv_ton","group_ton"]]

titan = n_titan.merge(ton_titan, on="Genus", how="outer")
titan['titan_ind'] = titan.apply(lambda row: True if row['group_n'] != 0 or row['group_ton'] != 0 else False, axis=1)

titan['titan_ind_dir'] = 0 
titan.loc[(titan['group_n'] == 1) | (titan['group_ton'] == 1), 'titan_ind_dir'] = 1
titan.loc[(titan['group_n'] == 2) | (titan['group_ton'] == 2), 'titan_ind_dir'] = 2

def get_regression(loc, chem):
    #import data
    regression = pd.read_csv(loc).rename(columns={"absolute_mean_shap_value":f"shap_{chem}","impact_direction":f"direction_{chem}"})
    #calculate shap as a percentage of total shap scores
    regression[f'shap_{chem}_perc'] = regression[f'shap_{chem}']/regression[f'shap_{chem}'].sum()*100
    #rank them based on % and discard the rank where the shap score is 0 as these have no input into the model
    regression[f'shap_{chem}_rank'] = regression[f'shap_{chem}_perc'].rank(method='dense', ascending=False)
    regression[f'shap_{chem}_rank'] = regression[f'shap_{chem}_rank'].replace(regression[f'shap_{chem}_rank'].max(), np.nan)
    regression[f'regression_{chem}_ind'] = regression[f'shap_{chem}_perc'] > regression[f'shap_{chem}_perc'].sum()/len(regression[f'shap_{chem}_perc'])
    return regression

n_regression = (get_regression(f"{os.getcwd()}/Output/{model}/{model}_nitrate as n_mean_log10_shap_importance.csv", "n"))
ton_regression = (get_regression(f"{os.getcwd()}/Output/{model}/{model}_nitrogen total oxidised as n_mean_log10_shap_importance.csv", "ton"))

regression = n_regression.merge(ton_regression, on="Genus", how="outer")
#mark as a regression index if the shap score is greater than the 
regression['regression_ind'] = regression[f'regression_n_ind'] | regression[f'regression_ton_ind']
results = titan.merge(regression, on="Genus", how="outer")

#add higher level tax annotatations
cols = results.columns.tolist()
tax = pd.read_csv(f"{os.getcwd()}/Example_data/16S_all_taxonomy.tsv", sep="\t")
higher_tax = ['Kingdom','Phylum','Class','Order','Family']
tax = tax[higher_tax + ['Genus']].drop_duplicates()
results = results.merge(tax, on="Genus", how="left")
results = results[higher_tax + cols]

#add mean relative abundance
treated_data = pd.read_csv(f"{os.getcwd()}/Example_data/filtered_asv_normal.csv")
numeric_df = treated_data.select_dtypes(include='number')
mean_values = numeric_df.mean()
results = results.merge(pd.DataFrame({'Genus': mean_values.index, 'mean relative abundance': mean_values.values}), on="Genus", how="left")
#this is to fix the fact that the relative abundances are based on the total abundance of all genera including ones removed as part of earlier filtering
results['mean relative abundance'] = (results['mean relative abundance'] / results['mean relative abundance'].sum() )*100

#add functional annotations
func = pd.read_csv(f"{os.getcwd()}/Functions/n_go_pa.csv").rename(columns={"genus":"Genus"})
results = results.merge(func, on="Genus", how="left")

results.to_csv(f"{os.getcwd()}/Output/results_summary_{model}.csv", index=False)

# COMMAND ----------

# DBTITLE 1,Table 3 Number of indicators for ML models
def get_regression(loc, chem):
    #import data
    regression = pd.read_csv(loc).rename(columns={"absolute_mean_shap_value":f"shap_{chem}","impact_direction":f"direction_{chem}"})
    #calculate shap as a percentage of total shap scores
    regression[f'shap_{chem}_perc'] = regression[f'shap_{chem}']/regression[f'shap_{chem}'].sum()*100
    #rank them based on % and discard the rank where the shap score is 0 as these have no input into the model
    regression[f'shap_{chem}_rank'] = regression[f'shap_{chem}_perc'].rank(method='dense', ascending=False)
    regression[f'shap_{chem}_rank'] = regression[f'shap_{chem}_rank'].replace(regression[f'shap_{chem}_rank'].max(), np.nan)
    regression[f'regression_{chem}_ind'] = regression[f'shap_{chem}_perc'] > regression[f'shap_{chem}_perc'].sum()/len(regression[f'shap_{chem}_perc'])
    print(f"Num {chem}_{model} indicators: {regression[f'shap_{chem}_rank'].count()}")
    print(f"    Num over equal: {len(regression[f'shap_{chem}_perc'][regression[f'shap_{chem}_perc'] > regression[f'shap_{chem}_perc'].sum()/len(regression[f'shap_{chem}_perc'])])}")
    print(f"    Max SHAP score: {regression[f'shap_{chem}_perc'].max()}")

for model in ["RF","XGB"]:
    get_regression(f"{os.getcwd()}/Output/{model}/{model}_nitrate as n_mean_log10_shap_importance.csv", "n")
    get_regression(f"{os.getcwd()}/Output/{model}/{model}_nitrogen total oxidised as n_mean_log10_shap_importance.csv", "ton")


print(f"\nNum XGB N+TON indicators: {((results['shap_n_perc'] > 0) | (results['shap_ton_perc'] > 0)).sum()}")
print(f"    Num over equal: {((results['shap_n_perc'] > results['shap_n_perc'].sum()/len(results['shap_n_perc'])) | (results['shap_ton_perc'] > results['shap_ton_perc'].sum()/len(results['shap_ton_perc']))).sum()}")
print()

# COMMAND ----------

# DBTITLE 1,Table 4 Top 20 genera and SHAP rank
tab4 = results[['Kingdom','Phylum','Class','Order','Family','Genus','shap_n_rank','shap_ton_rank']]

tab4['Taxonomy'] = ("("+
    tab4['Kingdom'].astype(str) + '; ' +
    tab4['Phylum'].astype(str) + '; ' +
    tab4['Class'].astype(str) + '; ' +
    tab4['Order'].astype(str) + '; ' +
    tab4['Family'].astype(str) + ') ' +
    tab4['Genus'].astype(str)
)

tab4 = tab4[['Taxonomy','Genus','shap_n_rank','shap_ton_rank']]

tab4['shap_n_rank'] = tab4['shap_n_rank'].fillna(tab4['shap_n_rank'].max()+1).astype(int)
tab4['shap_ton_rank'] = tab4['shap_ton_rank'].fillna(tab4['shap_ton_rank'].max()+1).astype(int)

tab4 = tab4[~((tab4['shap_n_rank'] > 20) & (tab4['shap_ton_rank'] > 20))]

display(tab4)

# COMMAND ----------

# DBTITLE 1,Table 5 num genera TITAN summary
print("Predicitve threshold RESULTS")
print()

results1 = results[results['regression_ind'] == True]

print(f"Num TITAN N indicators: {sum(results1['group_n'] == 1) + sum(results1['group_n'] == 2)}")
print(f"Num TITAN TON indicators: {sum(results1['group_ton'] == 1) + sum(results1['group_ton'] == 2)}")
print(f"Num TITAN N+TON indicators: {sum(results1['titan_ind'] == True)}")
print()

print(pd.crosstab(results1['group_n'], results1['group_ton'], margins=False).rename(columns={1:"TON Sensitive", 2:"TON Tolerant", 0: "TON Other"}, index={1:"N Sensitive", 2:"N Tolerant", 0: "N Other"}))
print()
overlap = pd.crosstab(results1['titan_ind_dir'], results1['regression_ind'], margins=False).rename(columns={False:"XGB Not-predictive", True:"XGB Predictive"}, index={1:"Titan Sensitive", 2:"Titan Tolerant", 0: "Titan Other"})
print(overlap)
print()
print((overlap/(len(results1)))*100)

# COMMAND ----------

# DBTITLE 1,Figure 5 Sankey diagram

df = results.copy()

#map labels
titan_map = {
    0: "Not Indicator",
    1: "Sensitive Indicator",
    2: "Tolerant Indicator"
    }
df['titan_label'] = df['titan_ind_dir'].map(titan_map)

xgb_map = {
    True: "Predictive Indicator",
    False: "Not Predictive Indicator"
    }
df['regression_ind_label'] = df['regression_ind'].map(xgb_map)

#over write non-threshold indicators
non_predictive_mask = (df['regression_ind'] == False)
df.loc[non_predictive_mask, 'titan_label'] = "Not Indicator"

#Create links between data
link1_fixed = df.groupby('regression_ind_label').size().reset_index(name='value')
link1_fixed.columns = ['target', 'value']
link1_fixed['source'] = "All Genera" # Set a single label for the source node
link2 = df.groupby(['regression_ind_label', 'titan_label']).size().reset_index(name='value')
link2.columns = ['source', 'target', 'value']
all_links = pd.concat([link1_fixed, link2], ignore_index=True)


all_nodes = pd.Series(
    pd.concat([all_links['source'].astype(str), all_links['target'].astype(str)]).unique()
).sort_values(ignore_index=True)

node_counts = all_links.groupby('target')['value'].sum()
total_genera = all_links[all_links['source'] == 'All Genera']['value'].sum()
node_counts.loc['All Genera'] = total_genera

node_counts.index = node_counts.index.astype(str)

#get labels
labeled_nodes = []
for label in all_nodes.tolist():
    count = node_counts.get(label)
    if count is not None:
        labeled_nodes.append(f"{label} ({int(count)})")
    else:
        labeled_nodes.append(label)

label_to_index = {label: i for i, label in enumerate(all_nodes)}

source_indices = all_links['source'].astype(str).map(label_to_index)
target_indices = all_links['target'].astype(str).map(label_to_index)
values = all_links['value'].tolist()

#plotting 
color_map = {
    "All Genera": "#543005", #darkest brown
    "Predictive Indicator": "#35978f", #green darker
    "Not Predictive Indicator": "#8c510a", #m brown
    "Not Indicator": "#bf812d", #brown
    "Sensitive Indicator": "#80cdc1", #green
    "Tolerant Indicator": "#e46d76" #light red
}
node_colors = [color_map[label] for label in all_nodes]

fig = go.Figure(data=[go.Sankey(
    node=dict(
      pad=15,
      thickness=20,
      line=dict(color="black", width=0.5),
      #label=labeled_nodes, #unhash for labels to display
      color=node_colors
    ),
    link=dict(
      source=source_indices.tolist(), 
      target=target_indices.tolist(),
      value=values,
    )
)])

fig.update_layout(
    font_size=20
)
fig.show()


# COMMAND ----------

# DBTITLE 1,Create weighted function table
genera_all_n = pd.read_csv(f"{os.getcwd()}/Functions/n_go_pa.csv")
metadata = pd.read_csv(f"{os.getcwd()}/Example_data/filtered_metadata_normal.csv").rename(columns={"ANON_ID":"Sample_ID"})
treated_data = pd.read_csv(f"{os.getcwd()}/Example_data/filtered_asv_normal.csv")

#prepare functional n_terms to drop other columns
function_matrix = genera_all_n.set_index('genus')
go_term_columns = [col for col in function_matrix.columns if '(GO:' in col]
function_matrix = function_matrix[go_term_columns]

#filter abundance df to only genera there are reference genomes for
known_taxa = set(genera_all_n['genus'].unique())
columns_to_keep = ['Sample_ID'] + [
    col for col in treated_data.columns
    if col in known_taxa
]
filtered_treated_data = treated_data[columns_to_keep]
abundance_matrix = filtered_treated_data.set_index('Sample_ID').T

#Times the two matricies together
functional_potential = abundance_matrix.T.dot(function_matrix)
functional_potential.reset_index(inplace=True)

biochem = pd.merge(functional_potential, metadata, on="Sample_ID", how="inner")
biochem = biochem.dropna(subset=['Nitrate as N_mean'])
biochem['Nitrate as N_mean_log10'] = np.log10(biochem['Nitrate as N_mean'])

# COMMAND ----------

# DBTITLE 1,Figure 6 Pearsons predicted v func
def plot_single_n_v_func(df: pd.DataFrame, ax, n: str, func: str):
    if func not in df.columns or n not in df.columns:
        ax.text(0.5, 0.5, f"Missing data for {func} or {n}", transform=ax.transAxes, ha='center')
        ax.set_title(f'{n} vs {func}')
        return None, None

    desc, go_term = func.split("(")
    go_term = go_term.replace(")", "")

    pearson_corr, p_value = stats.pearsonr(df[n], df[func])
    slope, intercept, r_value, p_val_reg, std_err = stats.linregress(df[n], df[func])
    line = slope * df[n] + intercept

    if n == "predicted_n_log10":
        colour = "#35978f"
    elif n == "predicted_ton_log10":
        colour = "#bf812d"
    else:
        colour = "pink"

    ax.scatter(df[n], df[func], alpha=0.7, color=colour)
    x_sorted = np.sort(df[n])
    line_sorted = slope * x_sorted + intercept
    ax.plot(x_sorted, line_sorted, color='#a54b53', label=f'Best Fit: $y={slope:.2f}x + {intercept:.2f}$')

    text_label = f"Pearson's $r$: {pearson_corr:.3f}\n$p$: {p_value:.2g}"
    ax.text(0.05, 0.95, text_label, transform=ax.transAxes,
            fontsize=14, verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # Set labels and title
    #ax.set_xlabel(n)
    #ax.set_ylabel(f"{desc.strip()}")
    #ax.set_title(f'{go_term}', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Return correlation values for the summary DataFrame
    return pearson_corr, p_value

def plot_multi_n_v_func(df: pd.DataFrame, func_list: list):
    n_vars = ["predicted_n_log10", "predicted_ton_log10"]
    num_rows = len(func_list) 
    num_cols = len(n_vars)

    height = (num_rows * 2.8)+1

    fig, axes = plt.subplots(num_rows, num_cols, 
                             figsize=(10, height), 
                             sharex='col', sharey=False,
                             squeeze=False) 

    ns = []
    funcs = []
    pearsons = []
    p_values = []

    for row_idx, func in enumerate(func_list):
        desc, go_term = func.split("(")
        desc = desc.strip()
        go_term_text = go_term.replace(")", "")
        
        for col_idx, n in enumerate(n_vars):
            ax = axes[row_idx, col_idx]
            pearson_corr, p_value = plot_single_n_v_func(df, ax, n, func)
            
            # Collect results
            if pearson_corr is not None:
                ns.append(n)
                funcs.append(func)
                pearsons.append(pearson_corr)
                p_values.append(p_value)
                
            # --- Labeling for Vertical Layout ---
            
            # 1. Set Title (N variable name) - Top row only
            if row_idx == 0:
                ax.set_title(n, fontsize=14)

            if desc == "nitrogen cycle metabolic process":
                desc = "nitrogen cycle other processes"
                
            # 2. Set Y-Label (Functional Term Description) - First column only
            if col_idx == 0:
                # Add the full description and GO term as the y-axis label for the row
                ax.set_ylabel(f"{desc} \n({go_term_text})", fontsize=12)
            else:
                # Remove y-axis tick labels for the second column
                ax.tick_params(axis='y', labelleft=False)

            # 3. Set X-Label (N variable name) - Bottom row only
            if row_idx == num_rows - 1:
                ax.set_xlabel(n, fontsize=12)
            else:
                # Remove x-axis tick labels for upper rows
                ax.tick_params(axis='x', labelbottom=False)
    
    # Add a main title for the figure
    fig.suptitle('Predicted Nitrogen Variables vs. Functional Potential', fontsize=18, y=0.99)
    
    # Adjust layout to prevent overlap and display the plot
    # The rect adjusts the plotting area to give space for the suptitle and xlabels
    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) 
    plt.show()

    # Return the summary DataFrame
    return pd.DataFrame({"n": ns, "func": funcs, "pearson": pearsons, "p_value": p_values})

file_path = f"{os.getcwd()}/Output/{model}/xgb_regressors.pkl"

with open(file_path, 'rb') as file:
    loaded_data = pickle.load(file)

regressors_all = loaded_data['regressors_all']
regressor_scores_all = loaded_data['regressor_scores_all']
regressor_predictions_all = loaded_data['regressor_predictions_all']

#extract predictions
predictions_n = pd.DataFrame({"Sample_ID": regressor_predictions_all["nitrate as n_mean_log10"]["sample_ids"],"predicted_n_log10": regressor_predictions_all["nitrate as n_mean_log10"]['y_pred']})
predictions_n["predicted_n"] = 10**predictions_n["predicted_n_log10"]
predictions_ton = pd.DataFrame({"Sample_ID": regressor_predictions_all["nitrogen total oxidised as n_mean_log10"]["sample_ids"],"predicted_ton_log10": regressor_predictions_all["nitrogen total oxidised as n_mean_log10"]['y_pred']})
predictions_ton["predicted_ton"] = 10**predictions_ton["predicted_ton_log10"]
predictions_n = predictions_n.merge(predictions_ton, on="Sample_ID", how="inner")
df = predictions_n.merge(functional_potential, on="Sample_ID", how="inner")


go_terms = ['nitrogen fixation(GO:0009399)',
            'denitrification pathway(GO:0019333)',
            'urea metabolic process(GO:0019627)',
            'nitrate assimilation(GO:0042128)',
            'nitrogen cycle metabolic process(GO:0071941)']

# Run the multi-plot function
summary_df = plot_multi_n_v_func(df, go_terms)

go_terms = ['denitrification pathway(GO:0019333)',
            'nitrogen cycle metabolic process(GO:0071941)']

# Run the multi-plot function
summary_df = plot_multi_n_v_func(df, go_terms)


# COMMAND ----------

# DBTITLE 1,Linking functional processes to predictive-threshold indicators
results = pd.read_csv(f"{os.getcwd()}/Output/results_summary_{model}.csv")
go_cols_to_run = [
    "nitrogen cycle metabolic process(GO:0071941)",
    "denitrification pathway(GO:0019333)",
    "nitrogen fixation(GO:0009399)",
    "urea metabolic process(GO:0019627)",
    "nitrate assimilation(GO:0042128)"
]

n_label = "Nitrate and TON"
regression_col = "regression_ind"
group_col = "titan_ind_dir"


num_plots = len(go_cols_to_run)
fig, axes = plt.subplots(
    nrows=3, 
    ncols=2, 
    figsize=(10, 15), 
    constrained_layout=True
)
axes = axes.flatten()

chi2_results = {}

for i, go_col in enumerate(go_cols_to_run):
    ax = axes[i]
    df_temp = results.copy()

    #parse GO term name
    description, go = go_col.split("(")
    go = go.replace(")", "")
    
    df_temp = df_temp[~df_temp[go_col].isna()]
    conditions ={
        'Predictive indicators Titan Sensitive': (df_temp['regression_ind'] == True) & (df_temp['titan_ind_dir'] == 1),
        'Predictive indicators Titan Tolerant': (df_temp['regression_ind'] == True) & (df_temp['titan_ind_dir'] == 2)
    }
    df_temp['Indicator status'] = np.select(conditions.values(), conditions.keys(), default="Non-indicators")
    grouped_counts = df_temp.groupby(['Indicator status', go_col]).size().reset_index(name='count')

    contingency_table = grouped_counts.pivot_table(
        index='Indicator status', 
        columns=go_col, 
        values='count', 
        fill_value=0
    )
    
    if len(contingency_table.columns) < 2 or contingency_table.shape[0] < 2:
        print(f"Skipping plot {i+1} for {go_col}: Insufficient data.")
        ax.set_title(f"Insufficient Data for {description}")
        continue 
    
    #label consistently (0: Incapable, 1: Capable)
    contingency_table.columns = ['Count_0', 'Count_1']
    group_totals = contingency_table.sum(axis=1)
    N = contingency_table.values.sum() 
    
    chi2, p_overall, dof, expected = chi2_contingency(contingency_table)
    chi2_results[go_col] = (chi2, p_overall, dof, N)
    
    grouped_proportions = contingency_table.div(group_totals, axis=0)
    plot_data = grouped_proportions
    group_labels = plot_data.index.tolist() 
    ax.bar(
        plot_data.index, 
        plot_data['Count_0'], 
        bottom=plot_data['Count_1'],
        label='Incapable', 
        color='lightgrey'
    )
    ax.bar(
        plot_data.index, 
        plot_data['Count_1'], 
        label='Capable', 
        color='Green'
    )

    for k, label in enumerate(group_labels):
        total = group_totals.loc[label]
        prop_capable = plot_data['Count_1'].loc[label]
        prop_incapable = plot_data['Count_0'].loc[label]

        ax.text(
            k, 1.02,
            f'n={int(total)}', 
            ha='center', 
            va='bottom', 
            fontsize=10, 
            fontweight='bold'
        )
        if prop_capable > 0.05: 
            ax.text(
                k, 
                prop_capable / 2, 
                f'{prop_capable*100:.1f}%', 
                ha='center', 
                va='center',
                fontsize=9,
                color='black'
            )
        if prop_incapable > 0.05: 
            y_pos_incapable = prop_capable + (prop_incapable / 2) 
            ax.text(
                k, 
                y_pos_incapable, 
                f'{prop_incapable*100:.1f}%', 
                ha='center', 
                va='center',
                fontsize=9,
                color='black'
            )
            
    #titles and labels
    ax.set_title(f'{description} (p={p_overall:.3f})', fontsize=12, pad=10)
    ax.set_ylabel('Proportion of Taxa', fontsize=10)
    ax.set_ylim(0, 1.1) 
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.tick_params(axis='x', rotation=15)

    if i == 0:
        ax.legend(title='Capability Status', loc='lower left', bbox_to_anchor=(-0.1, -0.3), fancybox=True, shadow=True)


for j in range(num_plots, len(axes)):
    fig.delaxes(axes[j])

fig.suptitle(f'Comparison of GO Term Capabilities across Indicator Status (Nitrogen Species: {n_label})', 
             fontsize=16, 
             fontweight='bold')

plt.show()


print("--- Summary of Chi-Squared P-values ---")
print(f"{'GO Term':<40} {'χ2':<10} {'dof':<5} {'N':<5} {'P-value':<10}")
print("-" * 70)
for go_col, result in chi2_results.items():
    #unpack tuple (chi2, p_overall, dof, N)
    chi2, p_val, dof, N = result
    print(f"{go_col.split('(')[0]:<40} {chi2:.5f} {dof:} {N:} {p_val:.5f}")

