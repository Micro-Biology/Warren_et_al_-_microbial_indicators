# Databricks notebook source

import pandas as pd
import numpy as np
import os

model = "XGB" #"RF"
n = "nitrate as n_mean"
ton = "nitrogen total oxidised as n_mean"

wkdir = os. getcwd()

n_titan = pd.read_csv(f"{os. getcwd()}/Output/TITAN/{n}/titan_tax.csv").rename(columns={"Unnamed: 0":"Genus", "zenv.cp":"zenv_n", "filter":"group_n"})[["Genus","zenv_n","group_n"]]
ton_titan = pd.read_csv(f"{os. getcwd()}/Output/TITAN/{ton}/titan_tax.csv").rename(columns={"Unnamed: 0":"Genus", "zenv.cp":"zenv_ton", "filter":"group_ton"})[["Genus","zenv_ton","group_ton"]]

titan = n_titan.merge(ton_titan, on="Genus", how="outer")
titan['titan_ind'] = titan.apply(lambda row: True if row['group_n'] != 0 or row['group_ton'] != 0 else False, axis=1)

titan['titan_ind_dir'] = 0 
titan.loc[(titan['group_n'] == 1) | (titan['group_ton'] == 1), 'titan_ind_dir'] = 1
titan.loc[(titan['group_n'] == 2) | (titan['group_ton'] == 2), 'titan_ind_dir'] = 2

def get_regression(loc, chem):
    regression = pd.read_csv(loc).rename(columns={"absolute_mean_shap_value":f"shap_{chem}","impact_direction":f"direction_{chem}"})
    regression[f'shap_{chem}_perc'] = regression[f'shap_{chem}']/regression[f'shap_{chem}'].sum()*100
    regression[f'shap_{chem}_rank'] = regression[f'shap_{chem}_perc'].rank(method='dense', ascending=False)
    regression[f'shap_{chem}_rank'] = regression[f'shap_{chem}_rank'].replace(regression[f'shap_{chem}_rank'].max(), np.nan)
    return regression

n_regression = (get_regression(f"{os. getcwd()}/Output/{model}/{model}_nitrate as n_mean_log10_shap_importance.csv", "n"))
ton_regression = (get_regression(f"{os. getcwd()}/Output/{model}/{model}_nitrogen total oxidised as n_mean_log10_shap_importance.csv", "ton"))

regression = n_regression.merge(ton_regression, on="Genus", how="outer")
regression['regression_ind'] = ((regression['shap_n_perc'] > regression['shap_n_perc'].sum()/len(regression['shap_n_perc'])) | (regression['shap_ton_perc'] > regression['shap_ton_perc'].sum()/len(regression['shap_ton_perc'])))

results = titan.merge(regression, on="Genus", how="outer")

results.to_csv(f"{os. getcwd()}/Output/results_summary_{model}.csv")

# COMMAND ----------

print(f"Num TITAN N indicators: {sum(results['group_n'] == 1) + sum(results['group_n'] == 2)}")
print(f"Num TITAN TON indicators: {sum(results['group_ton'] == 1) + sum(results['group_ton'] == 2)}")
print(f"Num TITAN N+TON indicators: {sum(results['titan_ind'] == True)}")
print()
print(f"Num XGB N indicators: {results['shap_n_rank'].count()}")
print(f"    Num over equal: {len(results['shap_n_perc'][results['shap_n_perc'] > results['shap_n_perc'].sum()/len(results['shap_n_perc'])])}")

print(f"Num XGB TON indicators: {results['shap_ton_rank'].count()}")
print(f"    Num over equal: {len(results['shap_ton_perc'][results['shap_ton_perc'] > results['shap_ton_perc'].sum()/len(results['shap_ton_perc'])])}")

print(f"Num XGB N+TON indicators: {((results['shap_n_perc'] > 0) | (results['shap_ton_perc'] > 0)).sum()}")
print(f"    Num over equal: {((results['shap_n_perc'] > results['shap_n_perc'].sum()/len(results['shap_n_perc'])) | (results['shap_ton_perc'] > results['shap_ton_perc'].sum()/len(results['shap_ton_perc']))).sum()}")
print()

print(pd.crosstab(results['group_n'], results['group_ton'], margins=False).rename(columns={1:"TON Sensitive", 2:"TON Tolerant", 0: "TON Other"}, index={1:"N Sensitive", 2:"N Tolerant", 0: "N Other"}))
print()
overlap = pd.crosstab(results['titan_ind_dir'], results['xgb_ind'], margins=False).rename(columns={False:"XGB Not-predictive", True:"XGB Predictive"}, index={1:"Titan Sensitive", 2:"Titan Tolerant", 0: "Titan Other"})
print(overlap)
print((overlap/818)*100)

# COMMAND ----------

display(results)
