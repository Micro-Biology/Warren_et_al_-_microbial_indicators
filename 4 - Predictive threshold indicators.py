# Databricks notebook source
# DBTITLE 1,r=0.8
import pandas as pd
import numpy as np
import os

wkdir = os. getcwd()
model = "XGB" #"RF"

n_titan = pd.read_csv(f"{os. getcwd()}/Output/TITAN/titan_n_tax.csv").rename(columns={"Unnamed: 0":"Genus", "zenv.cp":"zenv_n", "filter":"group_n"})[["Genus","zenv_n","group_n"]]
ton_titan = pd.read_csv(f"{os. getcwd()}/Output/TITAN/titan_ton_tax.csv").rename(columns={"Unnamed: 0":"Genus", "zenv.cp":"zenv_ton", "filter":"group_ton"})[["Genus","zenv_ton","group_ton"]]

titan = n_titan.merge(ton_titan, on="Genus", how="outer")
titan['titan_ind'] = titan.apply(lambda row: True if row['group_n'] != 0 or row['group_ton'] != 0 else False, axis=1)

titan['titan_ind_dir'] = 0 
titan.loc[(titan['group_n'] == 1) | (titan['group_ton'] == 1), 'titan_ind_dir'] = 1
titan.loc[(titan['group_n'] == 2) | (titan['group_ton'] == 2), 'titan_ind_dir'] = 2

def get_xgb(loc, chem):
    xgb = pd.read_csv(loc).rename(columns={"absolute_mean_shap_value":f"shap_{chem}","impact_direction":f"direction_{chem}"})
    xgb[f'shap_{chem}_perc'] = xgb[f'shap_{chem}']/xgb[f'shap_{chem}'].sum()*100
    xgb[f'shap_{chem}_rank'] = xgb[f'shap_{chem}_perc'].rank(method='dense', ascending=False)
    xgb[f'shap_{chem}_rank'] = xgb[f'shap_{chem}_rank'].replace(xgb[f'shap_{chem}_rank'].max(), np.nan)
    return xgb

n_xgb = (get_xgb(f"{os. getcwd()}/Output/XGB/XGB_nitrate as n_mean_log10_shap_importance.csv", "n"))
ton_xgb = (get_xgb(f"{os. getcwd()}/Output/XGB/XGB_nitrogen total oxidised as n_mean_log10_shap_importance.csv", "ton"))

xgb = n_xgb.merge(ton_xgb, on="Genus", how="outer")
xgb['xgb_ind'] = ((xgb['shap_n_perc'] > xgb['shap_n_perc'].sum()/len(xgb['shap_n_perc'])) | (xgb['shap_ton_perc'] > xgb['shap_ton_perc'].sum()/len(xgb['shap_ton_perc'])))

results = titan.merge(xgb, on="Genus", how="outer")

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
