# Databricks notebook source
# MAGIC %md
# MAGIC ### Tidy up the data in python for R

# COMMAND ----------

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

threshold=0.5 #value at which if no model for that variable passes we wont make a titan2 model

# COMMAND ----------

wkdir = os. getcwd()
rf = pd.read_csv(f"{wkdir}/Output/RF/randomized_search_results_hyperparam_RF.csv")
rf['model'] = 'rf'
xgb = pd.read_csv(f"{wkdir}/Output/XGB/randomized_search_results_hyperparam_XGB.csv")
xgb['model'] = 'xgb'
models = pd.concat([rf[['wq_variable', 'highest_cv_score', 'test_r_squared_score', 'model']], xgb[['wq_variable', 'highest_cv_score', 'test_r_squared_score', 'model']]], axis = 0)

def get_base_variable(variable_name):
    if variable_name.endswith('_log10'):
        return variable_name[:-6]
    return variable_name

def islog10(variable_name):
    if variable_name.endswith('_log10'):
        return 'transformed'
    return 'untransformed'

models['base_wq_variable'] = models['wq_variable'].apply(get_base_variable) #get base variable
models['log10'] = models['wq_variable'].apply(islog10) #get base variable
models_sorted = models.sort_values(by=['base_wq_variable', 'test_r_squared_score'], ascending=[True, False]) #sort by base variable and test r squared score
models_filtered = models_sorted.drop_duplicates(subset='base_wq_variable', keep='first') #keep only best models
titan_chems = models_filtered[models_filtered['test_r_squared_score'] >= threshold]['base_wq_variable'].tolist()

# Define a color palette for the model and transformation combinations
color_palette = {
    ('rf', 'untransformed'): '#c2a5cf',
    ('rf', 'transformed'): '#7b3294',
    ('xgb', 'untransformed'): '#a6dba0',
    ('xgb', 'transformed'): '#008837'
}

# Pivot the dataframe
models_pivot = models_sorted.pivot_table(index='base_wq_variable', columns=['model', 'log10'], values='test_r_squared_score')

models_pivot.plot(kind='bar', figsize=(10,6), color=[color_palette[col] for col in models_pivot.columns])
plt.title('Best Model and Transformation for Each Variable')
plt.xlabel('Base WQ Variable')
plt.ylabel('Test R Squared Score')
plt.legend(title='Model and Transformation')
plt.xticks(rotation=90)
display()

# COMMAND ----------

input_dir = f"{os. getcwd()}/Example_data"

metadata = pd.read_csv(f"{input_dir}/filtered_metadata.csv")
treated_data = pd.read_csv(f"{input_dir}/filtered_normalised_abundance.csv")

output_directory = f"{os. getcwd()}/Output/TITAN"
os.makedirs(output_directory, exist_ok=True)

print(f"treated_data table shape {treated_data.shape}")
print(f"meta table shape {metadata.shape}")

new_names = {}
for c in metadata.columns.tolist():
    if c not in ['Extended_id']: #cols to ignore
        interim = c.lower()
        interim = interim.replace(',','')
        interim = interim.replace('temperature_of_water','temp')
        interim = interim.replace('mini','min')
        interim = interim.replace('maxi','max')
        new = interim.replace('carbon_organic_dissolved_as_c_:_{doc}_','dissolved_organic_carbon_as_c_')
        new_names[c]=new
metadata = metadata.rename(columns=new_names)

def filter_col_asvs(asvs, metadata, column, values, keep=False):
    """Filters by 'column' from the asv table using the metadata. Returns tuple of asvs and metadata to be unpacked."""
    df = asvs.copy().rename(columns={'Sample_ID':'Extended_id'})
    df_m = pd.merge(df, metadata, on='Extended_id')
    df_m = filter_col(df_m, column, values, keep=keep)
    df_m = df_m[df.columns].rename(columns={'Extended_id':'Sample_ID'})
    return df_m, filter_col(metadata, column, values, keep=keep)

def filter_col(data, column, values, keep=False):
    """Filter by values in list 'values' from dataframe using metadata 'column'.
    If values is a list it will filter by those values, if values is numeric it will filter less than/greater than."""
    df = data.copy()
    if values=="null":
        df = df.dropna(subset=[column])
    elif type(values) == list:
        col_values = df[column].unique()
        for value in values:
            if value not in col_values:
                print(f"Values passed though not in metadata. Available {column}:")
                print(col_values)
                raise KeyError(f"Check {column} has the above values.")
        if keep == True:
            df = df[df[column].isin(values)]
        elif keep == False:
            df = df[~df[column].isin(values)]
        else:
            return None
    elif type(values) in [int, float]:
        if keep == True:
            df = df[df[column] > values]
        elif keep == False:
            df = df[df[column] < values]
        else:
            return None
    else:
        return None
    return df

print("Chemicals to run titan2 on:")
for chem in titan_chems:
    print(f"    {chem}")
    output_directory = f"{os. getcwd()}/Output/TITAN/{chem}"
    os.makedirs(output_directory, exist_ok=True)
    treated_data_n, meta_n = filter_col_asvs(treated_data, metadata, chem, "null", keep=True)
    treated_data_n.to_csv(f"{output_directory}/tidy_abun.csv", index=False)
    meta_n.to_csv(f"{output_directory}/tidy_meta.csv", index=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Do analysis in R

# COMMAND ----------

# MAGIC %r
# MAGIC if (!require("BiocManager", quietly = TRUE))
# MAGIC     install.packages("BiocManager")
# MAGIC
# MAGIC BiocManager::install("dplyr")
# MAGIC install.packages("TITAN2")
# MAGIC #install.packages("parallel")

# COMMAND ----------

# DBTITLE 1,Run TITAN
# MAGIC %r
# MAGIC library("TITAN2")
# MAGIC library("parallel")
# MAGIC
# MAGIC # Set path
# MAGIC path <- file.path(getwd(), "Output","TITAN")
# MAGIC
# MAGIC set.seed(42)
# MAGIC
# MAGIC ASV_tab <- read.table(file.path(path, "tidy_abun.csv"), header = TRUE, row.names = 1, check.names = FALSE, sep = ",")
# MAGIC meta <- read.table(file.path(path, "tidy_meta.csv"), header = TRUE, check.names = FALSE, sep = ",")
# MAGIC rownames(meta) <- meta$Extended_id
# MAGIC
# MAGIC # Subset metadata to numeric variables of interest
# MAGIC variables <- "Nitrate as N_mean"
# MAGIC meta_sub <- meta[variables]
# MAGIC
# MAGIC start <- date()
# MAGIC titan_genera_nitrate <- titan(meta_sub, ASV_tab, minSplt = 5, numPerm = 500, boot = T, nBoot = 500, imax = F, ivTot = F, pur.cut = 0.8, rel.cut = 0.95, ncpus = 8)
# MAGIC end <- date()
# MAGIC
# MAGIC #renamed .rds files manually
# MAGIC saveRDS(titan_genera_nitrate, file = file.path(path, "titan.rds"))

# COMMAND ----------

# MAGIC %r
# MAGIC library("TITAN2")
# MAGIC
# MAGIC library("TITAN2")
# MAGIC 
# MAGIC chemicals <-list("nitrogen total oxidised as n_mean", "nitrate as n_mean")
# MAGIC 
# MAGIC for (chem in chemicals) {
# MAGIC   print(chem)
# MAGIC   # Set path
# MAGIC   path <- file.path(getwd(), "Output","TITAN",chem)
# MAGIC 
# MAGIC   set.seed(42)
# MAGIC 
# MAGIC   ASV_tab <- read.table(file.path(path, "tidy_abun.csv"), header = TRUE, row.names = 1, check.names = FALSE, sep = ",")
# MAGIC   meta <- read.table(file.path(path, "tidy_meta.csv"), header = TRUE, check.names = FALSE, sep = ",")
# MAGIC   rownames(meta) <- meta$Extended_id
# MAGIC 
# MAGIC   # Subset metadata to numeric variables of interest
# MAGIC   meta_sub <- meta[chem]
# MAGIC 
# MAGIC   start <- date()
# MAGIC   #this uses a lot of RAM - we ran this on a server with 256GB
# MAGIC   titan_genera <- titan(meta_sub, ASV_tab, minSplt = 5, numPerm = 500, boot = T, nBoot = 500, imax = F, ivTot = F, pur.cut = 0.8, rel.cut = 0.95, ncpus = 8)
# MAGIC   end <- date()
# MAGIC 
# MAGIC   saveRDS(titan_genera, file = file.path(path, "titan.rds"))
# MAGIC   write.csv(titan_genera_n$sppmax, file.path(path, "titan_tax.csv"))
# MAGIC   write.csv(titan_genera_n$sumz.cp, file.path(path, "titan_thres.csv"))
# MAGIC }
