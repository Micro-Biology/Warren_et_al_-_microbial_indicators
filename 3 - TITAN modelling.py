# Databricks notebook source
# MAGIC %md
# MAGIC ### Tidy up the data in python for R

# COMMAND ----------

import pandas as pd
import numpy as np
import os

# COMMAND ----------

# DBTITLE 1,Tidy data
input_dir = f"{os. getcwd()}/Example_data"

metadata = pd.read_csv(f"{input_dir}/example_metadata.csv")
treated_data = pd.read_csv(f"{input_dir}/example_normalised_abundance.csv")

output_directory = f"{os. getcwd()}/Output/TITAN"
os.makedirs(output_directory, exist_ok=True)

print(f"treated_data table shape {treated_data.shape}")
print(f"meta table shape {metadata.shape}")
treated_abundance_table = treated_data.drop(columns=["Sample_ID"]).values

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

treated_data_n, meta_n = filter_col_asvs(treated_data, metadata, "Nitrate as N_mean", "null", keep=True)

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
# MAGIC # Set path
# MAGIC path <- file.path(getwd(), "Output","TITAN")
# MAGIC
# MAGIC #change manually
# MAGIC titan_genera_n <- readRDS(file.path(path, "titan_genera_nitrate.rds")) 
# MAGIC write.csv(titan_genera_n$sppmax, file.path(path, "titan_n_tax.csv"))
# MAGIC write.csv(titan_genera_n$sumz.cp, file.path(path, "titan_n_thres.csv"))
