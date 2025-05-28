# Warren et al - Microbial indicators

Tested on the databricks platform on runtime 16.3 all workbooks were ran on a 16 core 56GB compute with the exception of TITAN which required more memory. All scripts are designed to be ran within this folder and will save outputs within the folder by default, inputs are currently set to the example data which are the data used for the manuscript.

### 1 - RF modelling
Brings in metadata and abundanc table, find optimal _RandomForestRegressor()_ hyperparameters within search range, creates and pickles final model, generates SHAP scores and saves beeswarm, violin and SHAP feature importance. 

### 2 - XGB modelling
Brings in metadata and abundanc table, find optimal _XGBRegressor()_ hyperparameters within search range, creates and pickles final model, generates SHAP scores and saves beeswarm, violin and SHAP feature importance. 

### 3 - TITAN modelling
Brings in metadata and abundanc table and filters within python and then runs the analysis using the _TITAN2 R_ package.

### 4 - Predictive threshold indicators
Brings in results of one of the predictive models and _TITAN_ analysis together - this will require editing to get file names and pressures correctly labelled.

### Further considerations:

- Hyperparemeter optmisation: We found the search range for hyperparameters to be best for our data but we ran _RandomizedSearchCV()_ to optimise for time. Further optimisations could be made to these hyperparameters and _GridSearchCV_ ran instead.
- _TITAN2 pur.cut_: We found our sum+/- to be different from our fsum+/- due to outliers and so reran analysis with a reduced _pur.cut_ (0.8) to include less pure but still statistically significant genera.
- Sparsity and ASV grouping: As highlighted in the discussion, the appropriate level to group ASVs will depend on data sparsity. We reccomend against OTUs as whilst they will reduce sparsity, they are not cmparable across bioinformatic runs making their use limited if the goal is metric development.