# Databricks notebook source
# MAGIC %pip install shap==0.46

# COMMAND ----------

import pandas as pd
import numpy as np
import os
import pickle

from scipy.stats import randint
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV

import shap
import matplotlib.pyplot as plt

# COMMAND ----------

# MAGIC %md
# MAGIC #### Import data and generate optimal hyperparameters
# MAGIC
# MAGIC Important to manually tune hyperparam limits if models are underfitting or skewed

# COMMAND ----------

input_dir = f"{os. getcwd()}/Example_data"

metadata = pd.read_csv(f"{input_dir}/example_metadata.csv")
treated_data = pd.read_csv(f"{input_dir}/example_normalised_abundance.csv")

print(f"treated_data table shape {treated_data.shape}")
print(f"meta table shape {metadata.shape}")
treated_abundance_table = treated_data.drop(columns=["Sample_ID"]).values

#taxa = predictors
taxa = treated_data.columns.tolist()[1:]

#merge all data
bio_chem = pd.merge(treated_data, metadata, left_on = 'Sample_ID', right_on = 'Extended_id', how = 'inner')

new_names = {}
for c in bio_chem.columns.tolist():
    if c not in taxa:
        interim = c.lower()
        interim = interim.replace(',','')
        interim = interim.replace('temperature_of_water','temp')
        interim = interim.replace('mini','min')
        interim = interim.replace('maxi','max')
        new = interim.replace('carbon_organic_dissolved_as_c_:_{doc}_','dissolved_organic_carbon_as_c_')
        new_names[c]=new

bio_chem = bio_chem.rename(columns=new_names)

# COMMAND ----------

output_directory = f"{os. getcwd()}/Output/RF"
os.makedirs(output_directory, exist_ok=True)

def run_random_forest_search(df, n_iter=500, random_state=42, name="_all", ran=[], criterion='squared_error', target_vars=[]):
    """Takes a list with multiple dataframes and runs a randomised search over the hyperparameters of a random forest regressor.
    Each dataframe must have the predictors (stored in global value 'taxa')"""
    #setting an initial grid of hyperparameters for the randomised search to pull from
    param_distributions = {
        'n_estimators': [100, 200, 300, 400, 500], # num trees
        'max_depth': [10, 20, 30, 40, None], # max depth of each tree
        'min_samples_split': randint(2, 20), # min num of samples required to spit an internal node
        'min_samples_leaf': randint(1, 20),
        'max_features': ['sqrt', 'log2', 0.5, None] #None, 1.0 and auto have the same output - number of features to consider when looking for the best split
    }
    
    #only using mean values as target variables
    def get_target_vars(df):
        return [col for col in df.columns if "mean" in col]

    #calculate pred_vars and target_vars for the current dataframe you're iterating over
    global taxa
    pred_vars = taxa.copy()
    if target_vars == []:
        target_vars = get_target_vars(df)
    for r in ran:
        try:
            target_vars.remove(r)
        except ValueError as er:
            print(r)
            raise er
    results = []

    for target in target_vars:
        print(f"  Processing {target}...")
        #subsets data
        all_cols = pred_vars + [target]
        bio_chem_subset = df[all_cols]
        bio_chem_subset_clean = bio_chem_subset.dropna()
        #handles small dataframes due to too many na values
        if bio_chem_subset_clean.shape[0] < 100:
            results.append({
                'wq_variable': target,
                'optimal_parameters': {'max_depth': "NA too little data", 'max_features': "NA too little data", 'min_samples_leaf': "NA too little data", 'min_samples_split': "NA too little data", 'n_estimators': "NA too little data"},
                'highest_cv_score': 0,
                'test_r_squared_score': 0,
                'total_number_samples':bio_chem_subset_clean.shape[0]
            })
            continue
            
        #prepping and splitting data
        bio_chem_y = bio_chem_subset_clean[[target]]
        bio_chem_X = bio_chem_subset_clean[pred_vars]
        bio_chem_X_arr = bio_chem_X.to_numpy()
        bio_chem_y_arr = bio_chem_y.to_numpy().ravel()  
        X_train, X_test, y_train, y_test = train_test_split(bio_chem_X_arr, bio_chem_y_arr, test_size=0.3, random_state=random_state)

        regressor = RandomForestRegressor(random_state=random_state, criterion=criterion)
        randomized_search = RandomizedSearchCV(
                estimator=regressor,
                param_distributions=param_distributions,
                n_iter=n_iter,
                cv=10,
                n_jobs=-1,
                verbose=2,
                random_state=random_state
        )
            
        randomized_search.fit(X_train, y_train)
        #extracting the optimal parameters for each run
        #the best cv score and the best r-squared score for the regressor
        best_params = randomized_search.best_params_
        best_score = randomized_search.best_score_
        best_regressor = randomized_search.best_estimator_
         
        #R2 of test data reported to compare against average model R2 (cv) of the 10 iterations - if drastically different determine why
        y_pred = best_regressor.predict(X_test)
        score = best_regressor.score(X_test, y_test)
            
        result= {
                'wq_variable': target,
                'optimal_parameters': best_params,
                'highest_cv_score': best_score,
                'test_r_squared_score': score
        }
        results.append(result)
        print(result)
 
    #putting results into a dataframe, transposing the dictionary from above into columns
    results_df = pd.DataFrame(results)
    results_df_clean = pd.concat([results_df.drop(columns=['optimal_parameters']), results_df['optimal_parameters'].apply(pd.Series)], axis=1)

    #saving the results to csvs
    output_file = os.path.join(output_directory, f'randomized_search_results_hyperparam_RF.csv')
    results_df_clean.to_csv(output_file, index=False)
    return results_df_clean

# COMMAND ----------


results_df = run_random_forest_search(bio_chem, target_vars=['nitrate as n_mean_log10','nitrogen total oxidised as n_mean_log10'])

# COMMAND ----------

output_directory = f"{os. getcwd()}/Output/RF"
output_file = os.path.join(output_directory, f'randomized_search_results_hyperparam_RF.csv')
results_df = pd.read_csv(output_file)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Make the RF models

# COMMAND ----------

def model_maker(dataframe, hyperparamframe, random_state=42, optimal=True):
    # getting lists of predictor variables and target variables
    global taxa
    pred_vars = taxa.copy()
    
    target_vars = hyperparamframe['wq_variable'].tolist()

    #dictionary will store regressors
    regressors = {}
    scores = {}
    predictions = {}

    for target_var in target_vars:
        #subsets data
        all_cols = ['sample_id'] + pred_vars + [target_var]
        #dataframe_subset = dataframe[all_cols]
        dataframe_subset = dataframe[all_cols].dropna(subset=[target_var] + pred_vars)
        
        #separating target and predictor variables
        sample_ids = dataframe_subset['sample_id'].values  # Store sample IDs separately
        dataframeX = dataframe_subset[pred_vars]
        dataframey = dataframe_subset[[target_var]]
        
        #convert the dataframes to numpy arrays
        dataframeX_arr = dataframeX.to_numpy()
        dataframey_arr = dataframey.to_numpy().ravel()  # Flatten the array for the regressor
        
        #splitting the data into training and test datasets with a 30% test size
        X_train, X_test, y_train, y_test, sample_ids_train, sample_ids_test = train_test_split(dataframeX_arr, dataframey_arr, sample_ids, test_size=0.3, random_state=random_state)

        #find the corresponding hyperparameters for the specified wq_variable
        param_row = hyperparamframe[hyperparamframe['wq_variable'] == target_var]

        if param_row.empty:
            print(f"No parameters found for target variable: {target_var}")
            continue

        param_row = param_row.iloc[0]

        #extracting the hyperpraters - had to specify datatypes for 
        max_depth = param_row['max_depth']
        if pd.isna(max_depth):
            max_depth = None
        else:
            max_depth = int(max_depth)

        if pd.isna(max_features):
            max_features = None
        elif max_features == 'sqrt' or max_features == 'log2':
                pass
        else:
            max_features = float(max_features)

        min_samples_leaf = int(float(param_row['min_samples_leaf']))
        min_samples_split = int(float(param_row['min_samples_split']))
        n_estimators = int(float(param_row['n_estimators']))

        #initialising the regressor with optimal hyperparams extracted
        if optimal == True:
            regressor = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                min_samples_split=min_samples_split,
                max_features=max_features,
                random_state=random_state
            )
        else: 
            regressor = RandomForestRegressor(random_state=random_state)

        #storing the regressor in the dictionary made at top of function
        regressors[f"{target_var}"] = regressor


        #fitting the regressor
        regressor.fit(X_train, y_train)
        
        y_pred = regressor.predict(X_test)
        r_squared = regressor.score(X_test, y_test)
        rmse = np.sqrt(metrics.mean_squared_error(y_test,y_pred))
        scores[target_var] = (r_squared, rmse)

        scores[target_var] = {'r_squared': r_squared, 'rmse': rmse}
        predictions[target_var] = {'y_test': y_test, 'y_pred': y_pred, 'X_test': X_test, 'sample_ids': sample_ids_test}

        #print for initial look?/to track progress
        print(f"Regressor '{target_var}' R^2 score: {r_squared} presplit shape: {dataframe_subset.shape}")

    return regressors, scores, predictions

# COMMAND ----------

regressors_all, regressor_scores_all, regressor_predictions_all = model_maker(bio_chem, results_df)

# COMMAND ----------

objects_to_pickle = {
    'regressors_all': regressors_all,
    'regressor_scores_all': regressor_scores_all,
    'regressor_predictions_all': regressor_predictions_all
}

with open(f'{output_directory}/rf_regressors.pkl', 'wb') as file:
    pickle.dump(objects_to_pickle, file)

# COMMAND ----------

output_directory = f"{os. getcwd()}/Output/RF"

with open(f'{output_directory}/rf_regressors.pkl', 'rb') as file:
    objects_to_pickle = pickle.load(file)

regressors_all = objects_to_pickle['regressors_all']
regressor_scores_all = objects_to_pickle['regressor_scores_all']
regressor_predictions_all = objects_to_pickle['regressor_predictions_all']

# COMMAND ----------

# MAGIC %md
# MAGIC #### SHAP analysis

# COMMAND ----------

def shap_plot_maker(regressors, predictions, scores, taxa, saveas="", check=True, score="RMSE", model="XGB"):

    output_directory = f"{os. getcwd()}/Output/{model}"
    os.makedirs(output_directory, exist_ok=True)

    if saveas == "":
        pass
    elif saveas[0] not in ["_","-"," "]:
        saveas=f"_{saveas}"

    chemicals = regressors.keys()

    #iterate over the list of chemical variables
    for variable in chemicals:

        #get the x_test data stored in the predictions dictionary
        try:
            X_test = pd.DataFrame(predictions[variable]['X_test'].astype(float), columns=taxa)
        except KeyError:
            continue

        rsq = scores[variable]['r_squared']
        score_val = scores[variable][score.lower()]

        # Create an explainer using the model for the specific chemical
        explainer = shap.Explainer(regressors[variable], X_test)

        # Calculate SHAP values
        shap_values = explainer(X_test, check_additivity=check)

        # Generate SHAP violin plot and save as PNG
        plt.figure(figsize=(20, 15))  # Create a new figure for each plot
        clean_variable = variable.replace('_', ' ')
        clean_variable = clean_variable.title()
        clean_variable = clean_variable.replace('As', 'as')
        clean_variable = clean_variable.replace('At', 'at')
        clean_variable = clean_variable.replace(' Ph ', ' pH ')
        clean_variable = clean_variable.replace(' To ', ' to ')
        clean_variable = clean_variable.replace('Caco3', 'CaCO3')

        output_file = os.path.join(output_directory, f'randomized_search_results_hyperparam_{model}.csv')

        #VIOLIN
        title = f"SHAP Violin Plot for {clean_variable}. \n R2: {round(rsq, 3)}   {score} :{round(score_val,3)}"
        plt.title(title, fontsize=16, loc='center', pad=20)
        shap.summary_plot(shap_values, X_test, plot_type="violin", show=False)
        plt.xlabel(f'SHAP Value - Impact of feature on model prediction \nfor {clean_variable}')
        plt.savefig(f"{output_directory}/{model}_{variable}_violin_shap_cluster{saveas}.png", bbox_inches='tight')
        plt.close() 

        #BEESWARM
        title = f"SHAP Beeswarm Plot for {clean_variable}. \n R2: {round(rsq, 3)}   {score} :{round(score_val,3)}"
        plt.title(title, fontsize=16, loc='center', pad=20)
        shap.summary_plot(shap_values, X_test, plot_type="dot", show=False)
        plt.xlabel(f'SHAP Value - Impact of feature on model prediction \nfor {clean_variable}')
        plt.savefig(f"{output_directory}/{model}_{variable}_beeswarm_shap_cluster{saveas}.png", bbox_inches='tight')
        plt.close() 
        
        #extract and save shap feature importance
        #   From: https://github.com/shap/shap/issues/632#issuecomment-854196513     
        absolute_mean_shap_values = np.abs(shap_values.values).mean(axis=0)
        feature_impact_df = pd.DataFrame({
            'Genus': taxa,
            'absolute_mean_shap_value': absolute_mean_shap_values
        })
        feature_impact_df.sort_values(by='absolute_mean_shap_value', ascending=False, inplace=True)
        feature_impact_df.to_csv(f"{output_directory}/{model}_{variable}_shap_importance{saveas}.csv", index=False)


def rename(regressors):
    temp_regressors = {}
    for key, value in regressors.items():
        # Rename the key by removing single quotes around 'all_clusters'
        new_key = key.replace("['", "[")
        new_key = new_key.replace("']", "]")
        temp_regressors[new_key] = value
    return temp_regressors

shap_plot_maker(rename(regressors_all), regressor_predictions_all, regressor_scores_all, taxa, model="RF")
