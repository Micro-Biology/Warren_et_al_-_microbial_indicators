# Databricks notebook source
# MAGIC %pip install shap==0.46 xgboost==3.0.0

# COMMAND ----------

import pandas as pd
import numpy as np
import os
import pickle

from sklearn.metrics import r2_score
from scipy.stats import linregress
from scipy.stats import randint, uniform
from sklearn import metrics
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV

import shap
import matplotlib.pyplot as plt

# COMMAND ----------

# DBTITLE 1,FUNCTIONS
def run_xgboost_search(df, n_iter=500, random_state=42, name="_all", ran=[], eval_metric='rmse', target_vars=[]):

    param_distributions = {
        'n_estimators': [20, 50, 100, 200, 300, 400],  # number of trees, step sizes from Amanda's code
        'max_depth': randint(3, 10),  # max depth of the trees
        'learning_rate': [0.1, 0.2, 0.3, 0.4],  # learning rate (eta)
        'subsample': [0.6, 0.7, 0.8, 0.9, 1],  # percentage of rows used per tree, uniform distribution between 0.6 and 1
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],  # percentage of features used per tree, uniform distribution between 0.6 and 1
        'gamma': uniform(0, 0.2),  # minimum loss reduction, uniform distribution between 0 and 0.2
        'min_child_weight': randint(1, 10),  # minimum sum of instance weight (hessian) needed in a child
        'reg_alpha': [0, 0.1, 0.5],  # L1 regularization, uniform distribution between 0 and 1
        'reg_lambda': uniform(0, 1)}  # L2 regularization, uniform distribution between 0 and 1

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
        
    for target in target_vars:
        assert target in df.columns.tolist()
    
    results = []
    for target in target_vars:
        print(f"  Processing {target}...")
        #subsets data
        all_cols = pred_vars + [target]
        bio_chem_subset = df[all_cols]
        bio_chem_subset_clean = bio_chem_subset.dropna(axis='index',subset=[target]).fillna(0)
        if bio_chem_subset_clean.shape[0] < 100:
            results.append({
                'wq_variable': target,
                'optimal_parameters': {'colsample_bytree': "NA too little data", 
                                       'gamma': "NA too little data", 
                                       'learning_rate': "NA too little data", 
                                       'max_depth': "NA too little data",
                                       'min_child_weight': "NA too little data",
                                       'n_estimators': "NA too little data",
                                       'reg_alpha': "NA too little data",
                                       'reg_lambda': "NA too little data",
                                       'subsample': "NA too little data"},
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

        regressor = XGBRegressor(random_state=random_state, eval_metric=eval_metric)
        randomized_search = RandomizedSearchCV(
            estimator=regressor,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=10,
            n_jobs=-1,
            verbose=0,
            random_state=random_state
        )

        randomized_search.fit(X_train, y_train)
        # Extract the best parameters and score
        best_params = randomized_search.best_params_
        best_score = randomized_search.best_score_
        best_regressor = randomized_search.best_estimator_

        #R2 of test data reported to compare against average model R2 (cv) of the 10 iterations - if drastically different determine why
        y_pred = best_regressor.predict(X_test)
        score = best_regressor.score(X_test, y_test)  # R-squared

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
    output_file = os.path.join(output_directory, f'randomized_search_results_hyperparam_XGB.csv')
    results_df_clean.to_csv(output_file, index=False)
    return results_df_clean

def xgb_model_maker(dataframe, hyperparamframe, random_state=42, optimal=True):
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
        #dataframe_subset = dataframe[all_cols].dropna(subset=[target_var] + pred_vars) #old version
        dataframe_subset = dataframe[all_cols].dropna(subset=[target_var]) #new version
        
        #separating target and predictor variables
        sample_ids = dataframe_subset['sample_id'].values  # Store sample IDs separately
        dataframeX = dataframe_subset[pred_vars]
        dataframey = dataframe_subset[[target_var]]
        
        #convert the dataframes to numpy arrays
        dataframeX_arr = dataframeX.to_numpy()
        dataframey_arr = dataframey.to_numpy().ravel()  # flatten the array for the regressor
        
        #splitting the data into training and test datasets with a 30% test size
        X_train, X_test, y_train, y_test, sample_ids_train, sample_ids_test = train_test_split(dataframeX_arr, dataframey_arr, sample_ids, test_size=0.3, random_state=random_state)


        #find the corresponding hyperparameters for the specified wq_variable
        param_row = hyperparamframe[hyperparamframe['wq_variable'] == target_var]

        #can take this out because you'd rather the code breaks? Will notice more quickly?
        if param_row.empty:
            print(f"No parameters found for target variable: {target_var}")
            continue

        param_row = param_row.iloc[0]  #because we've subset there's only one row in each iteration

        #extracting the hyperpraters - had to specify datatypes for 
        max_depth = param_row['max_depth']
        if pd.isna(max_depth):
            max_depth = None  #set to None if NaN as this got lost in hyperparam frame
        else:
            max_depth = int(max_depth)

        colsample_bytree = param_row['colsample_bytree']
        gamma = param_row['gamma']
        learning_rate = param_row['learning_rate']
        max_depth = int(param_row['max_depth'])
        min_child_weight = int(param_row['min_child_weight'])
        n_estimators = int(param_row['n_estimators'])
        reg_alpha = param_row['reg_alpha']
        reg_lambda = param_row['reg_lambda']
        subsample = param_row['subsample']

        #initialising the regressor with optimal hyperparams extracted
        if optimal == True:
            regressor = XGBRegressor(random_state=random_state,
                colsample_bytree=colsample_bytree,
                gamma=gamma,
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_child_weight=min_child_weight,
                n_estimators=n_estimators,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                subsample=subsample
            )
        else:
            regressor = XGBRegressor(random_state=random_state)

        #storing the regressor in the dictionary made at top of function
        regressors[target_var] = regressor

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

        plot_pred_v_obsv(y_test, y_pred, target_var, model="XGB", r_squared=r_squared)

    return regressors, scores, predictions

def plot_pred_v_obsv(y_test, y_test_pred, chem, model="XGB", r_squared=None):
    output_directory = f"{os.getcwd()}/Output/{model}"
    os.makedirs(output_directory, exist_ok=True)
    if r_squared is None:
        from sklearn.metrics import r2_score
        r_squared = r2_score(y_test, y_test_pred)

    slope, intercept, r_value, p_value, std_err = linregress(y_test, y_test_pred)
    x_min, x_max = -2.6, 1.6
    x_line = np.linspace(x_min, x_max, 100)
    y_line = slope * x_line + intercept

    colours = {
        "nitrate as n_mean_log10": {"RF":"#01665e","XGB":"#35978f"},
        "nitrogen total oxidised as n_mean_log10": {"RF":"#8c510a","XGB":"#bf812d"},
    }

    plt.figure(figsize=(6, 6))
    test_label = f"Test Data ({model} R² = {r_squared:.3f})"
    plt.scatter(y_test, y_test_pred, color=colours[chem][model], alpha=0.6, label=test_label)
    plt.axline(xy1=(0, 0), xy2=(1, 1), color="black", linestyle="--", linewidth=2, label=r"Ideal Line $y=x$")
    reg_label = f"Linear Fit ($m={slope:.2f}$, $b={intercept:.2f}$)"
    plt.plot(x_line, y_line, color='#a54b53', linestyle='-', linewidth=2, label=reg_label) 
    plt.xlabel("Observed (Actual $y$)", fontsize=14)
    plt.ylabel("Predicted ($\hat{y}$)", fontsize=14)
    plt.title(f"{chem}: Observed vs. Predicted", fontsize=16)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xlim(x_min, x_max)
    plt.ylim(x_min, x_max)
    #plt.axis("equal")
    
    plt.savefig(f"{output_directory}/{chem}_{model}_pred_v_obsv_with_reg.png")
    plt.show()


def shap_plot_maker(regressors, predictions, scores, taxa, output_directory, saveas="", check=True, score="RMSE", model="XGB"):
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

# COMMAND ----------

# MAGIC %md
# MAGIC #### Import data and generate optimal hyperparameters
# MAGIC
# MAGIC Important to manually tune hyperparam limits if models are underfitting or skewed

# COMMAND ----------

input_dir = f"{os. getcwd()}/Example_data"

#Manually rereads the optimal hyperparameters reported in the paper - they will always be slightly different as we used RandomisedCVSearch() which has some inherent randomness to it
reoptimise = False

metadata = pd.read_csv(f"{input_dir}/filtered_metadata_normal.csv")
treated_data = pd.read_csv(f"{input_dir}/filtered_asv_normal.csv")

print(f"treated_data table shape {treated_data.shape}")
print(f"meta table shape {metadata.shape}")

#taxa = predictors
taxa = treated_data.columns.tolist()[1:]

#merge all data
bio_chem = pd.merge(treated_data, metadata, left_on = 'Sample_ID', right_on = 'ANON_ID', how = 'inner').drop(columns=['ANON_ID'])

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

bio_chem['nitrate as n_mean_log10'] = np.log10(bio_chem['nitrate as n_mean'])
bio_chem['nitrogen total oxidised as n_mean_log10'] = np.log10(bio_chem['nitrogen total oxidised as n_mean'])

output_directory = f"{os. getcwd()}/Output/XGB"
os.makedirs(output_directory, exist_ok=True)

#If True reruns the hyperparameter search, otherwise uses the hyperparameters reported previously - only do this if you change the filtering of the input not the names of the taxa
if reoptimise == True:
    results_df = run_xgboost_search(bio_chem, target_vars=['nitrate as n_mean_log10','nitrogen total oxidised as n_mean_log10'])

output_file = os.path.join(output_directory, f'randomized_search_results_hyperparam_XGB.csv')
results_df = pd.read_csv(output_file)

display(results_df)

regressors_all, regressor_scores_all, regressor_predictions_all = xgb_model_maker(bio_chem, results_df)

objects_to_pickle = {
    'regressors_all': regressors_all,
    'regressor_scores_all': regressor_scores_all,
    'regressor_predictions_all': regressor_predictions_all
}

with open(f'{output_directory}/xgb_regressors.pkl', 'wb') as file:
    pickle.dump(objects_to_pickle, file)

shap_plot_maker(rename(regressors_all), regressor_predictions_all, regressor_scores_all, taxa, output_directory, model="XGB")
