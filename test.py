import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, log_loss, roc_auc_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, f1_score
)
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm  # progress bar
import matplotlib.pyplot as plt
from plotnine import *  # For visualizations (ggplot2-style)
import dtreeviz
from patsy import dmatrix, ModelDesc, Term, EvalFactor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Lasso, LassoCV
import statsmodels.formula.api as smf
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
import shap



# Load Data, Create columns, and reorder

df = pd.read_csv('merged.csv')

df = df.rename(columns = { ' Timestamp' : 'Timestamp' })

df['Timestamp'] = pd.to_datetime(df['Timestamp'], dayfirst = True, format='mixed')

dfa = {}

for col in df.columns:
    if col not in ["Normal/Attack", "Timestamp"]:    
        dfa[f"{col}_diff_30s"] = df[col].diff(30)
        dfa[f"{col}_diff_5m"] = df[col].diff(300)
        dfa[f"{col}_diff_30m"] = df[col].diff(1800)
        dfa[f"{col}_diff_90m"] = df[col].diff(5400)
        dfa[f"{col}_std_30s"] = df[col].rolling(window=30).std()
        dfa[f"{col}_std_5m"] = df[col].rolling(window=300).std()

dfaa = pd.DataFrame(dfa)

df = pd.concat([df, dfaa], axis=1)

attack_cols = [c for c in df.columns if c.startswith("Normal/Attack")]

other_cols = [c for c in df.columns if c not in attack_cols]

df = df[other_cols + attack_cols]

df["Normal_Attack"] = df["Normal/Attack"]
df = df.drop(columns=["Normal/Attack"])

df = df.sort_values('Timestamp')
#The data is all the normal times, then all the atatck times, in their respective timestamp order. We reorder it to the timestamp overall.
# Drop first 5400 rows which have NaN as it breaks the scaler :(
df = df.dropna().reset_index(drop=True)

# Split Data

df['Normal_Attack'] = df['Normal_Attack'].map({'Normal': 0, 'Attack': 1}).astype('float32')

normal = df[df['Normal_Attack'] == 0]
attack = df[df['Normal_Attack'] == 1]

length_normal = len(normal)*.8
length_attack = len(attack)*.8

train_normal = normal.iloc[:int(length_normal)]
test_normal = normal.iloc[int(length_normal):]

train_attack = attack.iloc[:int(length_attack)]
test_attack = attack.iloc[int(length_attack):]

df_train = pd.concat([train_normal, train_attack]).sort_index()
df_test = pd.concat([test_normal, test_attack]).sort_index()

# Extract columns and create Test and Train
feature_cols = df.columns[1:-1]

X_train_raw = df_train[feature_cols].copy()
X_test_raw  = df_test[feature_cols].copy()

# Ensure numeric dtype (coerce non-numeric to NaN; XGBoost handles NaN by default)
X_train = X_train_raw.apply(pd.to_numeric, errors='coerce')
X_test  = X_test_raw.apply(pd.to_numeric, errors='coerce')

# Extract response variable
y_train = df_train['Normal_Attack'].astype(int)
y_test  = df_test['Normal_Attack'].astype(int)















# # Logistic Regression with Lasso



# # df['Normal_Attack'] = df['Normal_Attack'].map({'Normal': 0, 'Attack': 1}).astype('float32')
# y = df_train["Normal_Attack"].to_numpy()
# X = df_train.drop(columns=["Normal_Attack", 'Timestamp'])

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# features = X.columns

# x_data = pd.DataFrame(X_scaled, columns=features)


# # # Manually Select Lasso Alpha
# # lasso = Lasso(alpha=0.15, fit_intercept=True, max_iter=10000)
# # lasso.fit(X_scaled, y)

# # Or allow Cross-Validation to select Lasso Alpha
# lambda_seq = np.arange(0.1, 10.0 + 1e-12, 0.1)
# cv_model = LassoCV(alphas=lambda_seq, cv=10, fit_intercept=True, max_iter=10000)
# cv_model.fit(X_scaled, y)
# alpha_min = cv_model.alpha_
# lasso = Lasso(alpha=alpha_min, fit_intercept=True, max_iter=10000)
# lasso.fit(X_scaled, y)


# # OLS Regression with Lasso
# coef_series = pd.Series(lasso.coef_, index=features, name="coef")
# coef_df = coef_series.reset_index()

# selected_features = coef_series[coef_series != 0].index.tolist()

# predictors = [f'Q("{c}")' for c in selected_features]
# formula = 'Q("Normal_Attack") ~ ' + " + ".join(predictors)

# fit_3 = smf.logit(formula = formula, data = df).fit()

# print(fit_3.summary())
# # logistic regression predictions and accuracy
# logit_pred_prob = fit_3.predict(df_test[selected_features])
# logit_pred = (logit_pred_prob >= 0.5).astype(int)
# logit_acc = accuracy_score(y_test, logit_pred)
# print(f"\nLogistic Regression Accuracy: {logit_acc:.4f}")






# # Bagging and random forest 

# #This may be removable 
# # Drop NaN rows :(
# train_mask = ~X_train.isna().any(axis=1)
# X_train = X_train[train_mask]
# y_train = y_train[train_mask]

# test_mask = ~X_test.isna().any(axis=1)
# X_test = X_test[test_mask]
# y_test = y_test[test_mask]

# tree_model = DecisionTreeClassifier(max_depth=4, random_state=123)  # Initialize tree
# tree_model.fit(X_train, y_train) # Fit tree

# # Encode labels for visualziation
# le = preprocessing.LabelEncoder()
# y_train_enc = le.fit_transform(y_train)

# # Set up visualization
# viz_model = dtreeviz.model(
#     tree_model,
#     X_train=X_train,
#     y_train=y_train_enc,
#     feature_names=list(X_train.columns),
#     target_name="outcome",
#     class_names=[str(c) for c in le.classes_]
# )
# v = viz_model.view(fontname="DejaVu Sans")
# v.save("decision_tree_viz.svg") # Save visualization

# y_pred = tree_model.predict(X_test)
# acc = accuracy_score(y_test, y_pred)
# # y_pred_enc = tree_model.predict(X_test) # Create predictions
# # y_pred = le.inverse_transform(y_pred_enc) # Convert predicitons back to Win/Loss
# acc = accuracy_score(y_test, y_pred) # Calcualte accuracy
# print(f"\nDecision Tree Accuracy on Test Set: {acc:.4f}") # Print accuracy









# # XGBoost

# Set up XGBDmatrix
dtrain = xgb.DMatrix(data=X_train.values, label=y_train)
dtest  = xgb.DMatrix(data=X_test.values,  label=y_test)

# params = {
#         "objective": "binary:logistic", # Set objective
#         "eval_metric": ["auc", "error"],  # Track both AUC and error
#         "seed": 42, # set seed

#     }
# num_boost_round = 5 # Set number of rounds

# watchlist = [(dtrain, "train")] # Set data for evaluation
# booster = xgb.train(params, # Set parameters
#                     dtrain,  # Set training data
#                     num_boost_round=num_boost_round, # Set number of rounds
#                     evals=watchlist,  # Set data to evaluate on
#                     verbose_eval=50) # Set print out frequency


# test_pred_raw = booster.predict(dtest)

# test_pred_cls = (test_pred_raw >= 0.5).astype(int)


# print("\nConfusion matrix:")
# cm = (confusion_matrix(y_test, test_pred_cls))
# disp = ConfusionMatrixDisplay(confusion_matrix=cm)# Set class labels
# disp.plot(cmap="Blues") # Set color map
# plt.title("Confusion Matrix — XGBoost") # Set title
# plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
# plt.close()
# print("\nAccuracy):")
# print(accuracy_score(y_test, test_pred_cls)) # Get classification report


# # Now we will weight it!

# # Count values
counts = pd.Series(y_train).value_counts().sort_index()
neg = int(counts.get(0, 0)); pos = int(counts.get(1, 0)) # Calculate positive and negative samples
print(f"Number of negative samples: {neg}")
print(f"Number of positive samples: {pos}")

# Calculate ratio
ratio = neg / pos
# Set ratio as weight for positive samples
w_tr = np.where(y_train == 1, ratio, 1.0).astype(np.float32)

# Build weighted DMatrices
dtrain_w = xgb.DMatrix(X_train.values, label=y_train, weight=w_tr)

# watchlist = [(dtrain_w, "train")] # Set data for evaluation
# xgb_w = xgb.train(params, # Set parameters
#                     dtrain_w,  # Set training data
#                     num_boost_round=3, # Set number of rounds
#                     evals=watchlist,  # Set data to evaluate on
#                     verbose_eval=50) # Set print out frequency

# test_pred_w = xgb_w.predict(dtest) # Create predictions


# # Convert predictions into classes at 0.5
# test_pred_cls_w = (test_pred_w >= 0.5).astype(int)


# print("\nConfusion matrix:")
# cm = (confusion_matrix(y_test, test_pred_cls_w))
# disp = ConfusionMatrixDisplay(confusion_matrix=cm)# Set class labels
# disp.plot(cmap="Blues") # Set color map
# plt.title("Confusion Matrix — Weighted XGBoost") # Set title
# plt.savefig("confusion_matrix_weighted.png", dpi=300, bbox_inches="tight")
# plt.close()
# print("\nAccuracy):")
# print(accuracy_score(y_test, test_pred_cls_w)) # Get Accuracy











# #XGBoost Tuning to get the best parameters

# params = {
#     "objective": "binary:logistic",   # Set objective
#     "eta": 0.1,                       # Set learning rate
#     "eval_metric": ["auc", "error"],  # Track both AUC and error
#     "tree_method": "hist",
#     "seed": 111111,
#     "nthread": 1,                     # Parallel threads
# }

# # Run CV inside XGBoost
# cv_res = xgb.cv(
#     params=params,
#     dtrain=dtrain_w,              # Training data (DMatrix)
#     num_boost_round=400,       # Number of rounds
#     nfold=5,                    # 5-fold CV
#     verbose_eval=20,            # Print every 20 iters
#     stratified=True,            # Good practice for classification
#     shuffle=True,
# )

# cv_res

# # Identify best iteration
# #Remove idmix if you wan tot
# best_idx = cv_res['test-error-mean'].idxmin()
# best_iter = int(best_idx) + 1 # Increment by 1 to get iteration
# best_err  = float(cv_res.loc[best_idx, 'test-error-mean']) # Extract test error
# best_auc  = float(cv_res.loc[best_idx, 'test-auc-mean']) if 'test-auc-mean' in cv_res.columns else np.nan # Extract test AUC

# # Print results
# print(f"Best iteration (by min test error): {best_iter}")
# print(f"Min test error at best iter: {best_err:.6f}")
# if not np.isnan(best_auc):
#     print(f"Test AUC at best iter: {best_auc:.6f}")

# # Set range of parameter values to try
# grid = {
#     "max_depth": [3,  7, 10],
#     "min_child_weight": [ 5, 7, 10],
# }
# param_grid = list(ParameterGrid(grid))

# # Set base model parameters
# base_params = {
#     "objective": "binary:logistic",
#     "eta": 0.10,
#     "eval_metric": ["error", "auc"],
#     "tree_method": "hist",
#     "seed": 111111,
#     "nthread": 1,  # keep each worker single-threaded to avoid oversubscription
# }


# def run_one_cv(md, mcw):
#     """Run xgb.cv for a single (max_depth, min_child_weight) pair and return best metrics."""
#     params = base_params.copy()
#     params.update({"max_depth": int(md), "min_child_weight": int(mcw)})

#     cv = xgb.cv(
#         params=params,
#         dtrain=dtrain_w,              # DMatrix from earlier
#         num_boost_round=1000,        # nrounds = 100
#         nfold=5,                    # 5-fold CV
#         early_stopping_rounds=20,   # stop if no improvement
#         stratified=True,
#         shuffle=True,
#         verbose_eval=False,
#         seed=111111,
#     )

#     # Best round is the length of the early-stopped trace
#     best_round = len(cv)

#     # Read AUC & error at the best round row explicitly
#     best_row = cv.iloc[best_round - 1]
#     best_err = float(best_row["test-error-mean"])
#     best_auc = float(best_row["test-auc-mean"])

#     # Return results
#     return {
#         "max_depth": md,
#         "min_child_weight": mcw,
#         "best_round": best_round,
#         "test_error": best_err,
#         "test_auc": best_auc,
#     }

# results = []
# for p in tqdm(param_grid, desc="Grid CV (serial)"):
#     results.append(run_one_cv(p["max_depth"], p["min_child_weight"]))



# # Create and sort results data frame
# cv_results_df = (
#     pd.DataFrame(results)
#       .sort_values(["test_error", "test_auc"], ascending=[True, False])
#       .reset_index(drop=True)
# )

# # Identify best parameters0
# best_pair = cv_results_df.iloc[0].to_dict()
# print("Best (by min test_error, then max AUC):", best_pair)    


# # Create results data frame
# res_db = (
#     cv_results_df[["max_depth", "min_child_weight", "test_auc", "test_error"]]
#     .rename(columns={"test_auc": "auc", "test_error": "error"})
#     .copy()
# )

# tuned_max_depth = int(best_pair['max_depth']) # Extract max depth
# tuned_min_child = int(best_pair['min_child_weight']) # Extract min_child_weight

# # Define gamma grid
# gamma_vals = [0.00, 0.05, 0.10, 0.15, 0.20]

# # Set base parameters
# base_params = {
#     "objective": "binary:logistic",
#     "eta": 0.10,
#     "max_depth": tuned_max_depth, # Uuse tuned value of max depth
#     "min_child_weight": tuned_min_child, # Use tuned value of min child weight
#     "tree_method": "hist",
#     "eval_metric": ["auc", "error"],
#     "seed": 111111,
#     "nthread": 1,
# }

# ### Be careful this can take a long time to run ###
# rows = [] # Create data frame to store valus
# for g in tqdm(gamma_vals, desc="Gamma CV (serial)"): # For each gamma value
#     params = base_params.copy() # Create copy of base parameters
#     params["gamma"] = float(g) # Replace value with current gamma value

#     # Run xgb.cv
#     cv = xgb.cv(
#         params=params,
#         dtrain=dtrain_w,                # Set training data
#         num_boost_round=1000,          # Set number of rounds
#         nfold=5,                      # Set folds for cross validation
#         early_stopping_rounds=20,     # Set early stopping rounds
#         stratified=True,
#         shuffle=True,
#         verbose_eval=False,
#         seed=111111, #Set seed
#     )

#     # Best iteration is the length of the early-stopped trace
#     best_round = len(cv)
#     best_row = cv.iloc[best_round - 1]
#     # Store results from current iteration
#     rows.append({
#         "gamma": g,
#         "best_round": int(best_round),
#         "test_auc": float(best_row["test-auc-mean"]),
#         "test_error": float(best_row["test-error-mean"]),
#     })

# # Join results into data frame
# gamma_results = (pd.DataFrame(rows)
#                    .sort_values(['test_error', 'test_auc'], ascending=[True, False])
#                    .reset_index(drop=True))
# # # View results
# # display(gamma_results)
# # Extract best value
# best_gamma = float(gamma_results.iloc[0]['gamma'])
# # Print out vest value
# print(f"Selected gamma (by min test_error, then max AUC): {best_gamma:.2f}")

# tuned_gamma = float(best_gamma) # Extract best gamma value


# # Create grid of possible values
# grid = {
#     "subsample":        [0.6, 0.7, 0.8, 0.9, 1.0],
#     "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
# }
# # Convert into parameter grid and then list
# param_grid = list(ParameterGrid(grid))
# # Set base parameters
# base_params = {
#     "objective": "binary:logistic",
#     "eta": 0.10,
#     "max_depth": tuned_max_depth, # Use tuned value for max depth
#     "min_child_weight": tuned_min_child, # Use tuned value for min child weight
#     "gamma": tuned_gamma, # Use tuned value for gamma
#     "tree_method": "hist",
#     "eval_metric": ["auc", "error"],
#     "seed": 111111,
#     "nthread": 1,                     # single core
# }

# # Create function
# def run_one_cv(subsample, colsample_bytree):
#     """Run xgb.cv for a single (subsample, colsample_bytree) and return best metrics."""
#     params = base_params.copy() # Create copy of base parameters
#     params.update({ # Update with values of subsample and colsample_bytree
#         "subsample": float(subsample),
#         "colsample_bytree": float(colsample_bytree),
#     })
#     # Run xgb.cv
#     cv = xgb.cv(
#         params=params,  # Set parameters
#         dtrain=dtrain_w,  # Set training data
#         num_boost_round=1000, # Set number of rounds
#         nfold=5,  # Set cross-validation folds
#         early_stopping_rounds=20, # Set number of early stopping rounds
#         stratified=True,
#         shuffle=True,
#         verbose_eval=False,
#         seed=111111, # Set seed
#     )

#     best_round = len(cv)             # early-stopped length
#     best_row = cv.iloc[best_round - 1] # Identify best row
#     # Return results
#     return {
#         "subsample": subsample,
#         "colsample_bytree": colsample_bytree,
#         "best_round": int(best_round),
#         "test_auc": float(best_row["test-auc-mean"]),
#         "test_error": float(best_row["test-error-mean"]),
#     }

# ### Be careful this can take a long time to run ###
# rows = [] # Create empty list to store results
# # For each set of parameters
# for p in tqdm(param_grid, desc="Subsample × Colsample_bytree CV (serial)"):
#     rows.append(run_one_cv(p["subsample"], p["colsample_bytree"])) # Run tuning function and store results
# # Convert results into data frame
# sc_results = (pd.DataFrame(rows)
#                 .sort_values(['test_error','test_auc'], ascending=[True, False])
#                 .reset_index(drop=True))
# # # View results
# # display(sc_results.head(10))
# # Identify best results
# best_sc = sc_results.iloc[0].to_dict()
# # Store best results
# print(f"Selected subsample={best_sc['subsample']}, "
#       f"colsample_bytree={best_sc['colsample_bytree']} "
#       f"(min test_error={best_sc['test_error']:.6f}, AUC={best_sc['test_auc']:.6f}, "
#       f"best_round={best_sc['best_round']})")

# tuned_subsample = float(best_sc['subsample']) # Extract best subsample
# tuned_colsample = float(best_sc['colsample_bytree']) # Extract best colsample_bytree

# # Set ETA values to try
# etas = [0.3, 0.1, 0.05, 0.01, 0.005]
# # Set base parameters
# base_params = {
#     "objective": "binary:logistic",
#     "eval_metric": ["auc", "error"],
#     "max_depth": tuned_max_depth, # Use tuned value for max depth
#     "min_child_weight": tuned_min_child, # Use tuned value for min_child_weight
#     "gamma": tuned_gamma, # Use tuned value for gamma
#     "subsample": tuned_subsample, # Use tuned value for subsample
#     "colsample_bytree": tuned_colsample, # Use tuned value for colsample_bytree
#     "tree_method": "hist",
#     "seed": 111111,
#     "nthread": 1,                  # single core
# }

# ### Be careful this can take a long time to run ###
# curves = []     # per-iteration logs for plotting
# summaries = []  # one row per eta
# # For each learning rate
# for eta in tqdm(etas, desc="Learning-rate CV (serial)"):
#     params = base_params.copy() # Create copy of parmameters
#     params["eta"] = float(eta) # Update ETA value
#     # Apply xgb.cv
#     cv = xgb.cv(
#         params=params, # Set parameters
#         dtrain=dtrain_w, # Set training data
#         num_boost_round=1000,  # run to 1000 unless ES stops early
#         nfold=5, # Set folds for cross validation
#         early_stopping_rounds=20, # Set early stopping rounds
#         stratified=True,
#         shuffle=True,
#         verbose_eval=False,
#         seed=111111,
#     )

#     # Extract data for model performance
#     df_log = cv.reset_index().rename(columns={"index": "iter"})
#     df_log["iter"] = df_log["iter"] + 1 # Increment iterations to get real number
#     # fix hyphenated column names for plotnine
#     df_log = df_log.rename(columns=lambda c: c.replace("-", "_"))
#     df_log["eta"] = str(eta) # Store ETA value as a string
#     curves.append(df_log) # Add values to data store

#     # Identify best iteration
#     best_round = len(cv)
#     best_row = cv.iloc[best_round - 1] # Identify best row


#     best_err = float(best_row["test-error-mean"]) # Extract best error value
#     best_auc = float(best_row["test-auc-mean"]) # Extract best AUC value
#     # Store results
#     summaries.append({"eta": eta, "best_round": best_round, "test_error": best_err, "test_auc": best_auc})

# # Combine curve data
# curves_df = pd.concat(curves, ignore_index=True)
# # Create data frame of result data
# summ_df = pd.DataFrame(summaries).sort_values(
#     ["test_error","test_auc"] ,
#     ascending=[True, False]
# ).reset_index(drop=True)

# best_eta = float(summ_df.iloc[0]["eta"]) # Extract best learning rate
# best_round = int(summ_df.iloc[0]["best_round"]) # Extract best round
# print(f"Selected eta={best_eta} with best_round={best_round}, " # Print results
#       f"test_error={summ_df.iloc[0]['test_error']:.6f}, "
#       f"AUC={summ_df.iloc[0]['test_auc']:.6f}")

# tuned_eta = float(best_eta) # Extract best learning rate



# print(
#     f"\nFinal tuned hyperparameters:\n"
#     f"  max_depth        = {tuned_max_depth}\n"
#     f"  min_child_weight = {tuned_min_child}\n"
#     f"  gamma            = {tuned_gamma}\n"
#     f"  subsample        = {tuned_subsample}\n"
#     f"  colsample_bytree = {tuned_colsample}\n"
#     f"  eta              = {tuned_eta}\n"
#     f"  best_round       = {best_round}"
# )













# Best Parameters & Weighted
# DO NOT RUN THE TUNING PARAMETERS IT TAKES AROUND 1 HOUR ON CRC

max_depth        = 10
min_child_weight = 5
gamma            = 0.0
subsample        = 0.6
colsample_bytree = 0.6
eta              = 0.3
best_round       = 5








# Count values
counts = pd.Series(y_train).value_counts().sort_index()
neg = int(counts.get(0, 0)); pos = int(counts.get(1, 0)) # Calculate positive and negative samples
print(f"Number of negative samples: {neg}")
print(f"Number of positive samples: {pos}")

# Calculate ratio
ratio = neg / pos
# Set ratio as weight for positive samples
w_tr = np.where(y_train == 1, ratio, 1.0).astype(np.float32)

# Build weighted DMatrices
dtrain_w = xgb.DMatrix(X_train.values, label=y_train, weight=w_tr)

params = {
    "objective": "binary:logistic",
    "eval_metric": ["auc", "error"],
    "max_depth": max_depth, # Use tuned value for max depth
    "min_child_weight": min_child_weight, # Use tuned value for min_child_weight
    "gamma": gamma, # Use tuned value for gamma
    "subsample": subsample, # Use tuned value for subsample
    "colsample_bytree": colsample_bytree, # Use tuned value for colsample_bytree
    "eta": eta, # Use tuned value for eta
    "tree_method": "hist",
    "seed": 111111,
    "nthread": 1,                  # single core
}

num_boost_round = best_round # Set number of rounds

watchlist = [(dtrain_w, "train")] # Set data for evaluation
xgb_tuned = xgb.train(params, # Set parameters
                    dtrain_w,  # Set training data
                    num_boost_round=num_boost_round, # Set number of rounds
                    evals=watchlist,  # Set data to evaluate on
                    verbose_eval=50) # Set print out frequency

test_pred_w = xgb_tuned.predict(dtest) # Create predictions


# Convert predictions into classes at 0.5
test_pred_cls_w = (test_pred_w >= 0.5).astype(int)


print("\nConfusion matrix:")
cm = (confusion_matrix(y_test, test_pred_cls_w))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)# Set class labels
disp.plot(cmap="Blues") # Set color map
plt.title("Confusion Matrix — Weighted XGBoost") # Set title
plt.savefig("confusion_matrix_weighted_and_tuned.png", dpi=300, bbox_inches="tight")
plt.close()
print("\nAccuracy):")
print(accuracy_score(y_test, test_pred_cls_w)) # Get Accuracy


# SHAP values to see importance of each column in the weighted tuned XGBoost model

# Create TreeExplainer and compute SHAP values
explainer = shap.TreeExplainer(xgb_tuned)
shap_values = explainer(X_train)

plt.figure()  # start a clean figure
shap.plots.bar(shap_values, max_display=10)
plt.title("Top 10 SHAP Feature Importances")   # optional title
plt.savefig("shap_bar_weighted_and_tuned.png", dpi=300, bbox_inches="tight")
plt.close()

# Initialize JS (interactive if you're in notebook—safe to leave)
shap.initjs()

# Create and save beeswarm plot
plt.figure(figsize=(10, 8))   # optional: wider figure
shap.plots.beeswarm(shap_values, max_display=25)
plt.title("SHAP Beeswarm — Weighted & Tuned XGBoost")  # optional title
plt.savefig("shap_beeswarm_weighted_and_tuned.png", dpi=300, bbox_inches="tight")
plt.close()

