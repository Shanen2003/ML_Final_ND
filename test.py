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















# Logistic Regression with Lasso



# df['Normal_Attack'] = df['Normal_Attack'].map({'Normal': 0, 'Attack': 1}).astype('float32')
y = df["Normal_Attack"].to_numpy()
X = df.drop(columns=["Normal_Attack", 'Timestamp'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
features = X.columns

x_data = pd.DataFrame(X_scaled, columns=features)


# # Manually Select Lasso Alpha
# lasso = Lasso(alpha=0.15, fit_intercept=True, max_iter=10000)
# lasso.fit(X_scaled, y)

# Or allow Cross-Validation to select Lasso Alpha
lambda_seq = np.arange(0.1, 10.0 + 1e-12, 0.1)
cv_model = LassoCV(alphas=lambda_seq, cv=10, fit_intercept=True, max_iter=10000)
cv_model.fit(X_scaled, y)
alpha_min = cv_model.alpha_
lasso = Lasso(alpha=alpha_min, fit_intercept=True, max_iter=10000)
lasso.fit(X_scaled, y)


# OLS Regression with Lasso
coef_series = pd.Series(lasso.coef_, index=features, name="coef")
coef_df = coef_series.reset_index()

selected_features = coef_series[coef_series != 0].index.tolist()

predictors = [f'Q("{c}")' for c in selected_features]
formula = 'Q("Normal_Attack") ~ ' + " + ".join(predictors)

fit_3 = smf.logit(formula = formula, data = df).fit()

print(fit_3.summary())
# logistic regression predictions and accuracy
logit_pred_prob = fit_3.predict(df_test[selected_features])
logit_pred = (logit_pred_prob >= 0.5).astype(int)
logit_acc = accuracy_score(y_test, logit_pred)
print(f"\nLogistic Regression Accuracy: {logit_acc:.4f}")






# Bagging and random forest 

# Drop NaN rows :(
train_mask = ~X_train.isna().any(axis=1)
X_train = X_train[train_mask]
y_train = y_train[train_mask]

test_mask = ~X_test.isna().any(axis=1)
X_test = X_test[test_mask]
y_test = y_test[test_mask]

tree_model = DecisionTreeClassifier(max_depth=4, random_state=123)  # Initialize tree
tree_model.fit(X_train, y_train) # Fit tree

# Encode labels for visualziation
le = preprocessing.LabelEncoder()
y_train_enc = le.fit_transform(y_train)

# Set up visualization
viz_model = dtreeviz.model(
    tree_model,
    X_train=X_train,
    y_train=y_train_enc,
    feature_names=list(X_train.columns),
    target_name="outcome",
    class_names=[str(c) for c in le.classes_]
)
v = viz_model.view(fontname="DejaVu Sans")
v.save("decision_tree_viz.svg") # Save visualization

y_pred = tree_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
# y_pred_enc = tree_model.predict(X_test) # Create predictions
# y_pred = le.inverse_transform(y_pred_enc) # Convert predicitons back to Win/Loss
acc = accuracy_score(y_test, y_pred) # Calcualte accuracy
print(f"\nDecision Tree Accuracy on Test Set: {acc:.4f}") # Print accuracy









# XGBoost

# Set up XGBDmatrix
dtrain = xgb.DMatrix(data=X_train.values, label=y_train)
dtest  = xgb.DMatrix(data=X_test.values,  label=y_test)

params = {
        "objective": "binary:logistic", # Set objective
        "eval_metric": ["auc", "error"],  # Track both AUC and error
        "seed": 42, # set seed

    }
num_boost_round = 5 # Set number of rounds

watchlist = [(dtrain, "train")] # Set data for evaluation
booster = xgb.train(params, # Set parameters
                    dtrain,  # Set training data
                    num_boost_round=num_boost_round, # Set number of rounds
                    evals=watchlist,  # Set data to evaluate on
                    verbose_eval=50) # Set print out frequency


test_pred_raw = booster.predict(dtest)

test_pred_cls = (test_pred_raw >= 0.5).astype(int)


print("\nConfusion matrix:")
cm = (confusion_matrix(y_test, test_pred_cls))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)# Set class labels
disp.plot(cmap="Blues") # Set color map
plt.title("Confusion Matrix — XGBoost") # Set title
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()
print("\nAccuracy):")
print(accuracy_score(y_test, test_pred_cls)) # Get classification report


# Now we will weight it!

# # Count values
# counts = pd.Series(y_train).value_counts().sort_index()
neg = int(counts.get(0, 0)); pos = int(counts.get(1, 0)) # Calculate positive and negative samples
# print(f"Number of negative samples: {neg}")
# print(f"Number of positive samples: {pos}")

# Calculate ratio
ratio = neg / pos
# Set ratio as weight for positive samples
w_tr = np.where(y_train == 1, ratio, 1.0).astype(np.float32)

# Build weighted DMatrices
dtrain_w = xgb.DMatrix(X_train.values, label=y_train, weight=w_tr)

watchlist = [(dtrain_w, "train")] # Set data for evaluation
xgb_w = xgb.train(params, # Set parameters
                    dtrain_w,  # Set training data
                    num_boost_round=3, # Set number of rounds
                    evals=watchlist,  # Set data to evaluate on
                    verbose_eval=50) # Set print out frequency

test_pred_w = xgb_w.predict(dtest) # Create predictions


# Convert predictions into classes at 0.5
test_pred_cls_w = (test_pred_w >= 0.5).astype(int)


print("\nConfusion matrix:")
cm = (confusion_matrix(y_test, test_pred_cls_w))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)# Set class labels
disp.plot(cmap="Blues") # Set color map
plt.title("Confusion Matrix — Weighted XGBoost") # Set title
plt.savefig("confusion_matrix_weighted.png", dpi=300, bbox_inches="tight")
plt.close()
print("\nAccuracy):")
print(accuracy_score(y_test, test_pred_cls_w)) # Get Accuracy


