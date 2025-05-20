#%% md
# ### Case Study Groupe 10
# Arne Herlinghaus, Max Lütkemeyer, Thomas Mogos, Tim Strauss, Simon Luttmann
#%% md
# # Final Model creation Workflow
# 
#%% md
# #### Imports
#%%
import random
import datetime
from pathlib import Path

import numpy as np, pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from xgboost import XGBClassifier
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

np.random.seed(42)
random.seed(42)

#%% md
# #### Load original data
#%%
train_csv = "data_6_channels_train.csv" # ← Pfad ggf. anpassen
df = (pd.read_csv(train_csv)
      .rename(columns={"numerical_id": "forest_id",
                       "class": "is_disturbance",
                       "BLU": "blue", "GRN": "green", "RED": "red",
                       "NIR": "near_infrared",
                       "SW1": "shortwave_infrared_1", "SW2": "shortwave_infrared_2"}))
#%% md
# #### Feature engineering
#%%
def engineer_features(group: pd.DataFrame, lags: int = 3) -> pd.DataFrame:
    """Spektrale Indizes + Landsat-8 TCT + 3-Jahres-Stats + Lags/Deltas + Raum/Zeit."""
    eps = 1e-6
    g = group.copy()

    # ---------- Basis-Indizes ----------------------------------------
    g["NDVI"] = (g.near_infrared - g.red) / (g.near_infrared + g.red + eps)
    g["NDMI"] = (g.near_infrared - g.shortwave_infrared_1) / (g.near_infrared + g.shortwave_infrared_1 + eps)
    g["NDWI"] = (g.green - g.near_infrared) / (g.green + g.near_infrared + eps)
    g["NBR"] = (g.near_infrared - g.shortwave_infrared_2) / (g.near_infrared + g.shortwave_infrared_2 + eps)
    g["EVI"] = 2.5 * (g.near_infrared - g.red) / (g.near_infrared + 6 * g.red - 7.5 * g.blue + 1 + eps)
    g["NBR2"] = (g.shortwave_infrared_1 - g.shortwave_infrared_2) / (
                g.shortwave_infrared_1 + g.shortwave_infrared_2 + eps)
    
    # ---------- Trockenheits-Ratio -----------------------------------
    g["SWIR_ratio"] = g.shortwave_infrared_2 / (g.shortwave_infrared_1 + eps)

    # ---------- Landsat-8 Tasseled-Cap (Baig 2014) -------------------
    b2, b3, b4 = g.blue, g.green, g.red
    b5, b6, b7 = g.near_infrared, g.shortwave_infrared_1, g.shortwave_infrared_2
    g["TCB"] = 0.3029 * b2 + 0.2786 * b3 + 0.4733 * b4 + 0.5599 * b5 + 0.5080 * b6 + 0.1872 * b7
    g["TCG"] = -0.2941 * b2 - 0.2430 * b3 - 0.5424 * b4 + 0.7276 * b5 + 0.0713 * b6 - 0.1608 * b7
    g["TCW"] = 0.1511 * b2 + 0.1973 * b3 + 0.3283 * b4 + 0.3407 * b5 - 0.7117 * b6 - 0.4559 * b7
    g["TCT4"] = -0.8239 * b2 + 0.0849 * b3 + 0.4396 * b4 - 0.0580 * b5 + 0.2013 * b6 - 0.2773 * b7
    g["TCT5"] = -0.3294 * b2 + 0.0557 * b3 + 0.1056 * b4 + 0.1855 * b5 - 0.4349 * b6 + 0.8085 * b7
    g["TCT6"] = 0.1079 * b2 - 0.9023 * b3 + 0.4119 * b4 + 0.0575 * b5 - 0.0259 * b6 + 0.0252 * b7

    # --- Interaktions-Features --------------------------------------
    g["TCBxTCG"] = g.TCB * g.TCG
    g["TCBxTCW"] = g.TCB * g.TCW
    g["TCBxNBR"] = g.TCB * g.NBR

    # ---------- 3-Jahres-Median, Std, Anomalie (NDVI & NBR) ----------
    for idx in ("NDVI", "NBR"):
        g[f"{idx}_med3"] = g[idx].rolling(3, min_periods=2).median()
        g[f"{idx}_std3"] = g[idx].rolling(3, min_periods=2).std()
        g[f"{idx}_anom"] = (g[idx] - g[f"{idx}_med3"]) / (g[f"{idx}_med3"].abs() + eps)

    # ---------- Lags & Deltas (blockweise, effizient) ---------------
    base_cols = [
        "NDVI", "NDMI", "NDWI", "NBR", "EVI", "NBR2", "SWIR_ratio",
        "TCB", "TCG", "TCW", "TCT4", "TCT5", "TCT6",
        "TCBxTCG", "TCBxTCW", "TCBxNBR",
        "blue", "green", "red", "near_infrared", "shortwave_infrared_1", "shortwave_infrared_2"
    ]

    lag_features = []
    for lag in range(1, lags + 1):
        shifted = g[base_cols].shift(lag).rename(columns=lambda col: f"l{lag}_{col}")
        deltas = (g[base_cols] - g[base_cols].shift(lag)).rename(columns=lambda col: f"d{lag}_{col}")
        lag_features.extend([shifted, deltas])
    g = pd.concat([g] + lag_features, axis=1)

    # ---------- Raum- & Zeit-Features --------------------------------
    g["year_sin"] = np.sin(2 * np.pi * g.year / 10)
    g["year_cos"] = np.cos(2 * np.pi * g.year / 10)

    return g
#%%
ft_file = "all_data_all_features.csv"

if not Path(ft_file).exists():
    feats = (df.groupby("forest_id")
             .apply(engineer_features)
             .fillna(method="ffill").fillna(method="bfill")
             .reset_index(drop=True))
    feats.to_csv(ft_file, index=False)
else:
    feats = pd.read_csv(ft_file)

feature_cols = [c for c in feats.columns
                if c not in ("forest_id", "year", "is_disturbance")]

#%% md
# #### Test|Validation 80|20 Split with Groups by forest_id
#%%
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
tr_idx, va_idx = next(gss.split(feats, groups=feats["forest_id"]))

train_df, val_df = feats.iloc[tr_idx], feats.iloc[va_idx]
X_train, y_train = train_df[feature_cols].values, train_df["is_disturbance"].values
X_val, y_val = val_df[feature_cols].values, val_df["is_disturbance"].values
print(f"Train {len(train_df):,} | Val {len(val_df):,}")

#%% md
# # ASK TIM
#%% md
# #### Train an XGBoost Ensemble with GroupKFolds based on the train set
#%%
#TODO: DESCRIPTION
#%%
np.random.seed(42)
# Define the parameter grid for tree sizes
param_grid = {
    'n_estimators': [2000],
    'max_depth': [5],
    'learning_rate': [0.05],
    'subsample': [0.5],
    #'colsample_bytree': [0.6, 0.8, 1.0],
    'scale_pos_weight': [3],
    'gamma': [ 0.3],
    'min_child_weight': [2],
}

search_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="aucpr",
    eta=0.05,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=2,
    tree_method="hist",
    random_state=42,
    early_stopping_rounds=60
)

group_kfold = GroupKFold(n_splits=3)
# Initialize the GridSearchCV object
grid_search = GridSearchCV(
    estimator=search_model,
    param_grid=param_grid,
    scoring='f1',
    cv=group_kfold,  # This will be replaced by GroupKFold below
    verbose=1,
    n_jobs=-1
)

# To use group-based cross-validation, you need to pass a GroupKFold splitter to the cv argument:


# grid_search.cv = group_kfold.split(X_train, y_train, groups=train_df['forest_id'])

# Perform the grid search
grid_search.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False, groups=train_df['forest_id'])

# Get the best parameters and the corresponding F1 score
best_params = grid_search.best_params_
best_f1_score = grid_search.best_score_

print(f"Best Parameters: {best_params}")
print(f"Best F1 Score: {best_f1_score}")
# Perform prediction on the validation set
y_pred = grid_search.best_estimator_.predict(X_val)

# Calculate F1 score and confusion matrix
f1 = f1_score(y_val, y_pred)
cm = confusion_matrix(y_val, y_pred)

print(f"F1 Score on Validation Set: {f1}")
print(f"Confusion Matrix on Validation Set:\n{cm}")
#%% md
# We choose the XGBoost Ensemble model because it made the best predictions on our validation set compared to other Models like XGBoost, RandomForest, ExtraTrees, Logistic Regression, KNeighrest Neighboor and SVM.
#%% md
# #### Inferenz on test dataset
#%%
TEST_CSV = "data_6_channels_test_pub.csv"
OUT_CSV = "data_6_channels_test_pub_with_predictions.csv"
MODEL_PKL = "models/xgb_iso_ensemble.pkl"

# 6.1  Original-CSV laden (ohne Spaltenänderung merken)
orig_test = pd.read_csv(TEST_CSV)

# 6.2  Für das Modell temporär umbenennen
tmp = (orig_test
       .rename(columns={
    "numerical_id": "forest_id",
    "BLU": "blue", "GRN": "green", "RED": "red",
    "NIR": "near_infrared", "SW1": "shortwave_infrared_1",
    "SW2": "shortwave_infrared_2"})
       .copy())

# 6.3  Feature-Engineering
test_feat = (tmp.groupby("forest_id")
             .apply(engineer_features)
             .fillna(method="ffill").fillna(method="bfill")
             .reset_index(drop=True))

# 6.4  Ensemble + Threshold laden
with open(MODEL_PKL, "rb") as f:
    art = pickle.load(f)

X_test = test_feat[art["feature_cols"]].values
proba = sum(m.predict_proba(X_test)[:, 1] for m in art["models"]) / len(art["models"])
pred = (proba >= art["threshold"]).astype(int)

# 6.5  Prediction-DataFrame zum Mergen vorbereiten
pred_df = (test_feat[["forest_id", "year"]]
           .assign(is_disturbance=pred)
           .rename(columns={"forest_id": "numerical_id"}))

# 6.6  Mit Original-CSV zusammenführen  (inner-merge garantiert 1-zu-1)
merged = (orig_test
          .merge(pred_df, on=["numerical_id", "year"], how="left"))

# 6.7  Spaltenreihenfolge (wie Bild + neue Spalte)
col_order = ["fid", "year", "numerical_id",
             "BLU", "GRN", "RED", "NIR", "SW1", "SW2",
             "is_disturbance"]
merged = merged[col_order]

# 6.8  Datei schreiben
merged.to_csv(OUT_CSV, index=False)
print(f"✓ '{OUT_CSV}' geschrieben – {merged.shape[0]:,} Zeilen")

# 6.9  Kurzer Überblick
print(f"Label-Verteilung: 0 = {(merged.is_disturbance == 0).sum():,}  |  "
      f"1 = {(merged.is_disturbance == 1).sum():,}")
