# =============================================================================
# HOME CREDIT DEFAULT RISK 
# =============================================================================

# =============================================================================
# 0. IMPORTS
# =============================================================================
import gc
import warnings
import joblib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    classification_report, roc_auc_score,
    roc_curve, confusion_matrix, ConfusionMatrixDisplay
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import shap

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 80)
pd.set_option('display.float_format', '{:.4f}'.format)
sns.set_theme(style='whitegrid', palette='muted')

RANDOM_STATE = 42
DATA_PATH    = './'   # <-- update this to your folder if needed

print('Libraries loaded ✓')


# =============================================================================
# 1. MEMORY UTILITIES
# =============================================================================

def reduce_mem(df):
    """Downcast numeric columns to smallest safe dtype."""
    for col in df.columns:
        col_type = df[col].dtype
        # skip object/string columns entirely
        if col_type == object or str(col_type) == 'string':
            continue
        try:
            c_min = df[col].min()
            c_max = df[col].max()
            # skip if min/max are not actual numbers
            if pd.isna(c_min) or pd.isna(c_max):
                continue
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            elif str(col_type)[:5] == 'float':
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
        except (TypeError, ValueError):
            # if anything goes wrong with a column just leave it as-is
            continue
    return df

# =============================================================================
# 2. DATA LOADING
# =============================================================================

print('\n--- Loading tables ---')

print('Loading application_train...')
app = read_compressed(f'{DATA_PATH}application_train.csv')
print(f'  {app.shape} | {app.memory_usage(deep=True).sum()/1e6:.0f} MB')

print('Loading bureau...')
bureau = read_compressed(f'{DATA_PATH}bureau.csv')
print(f'  {bureau.shape} | {bureau.memory_usage(deep=True).sum()/1e6:.0f} MB')

print('Loading previous_application...')
prev = read_compressed(f'{DATA_PATH}previous_application.csv')
print(f'  {prev.shape} | {prev.memory_usage(deep=True).sum()/1e6:.0f} MB')

print('Loading installments (3M sample)...')
inst = read_compressed(f'{DATA_PATH}installments_payments.csv', sample_n=3_000_000)
print(f'  {inst.shape} | {inst.memory_usage(deep=True).sum()/1e6:.0f} MB')

gc.collect()
print('\nAll tables loaded ✓')


# =============================================================================
# 3. EXPLORATORY DATA ANALYSIS
# =============================================================================

print('\n--- EDA ---')

# 3.1 Target distribution
target_counts = app['TARGET'].value_counts()
target_pct    = app['TARGET'].value_counts(normalize=True) * 100

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(['Repaid (0)', 'Defaulted (1)'], target_counts.values,
            color=['#2ecc71', '#e74c3c'], edgecolor='white', linewidth=1.5)
axes[0].set_title('Target Distribution — Counts', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Number of Applications')
for i, v in enumerate(target_counts.values):
    axes[0].text(i, v + 500, f'{v:,}', ha='center', fontweight='bold')
axes[1].pie(target_pct.values, labels=['Repaid', 'Defaulted'],
            colors=['#2ecc71', '#e74c3c'], autopct='%1.1f%%',
            startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
axes[1].set_title('Target Distribution — Proportion', fontsize=13, fontweight='bold')
plt.suptitle('Class Imbalance: ~8% Default Rate — SMOTE Required',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('plot_01_target_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print(f'Default rate: {target_pct[1]:.1f}%')

# 3.2 Key feature distributions
plot_features = [
    ('AMT_CREDIT',       'Loan Amount'),
    ('AMT_INCOME_TOTAL', 'Annual Income'),
    ('DAYS_BIRTH',       'Age (Days, negative)'),
    ('DAYS_EMPLOYED',    'Days Employed (negative)'),
    ('AMT_ANNUITY',      'Annual Repayment'),
    ('EXT_SOURCE_2',     'External Credit Score 2'),
]

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, (col, label) in zip(axes.flatten(), plot_features):
    cap = app[col].quantile(0.99)
    for val, color, lbl in [(0, '#2ecc71', 'Repaid'), (1, '#e74c3c', 'Defaulted')]:
        d = app.loc[app['TARGET'] == val, col].dropna()
        d = d[d <= cap]
        ax.hist(d, bins=50, alpha=0.6, color=color, label=lbl, density=True)
    ax.set_title(label, fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_yticks([])
plt.suptitle('Feature Distributions by Default Status', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plot_02_feature_distributions.png', dpi=150, bbox_inches='tight')
plt.show()

# 3.3 External credit scores
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']):
    for val, color, lbl in [(0, '#2ecc71', 'Repaid'), (1, '#e74c3c', 'Defaulted')]:
        ax.hist(app.loc[app['TARGET'] == val, col].dropna(),
                bins=40, alpha=0.6, color=color, label=lbl, density=True)
    ax.set_title(col, fontweight='bold')
    ax.legend()
    ax.set_yticks([])
plt.suptitle('External Credit Bureau Scores — Lower Score = Higher Default Risk',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plot_03_ext_scores.png', dpi=150, bbox_inches='tight')
plt.show()

# 3.4 Missing values
missing = (app.isnull().sum() / len(app) * 100)
missing = missing[missing > 0].sort_values(ascending=False).head(20)
fig, ax = plt.subplots(figsize=(10, 6))
missing.plot(kind='barh', ax=ax, color='#3498db', edgecolor='white')
ax.set_xlabel('Missing (%)')
ax.set_title('Top 20 Features by Missing Rate', fontsize=13, fontweight='bold')
ax.axvline(40, color='red', linestyle='--', alpha=0.7, label='40% drop threshold')
ax.legend()
plt.tight_layout()
plt.savefig('plot_04_missing_values.png', dpi=150, bbox_inches='tight')
plt.show()

print('EDA complete ✓')


# =============================================================================
# 4. FEATURE ENGINEERING
# =============================================================================

print('\n--- Feature Engineering ---')

# 4.1 Application features
df = app.copy()

df['DEBT_TO_INCOME']    = df['AMT_CREDIT']   / (df['AMT_INCOME_TOTAL'] + 1)
df['ANNUITY_TO_INCOME'] = df['AMT_ANNUITY']  / (df['AMT_INCOME_TOTAL'] + 1)
df['CREDIT_TO_GOODS']   = df['AMT_CREDIT']   / (df['AMT_GOODS_PRICE']  + 1)
df['AGE_YEARS']         = -df['DAYS_BIRTH']  / 365
df['DAYS_EMPLOYED']     = df['DAYS_EMPLOYED'].replace(365243, np.nan)
df['EMPLOYED_YEARS']    = -df['DAYS_EMPLOYED'] / 365
df['EMPLOYMENT_TO_AGE'] = df['EMPLOYED_YEARS'] / (df['AGE_YEARS'] + 1)
df['EXT_SOURCE_MEAN']   = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
df['EXT_SOURCE_MIN']    = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].min(axis=1)
df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'] + 1)

print('Application features engineered ✓')

# 4.2 Bureau aggregation
bureau_agg = bureau.groupby('SK_ID_CURR').agg(
    BUREAU_LOAN_COUNT       = ('SK_ID_BUREAU',           'count'),
    BUREAU_ACTIVE_LOANS     = ('CREDIT_ACTIVE',          lambda x: (x == 'Active').sum()),
    BUREAU_AMT_CREDIT_SUM   = ('AMT_CREDIT_SUM',         'sum'),
    BUREAU_AMT_CREDIT_DEBT  = ('AMT_CREDIT_SUM_DEBT',    'sum'),
    BUREAU_AMT_OVERDUE      = ('AMT_CREDIT_SUM_OVERDUE', 'sum'),
    BUREAU_MAX_OVERDUE_DAYS = ('CREDIT_DAY_OVERDUE',     'max'),
    BUREAU_AVG_OVERDUE_DAYS = ('CREDIT_DAY_OVERDUE',     'mean'),
    BUREAU_CLOSED_LOANS     = ('CREDIT_ACTIVE',          lambda x: (x == 'Closed').sum()),
).reset_index()

bureau_agg['BUREAU_DEBT_RATIO'] = (
    bureau_agg['BUREAU_AMT_CREDIT_DEBT'] /
    (bureau_agg['BUREAU_AMT_CREDIT_SUM'] + 1)
)

df = df.merge(bureau_agg, on='SK_ID_CURR', how='left')
del bureau, bureau_agg
gc.collect()
print(f'Bureau merged ✓  |  {df.shape}')

# 4.3 Previous applications aggregation
prev_agg = prev.groupby('SK_ID_CURR').agg(
    PREV_APP_COUNT       = ('SK_ID_PREV',           'count'),
    PREV_APPROVED        = ('NAME_CONTRACT_STATUS', lambda x: (x == 'Approved').sum()),
    PREV_REFUSED         = ('NAME_CONTRACT_STATUS', lambda x: (x == 'Refused').sum()),
    PREV_AMT_CREDIT_MAX  = ('AMT_CREDIT',           'max'),
    PREV_AMT_CREDIT_MEAN = ('AMT_CREDIT',           'mean'),
    PREV_RATE_MEAN       = ('RATE_DOWN_PAYMENT',    'mean'),
).reset_index()

prev_agg['PREV_REFUSAL_RATE'] = (
    prev_agg['PREV_REFUSED'] / (prev_agg['PREV_APP_COUNT'] + 1)
)

df = df.merge(prev_agg, on='SK_ID_CURR', how='left')
del prev, prev_agg
gc.collect()
print(f'Previous apps merged ✓  |  {df.shape}')

# 4.4 Instalment payment aggregation
inst['DAYS_LATE']     = inst['DAYS_ENTRY_PAYMENT'] - inst['DAYS_INSTALMENT']
inst['PAYMENT_RATIO'] = inst['AMT_PAYMENT'] / (inst['AMT_INSTALMENT'] + 1)
inst['UNDERPAID']     = (inst['AMT_PAYMENT'] < inst['AMT_INSTALMENT']).astype(int)

inst_agg = inst.groupby('SK_ID_CURR').agg(
    INST_COUNT           = ('SK_ID_PREV',    'count'),
    INST_DAYS_LATE_MEAN  = ('DAYS_LATE',     'mean'),
    INST_DAYS_LATE_MAX   = ('DAYS_LATE',     'max'),
    INST_PAYMENT_RATIO   = ('PAYMENT_RATIO', 'mean'),
    INST_UNDERPAID_COUNT = ('UNDERPAID',     'sum'),
    INST_UNDERPAID_RATE  = ('UNDERPAID',     'mean'),
).reset_index()

df = df.merge(inst_agg, on='SK_ID_CURR', how='left')
del inst, inst_agg
gc.collect()
print(f'Instalments merged ✓  |  {df.shape}')

# 4.5 Final preprocessing
missing_rates = df.isnull().sum() / len(df)
cols_to_drop  = [c for c in missing_rates[missing_rates > 0.4].index if c != 'TARGET']
df = df.drop(columns=cols_to_drop)
print(f'Dropped {len(cols_to_drop)} high-missing columns')

df = df.drop(columns=['SK_ID_CURR'])

cat_cols = df.select_dtypes(include=['object']).columns.tolist()
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
print(f'One-hot encoded {len(cat_cols)} categorical columns')

df = df.fillna(df.median(numeric_only=True))
df = reduce_mem(df)

gc.collect()
print(f'\nFinal dataset : {df.shape[0]:,} rows x {df.shape[1]} features')
print(f'Missing values: {df.isnull().sum().sum()}')


# =============================================================================
# 5. MODELLING
# =============================================================================

print('\n--- Modelling ---')

# 5.1 Split
X = df.drop(columns='TARGET')
y = df['TARGET']

del df
gc.collect()

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.15, stratify=y_train_full, random_state=RANDOM_STATE
)

del X_train_full, y_train_full
gc.collect()

print(f'Train : {X_train.shape[0]:,} rows | Default rate: {y_train.mean():.2%}')
print(f'Val   : {X_val.shape[0]:,} rows | Default rate: {y_val.mean():.2%}')
print(f'Test  : {X_test.shape[0]:,} rows | Default rate: {y_test.mean():.2%}')

# 5.2 SMOTE — training data only
sm = SMOTE(random_state=RANDOM_STATE, sampling_strategy=0.3)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

del sm
gc.collect()

print(f'\nAfter SMOTE:')
print(y_train_res.value_counts())
print(f'New default rate: {y_train_res.mean():.2%}')

# 5.3 XGBoost — RandomizedSearchCV
xgb = XGBClassifier(
    random_state=RANDOM_STATE,
    eval_metric='logloss',
    tree_method='hist',
    n_jobs=1
)

param_dist = {
    'n_estimators'     : [200, 300],
    'max_depth'        : [3, 5],
    'learning_rate'    : [0.05, 0.1],
    'subsample'        : [0.8, 1.0],
    'colsample_bytree' : [0.8, 1.0],
    'min_child_weight' : [1, 5],
}

gc.collect()

search = RandomizedSearchCV(
    xgb,
    param_dist,
    n_iter=15,
    cv=3,
    scoring='roc_auc',
    n_jobs=1,
    random_state=RANDOM_STATE,
    verbose=1
)

search.fit(X_train_res, y_train_res)

best_xgb = search.best_estimator_
print(f'\nBest params : {search.best_params_}')
print(f'CV ROC-AUC  : {search.best_score_:.4f}')

del X_train_res, y_train_res
gc.collect()

# 5.4 Test set evaluation
y_test_proba = best_xgb.predict_proba(X_test)[:, 1]
y_test_pred  = best_xgb.predict(X_test)

print('\n=== Default threshold (0.5) ===')
print(classification_report(y_test, y_test_pred, target_names=['Repaid', 'Defaulted']))
print(f'ROC-AUC: {roc_auc_score(y_test, y_test_proba):.4f}')

fpr, tpr, _ = roc_curve(y_test, y_test_proba)
auc_score   = roc_auc_score(y_test, y_test_proba)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(fpr, tpr, color='#3498db', lw=2, label=f'XGBoost (AUC = {auc_score:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random classifier')
ax.fill_between(fpr, tpr, alpha=0.1, color='#3498db')
ax.set_xlabel('False Positive Rate (Good clients rejected)')
ax.set_ylabel('True Positive Rate (Defaulters caught)')
ax.set_title('ROC Curve — XGBoost Credit Default Model', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('plot_05_roc_curve.png', dpi=150, bbox_inches='tight')
plt.show()


# =============================================================================
# 6. THRESHOLD ANALYSIS — BUSINESS DECISION FRAMEWORK
# =============================================================================

print('\n--- Threshold Analysis ---')

y_val_proba              = best_xgb.predict_proba(X_val)[:, 1]
fpr_v, tpr_v, thresh_v   = roc_curve(y_val, y_val_proba)

LGD      = 0.45
INTEREST = 0.08
AVG_LOAN = app['AMT_CREDIT'].median() if 'AMT_CREDIT' in dir() else 450_000

cost_fn = LGD * AVG_LOAN
cost_fp = INTEREST * AVG_LOAN

print(f'Cost per False Negative : R{cost_fn:,.0f}  (missed default)')
print(f'Cost per False Positive : R{cost_fp:,.0f}  (rejected good client)')
print(f'FN is {cost_fn/cost_fp:.1f}x more costly than FP')

thresholds_sweep = np.arange(0.05, 0.95, 0.01)
total_costs = []

for t in thresholds_sweep:
    y_pred_t = (y_val_proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred_t).ravel()
    total_costs.append((fn * cost_fn) + (fp * cost_fp))

optimal_idx         = np.argmin(total_costs)
cost_optimal_thresh = thresholds_sweep[optimal_idx]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(thresholds_sweep, total_costs, color='#e74c3c', lw=2)
ax.axvline(cost_optimal_thresh, color='#2ecc71', linestyle='--', lw=2,
           label=f'Optimal = {cost_optimal_thresh:.2f}')
ax.axvline(0.5, color='grey', linestyle=':', lw=1.5, label='Default = 0.50')
ax.set_xlabel('Classification Threshold')
ax.set_ylabel('Total Expected Cost')
ax.set_title('Threshold Optimisation — Minimising Expected Credit Loss', fontweight='bold')
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'R{x/1e6:.0f}M'))
ax.legend()
plt.tight_layout()
plt.savefig('plot_06_threshold_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print(f'Cost-optimal threshold: {cost_optimal_thresh:.2f}')

# Apply to test set
y_pred_optimal = (y_test_proba >= cost_optimal_thresh).astype(int)

print(f'\n=== Cost-optimal threshold ({cost_optimal_thresh:.2f}) ===')
print(classification_report(y_test, y_pred_optimal, target_names=['Repaid', 'Defaulted']))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, (preds, title) in zip(axes, [
    (y_test_pred,    'Default Threshold (0.50)'),
    (y_pred_optimal, f'Optimal Threshold ({cost_optimal_thresh:.2f})')
]):
    cm   = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Repaid', 'Defaulted'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(title, fontweight='bold')
plt.suptitle('Confusion Matrix Comparison', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plot_07_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()


# =============================================================================
# 7. SHAP EXPLAINABILITY
# =============================================================================

print('\n--- SHAP Explainability ---')

X_shap      = X_test.sample(n=min(2000, len(X_test)), random_state=RANDOM_STATE)
explainer   = shap.TreeExplainer(best_xgb)
shap_values = explainer.shap_values(X_shap)
print('SHAP values computed ✓')

# Global summary
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_shap, max_display=20, show=False)
plt.title('SHAP Summary — Top 20 Default Drivers', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plot_08_shap_summary.png', dpi=150, bbox_inches='tight')
plt.show()

# Bar plot
plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values, X_shap, plot_type='bar', max_display=15, show=False)
plt.title('Mean |SHAP| — Average Feature Impact on Default Probability',
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plot_09_shap_bar.png', dpi=150, bbox_inches='tight')
plt.show()

# Individual waterfall — highest risk applicant
high_risk_idx = np.where(y_test_proba > 0.7)[0]
if len(high_risk_idx) > 0:
    idx = high_risk_idx[0]
    print(f'\nApplicant index               : {idx}')
    print(f'Predicted default probability : {y_test_proba[idx]:.2%}')
    print(f'Actual outcome                : {"Defaulted" if y_test.iloc[idx]==1 else "Repaid"}')
    shap.plots.waterfall(
        shap.Explanation(
            values        = shap_values[idx],
            base_values   = explainer.expected_value,
            data          = X_shap.iloc[idx],
            feature_names = X_shap.columns.tolist()
        ),
        max_display=12,
        show=True
    )


# =============================================================================
# 8. SAVE MODEL
# =============================================================================

print('\n--- Saving model ---')

joblib.dump(best_xgb,                 'home_credit_xgb_model.pkl')
joblib.dump(cost_optimal_thresh,      'home_credit_threshold.pkl')
joblib.dump(X_train.columns.tolist(), 'home_credit_features.pkl')

print('Saved: home_credit_xgb_model.pkl')
print('Saved: home_credit_threshold.pkl')
print('Saved: home_credit_features.pkl')


def predict_default_risk(X_new):
    """Production scoring function. Returns probability, decision, and label."""
    model     = joblib.load('home_credit_xgb_model.pkl')
    threshold = joblib.load('home_credit_threshold.pkl')
    proba     = model.predict_proba(X_new)[:, 1]
    decision  = (proba >= threshold).astype(int)
    return pd.DataFrame({
        'default_probability' : proba,
        'decision'            : decision,
        'decision_label'      : np.where(decision == 1, 'REJECT', 'APPROVE')
    })


print('\npredict_default_risk() ready ✓')
print('\n=== Pipeline complete ✓ ===')
