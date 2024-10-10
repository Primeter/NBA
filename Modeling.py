# %%
import pandas as pd
from collections import Counter
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import balanced_accuracy_score,make_scorer, f1_score, roc_auc_score,accuracy_score
from sklearn.model_selection import train_test_split,GridSearchCV, RepeatedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
# %% Load data
initial_data = pd.read_csv('NBA_Data/PlayersStats_SevereInjuries_clean.csv')
protected_data_S = pd.read_csv('NBA_Data/Protected_NBA_Sup.csv')
protected_data_G = pd.read_csv('NBA_Data/Protected_NBA_Gen.csv')
protected_data_SN = pd.read_csv('NBA_Data/Protected_NBA_SupNoi.csv')
protected_data_SG = pd.read_csv('NBA_Data/Protected_NBA_SupGen.csv')
protected_data_GN = pd.read_csv('NBA_Data/Protected_NBA_GenNoi.csv')
protected_data_SGN = pd.read_csv('NBA_Data/Protected_NBA_SupGenNoi.csv')


# %% Functions to prepare data
# re-index columns in the protected data sets
def reindex_cols(df):
    cols = df.columns.tolist()
    cols.insert(len(cols) - 1, cols.pop(cols.index('SevereInjury')))
    df = df.reindex(columns=cols)

    return df


def dealNaN(df):
    # df.isna().any()
    categorical_vars = df.select_dtypes(include=['object']).columns.tolist()
    numerical_vars = df.select_dtypes(include=['float']).columns.tolist()
    if categorical_vars:
        df[categorical_vars] = df[categorical_vars].fillna('na', inplace=False)
    if numerical_vars:
        df[numerical_vars] = df[numerical_vars].fillna(-999, inplace=False)
    return df


def dummies(df):
    # One-hot encode the data using pandas get_dummies
    df = pd.get_dummies(df)
    return df


# %% Functions to modeling
def load_data(df):
    # split into input and output elements
    df_val = df.values
    X, y = df_val[:, :-1], df_val[:, -1]
    # label encode the target variable to have the classes 0, 1 and 2
    y = LabelEncoder().fit_transform(y)

    return X, y


# evaluate a model
def evaluate_model(X, y):
    # split data 80/20
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)

    seed = 42
    rfc = RandomForestClassifier(random_state=seed)
    bc = BaggingClassifier(random_state=seed)
    xgb = XGBClassifier(objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=seed)
    svm = SVC(probability=True)

    # set parameters
    params = [
        {'classifier__n_estimators': [50, 100, 200, 500, 600],
        'classifier__max_depth': [4, 6, 8, 10],
        'classifier':[rfc]},
        {'classifier__n_estimators': [50, 100, 200, 500, 600],
        'classifier':[bc]},
        {'classifier__n_estimators': [50, 100, 200, 500, 600],
        'classifier__max_depth': [4, 6, 8, 10],
        'classifier__learning_rate': [0.1, 0.01, 0.001],
        'classifier':[xgb]},
        {'classifier__gamma': [1e-2, 1e-3, 1e-4, 1e-5],
        'classifier__C': [1, 10, 100],
        'classifier':[svm]}
    ]

    # define metric functions
    #scoring = ['accuracy', 'balanced_accuracy', 'f1_weighted', 'roc_auc_ovr_weighted', 'roc_auc_ovo_weighted']
    scoring = {
        'acc': 'accuracy',
        'bal_acc': 'balanced_accuracy',
        'f1': 'f1',
        'f1_weighted': 'f1_weighted',
        'roc_auc_weighted': 'roc_auc_ovr_weighted'
        }
    
    pipeline = Pipeline([('classifier', rfc)])
    
    print("Start modeling with CV")
    # Train the grid search model
    gs = GridSearchCV(
        pipeline,
        param_grid=params,
        cv=RepeatedKFold(n_splits=5, n_repeats=2),
        scoring=scoring,
        refit='acc',
        return_train_score=True,
        n_jobs=-1).fit(X_train, y_train)

    score_cv = {
    'params':[], 'model':[],
    'test_accuracy': [], 'test_balanced_accuracy': [], 'test_f1_weighted':[], 'test_roc_auc_balanced':[]
    }
    # Store results from grid search
    validation = pd.DataFrame(gs.cv_results_)

    validation['model'] = validation['param_classifier']
    validation['model'] = validation['model'].apply(lambda x: 'Random Forest' if 'RandomForest' in str(x) else x)
    validation['model'] = validation['model'].apply(lambda x: 'XGBoost' if 'XGB' in str(x) else x)
    validation['model'] = validation['model'].apply(lambda x: 'SVC' if 'SVM' in str(x) else x)
    validation['model'] = validation['model'].apply(lambda x: 'Bagging' if 'Bagging' in str(x) else x)

    print("Start modeling in out of sample")

    for i in range(len(validation)):
        # set each model for prediction on test
        clf_best = gs.best_estimator_.set_params(**gs.cv_results_['params'][i]).fit(X_train, y_train)
        clf = clf_best.predict(X_test)
        # Predict probabilities for ROC AUC
        clf_pred_proba = clf_best.predict_proba(X_test)

        score_cv['params'].append(str(gs.cv_results_['params'][i]))
        score_cv['model'].append(validation.loc[i, 'model'])
        score_cv['test_accuracy'].append(accuracy_score(y_test, clf))
        score_cv['test_f1_weighted'].append(f1_score(y_test, clf, average='weighted'))
        score_cv['test_balanced_accuracy'].append(balanced_accuracy_score(y_test, clf))
        score_cv['test_roc_auc_balanced'].append(roc_auc_score(y_test, clf_pred_proba, multi_class='ovr', average='weighted'))

    score_cv = pd.DataFrame(score_cv)

    return [validation, score_cv]


def save_results(file, results):
    """Create a folder if dooes't exist and save results

    Args:
        file (string): file name
        results (list of Dataframes): results for cross validation and out of sample
    """
    output_folder_val = ('results_modeling/validation')
    output_folder_test = ('results_modeling/test')
    if not os.path.exists(output_folder_val): os.makedirs(output_folder_val)
    if not os.path.exists(output_folder_test): os.makedirs(output_folder_test)

    results[0].to_csv(f'{output_folder_val}/{file}.csv', index=False)
    results[1].to_csv(f'{output_folder_test}/{file}.csv', index=False)


# %% Prepare baseline
# check NaN
initial_data.isnull().sum()
initial_data = dealNaN(initial_data)
initial_data.isnull().sum()

# deal with categorical attributes
initial_data = dummies(initial_data)
initial_data = reindex_cols(initial_data)

# summarize target distribution (in the initial data set)
target = initial_data.values[:, -1]
counter = Counter(target)
for k, v in counter.items():
    per = v / len(target) * 100
    print('Class=%d, Count=%d, Percentage=%.3f%%' % (k, v, per))

# %%
bins = [0, 1, 2, 3]  # 4 bin edges for 3 categories
names = ['Not injury', 'Minor injury', 'Severe injury']  # 3 labels

target = pd.cut(target, bins=bins, labels=names)


# %%
sns.set_style("darkgrid")
plt.figure(figsize=(6, 4))
# Create the bar plot
ax=sns.histplot(target,shrink=0.75)

# Add labels
plt.xlabel('')
plt.ylabel('Number of players')
ax.margins(x=0.02)
ax.set_yticks(np.arange(0, 2100, 500))
# plt.xticks(rotation=45)
sns.set(font_scale=1.3)
# Display the plot
plt.tight_layout()
# plt.show()
#plt.savefig(f'Plots/target.pdf', bbox_inches='tight')
# %% Baseline
X, y = load_data(initial_data)
results = evaluate_model(X, y)
save_results('orig', results)

# %%
# %% Suppression
protected_data_S.isnull().sum()

# deal with categorical attributes
protected_data_S = dummies(protected_data_S)
protected_data_S = reindex_cols(protected_data_S)

X, y = load_data(protected_data_S)
sup = evaluate_model(X, y)
save_results('sup', sup)

# %% Generalization
protected_data_G.isnull().sum()
protected_data_G = dealNaN(protected_data_G)
protected_data_G.isnull().sum()

# deal with categorical attributes
protected_data_G = dummies(protected_data_G)
protected_data_G = reindex_cols(protected_data_G)

X, y = load_data(protected_data_G)
gen = evaluate_model(X, y)
save_results('gen', gen)

# %% Suppression and noise
protected_data_SN.isnull().sum()

# deal with categorical attributes
protected_data_SN = dummies(protected_data_SN)
protected_data_SN = reindex_cols(protected_data_SN)

X, y = load_data(protected_data_SN)
supnoi = evaluate_model(X, y)
save_results('supnoi', supnoi)

# %% Suppression and generalization
protected_data_SG.isnull().sum()

# deal with categorical attributes
protected_data_SG = dummies(protected_data_SG)
protected_data_SG = reindex_cols(protected_data_SG)

X, y = load_data(protected_data_SG)
supgen = evaluate_model(X, y)
save_results('supgen', supgen)

# %% Generalization and noise
protected_data_GN.isnull().sum()
protected_data_GN = dealNaN(protected_data_GN)
protected_data_GN.isnull().sum()

# deal with categorical attributes
protected_data_GN = dummies(protected_data_GN)
protected_data_GN = reindex_cols(protected_data_GN)

X, y = load_data(protected_data_GN)
gennoi = evaluate_model(X, y)
save_results('gennoi', gennoi)

# %% Suppression, generalization and noise
protected_data_SGN.isnull().sum()

# deal with categorical attributes
protected_data_SGN = dummies(protected_data_SGN)
protected_data_SGN = reindex_cols(protected_data_SGN)

X, y = load_data(protected_data_SGN)
supgennoi = evaluate_model(X, y)
save_results('supgennoi', supgennoi)
