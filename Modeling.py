import pandas as pd
from collections import Counter
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from xgboost import XGBClassifier

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
def evaluate_model(X, y, res):
    # split data 70/30
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3)

    seed = 42
    rfc = RandomForestClassifier(random_state=seed)
    bc = BaggingClassifier(random_state=seed)
    xgb = XGBClassifier(random_state=seed)
    svm = SVC(probability=True)

    # set parameters
    param_grid = {
        'n_estimators': [50, 100, 200, 500, 600],
        'max_depth': [4, 6, 8, 10]
    }
    param_grid_bc = {
        'n_estimators': [50, 100, 200, 500, 600]
    }
    param_grid_xgb = {
        'n_estimators': [50, 100, 200, 500, 600],
        'max_depth': [4, 6, 8, 10],
        'learning_rate': [0.1, 0.01, 0.001]
    }
    param_grid_svm = {'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                      'C': [1, 10, 100]}

    # define metric functions
    scoring = ['accuracy', 'balanced_accuracy', 'f1_weighted', 'roc_auc_ovr_weighted', 'roc_auc_ovo_weighted']

    # create the parameter grid
    gs_rf = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5, scoring=scoring, refit='balanced_accuracy',
                         return_train_score=True)
    gs_bc = GridSearchCV(estimator=bc, param_grid=param_grid_bc, cv=5, scoring=scoring, refit='balanced_accuracy',
                         return_train_score=True)
    gs_xgb = GridSearchCV(estimator=xgb, param_grid=param_grid_xgb, cv=5, scoring=scoring, refit='balanced_accuracy',
                          return_train_score=True)
    gs_svm = GridSearchCV(estimator=svm, param_grid=param_grid_svm, cv=5, scoring=scoring, refit='balanced_accuracy',
                          return_train_score=True)

    # List of pipelines for ease of iteration
    grids = [gs_rf, gs_bc, gs_xgb, gs_svm]

    # Dictionary of pipelines and classifier types for ease of reference
    grid_dict = {0: 'Random Forest', 1: 'Bagging', 2: 'Boosting', 3: 'SVM'}

    # Fit the grid search objects
    print('Performing model optimizations...')

    for idx, gs in enumerate(grids):
        print('\nEstimator: %s' % grid_dict[idx])
        # Performing cross validation to tune parameters for best model fit
        gs.fit(X_train, y_train)
        # Best params
        print('Best params: %s' % gs.best_params_)
        # Best training data accuracy
        print('Best training accuracy: %.3f' % gs.best_score_)
        # Store results from grid search
        # res['cv_results_' + str(grid_dict[idx])] = pd.DataFrame.from_dict(scores.cv_results_)
        res['cv_results_' + str(grid_dict[idx])] = gs.cv_results_
        # Predict on test data with best params
        y_pred = gs.predict(X_test)
        print(confusion_matrix(y_test, y_pred))
        # Test data accuracy of model with best params
        print('Test set accuracy score for best params: %.3f ' % balanced_accuracy_score(y_test, y_pred))

    return gs, res


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


# %% Baseline
X, y = load_data(initial_data)
# store results from all grids
baseline = {}
grid, baseline = evaluate_model(X, y, baseline)

# %% Suppression
protected_data_S.isnull().sum()

# deal with categorical attributes
protected_data_S = dummies(protected_data_S)
protected_data_S = reindex_cols(protected_data_S)

X, y = load_data(protected_data_S)
# store results from all grids
sup = {}
grid_sup, sup = evaluate_model(X, y, sup)

# %% Generalization
protected_data_G.isnull().sum()
protected_data_G = dealNaN(protected_data_G)
protected_data_G.isnull().sum()

# deal with categorical attributes
protected_data_G = dummies(protected_data_G)
protected_data_G = reindex_cols(protected_data_G)

X, y = load_data(protected_data_G)
# store results from all grids
gen = {}
grid_gen, gen = evaluate_model(X, y, gen)

# %% Suppression and noise
protected_data_SN.isnull().sum()

# deal with categorical attributes
protected_data_SN = dummies(protected_data_SN)
protected_data_SN = reindex_cols(protected_data_SN)

X, y = load_data(protected_data_SN)
# store results from all grids
supnoi = {}
grid_supnoi, supnoi = evaluate_model(X, y, supnoi)

# %% Suppression and generalization
protected_data_SG.isnull().sum()

# deal with categorical attributes
protected_data_SG = dummies(protected_data_SG)
protected_data_SG = reindex_cols(protected_data_SG)

X, y = load_data(protected_data_SG)
# store results from all grids
supgen = {}
grid_supgen, supgen = evaluate_model(X, y, supgen)

# %% Generalization and noise
protected_data_GN.isnull().sum()
protected_data_GN = dealNaN(protected_data_GN)
protected_data_GN.isnull().sum()

# deal with categorical attributes
protected_data_GN = dummies(protected_data_GN)
protected_data_GN = reindex_cols(protected_data_GN)

X, y = load_data(protected_data_GN)
# store results from all grids
gennoi = {}
grid_gennoi, gennoi = evaluate_model(X, y, gennoi)

# %% Suppression, generalization and noise
protected_data_SGN.isnull().sum()

# deal with categorical attributes
protected_data_SGN = dummies(protected_data_SGN)
protected_data_SGN = reindex_cols(protected_data_SGN)

X, y = load_data(protected_data_SGN)
# store results from all grids
supgennoi = {}
grid_supgennoi, supgennoi = evaluate_model(X, y, supgennoi)

