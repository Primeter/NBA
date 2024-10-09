import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# %% Load data
initial_data = pd.read_csv('NBA_Data/PlayersStats_SevereInjuries_clean.csv')

initial_data.info()


# %% Convert attributes type
def convert_type(df, type, cols):
    for column in cols:
        df[column] = df[column].astype(type)

    return df


# %% Convert float64 to integer64 (pandas read into float by default)
columns = ['height', 'weight', 'born', 'year_start', 'year_end', 'Year', 'Age', 'NBA_Years', 'SevereInjury']

initial_data = convert_type(initial_data, 'Int64', columns)

# %% Birth state
country = initial_data.groupby(['birth_state']).size().reset_index(name='Count')
((country['Count']) == 1).sum()

# %% College
initial_data.rename(columns={'collage': 'college'}, inplace=True)
college = initial_data.groupby(['college']).size().reset_index(name='Count')
((college['Count']) == 1).sum()

# %% Plot height
height = initial_data.groupby(['height']).size().reset_index(name='Count')
plt.hist(initial_data['height'], color='gray', edgecolor='white', bins=20)
plt.ylabel('Number of players')
plt.xlabel('Height')
plt.show()

# %% Plot weight
weight = initial_data.groupby(['weight']).size().reset_index(name='Count')
plt.hist(initial_data['weight'], color='gray', edgecolor='white', bins=20)
plt.ylabel('Number of players')
plt.xlabel('Weight')
plt.show()

# %% Plot PER
per = initial_data.groupby(['PER']).size().reset_index(name='Count')
plt.hist(initial_data['PER'], color='gray', edgecolor='white', bins=20)
plt.ylabel('Number of players')
plt.xlabel('PER')
plt.show()

# %% New dataframe to work on transformations
protected_data = initial_data.copy()

######################
#       MASKING      #
######################

# %% Set of quasi-identifiers
qi_k = ['height', 'weight', 'college', 'born', 'birth_city', 'birth_state', 'year_start', 'year_end',
        'Year', 'Age', 'NBA_Years']

qi_ep = ['PER', 'G', 'GS', 'MP', 'ORB%', 'DRB%', 'TRB%', 'ORB', 'DRB', 'TRB', 'PTS']

# %% Privacy risk with all QI less 'PER' because it is float
protected_data['fk'] = protected_data.groupby(qi_k, dropna=False)['college'].transform('size')

initial_fk = protected_data.groupby('fk').size().reset_index(name='Count')

# %% SUPPRESSION
# Suppress attribute
protected_data_S = protected_data.copy()
protected_data_S['college'] = '*'
protected_data_S['birth_city'] = '*'
protected_data_S['year_start'] = '*'
protected_data_S['year_end'] = '*'
protected_data_S['birth_state'] = '*'

protected_data_S['fk'] = protected_data_S.groupby(qi_k, dropna=False)['college'].transform('size')

fk = protected_data_S.groupby('fk').size().reset_index(name='Count')

del protected_data_S['fk']

protected_data_S.to_csv('NBA_Data/Protected_NBA_Sup.csv', index=False)

# GENERALIZATION
# %% Create extra dataframes
protected_data_G = protected_data.copy()
# Columns that will be transformed
cols = ['Age', 'born', 'height', 'weight', 'NBA_Years', 'Year']


# %% Set the parameters to generalization
hg = [3, 5, 10, 12]
wg = [3, 5, 10, 12]
born = [3, 5, 6]
age = [3, 5, 6]
nbay = [3, 5, 6]
years = [3, 5, 6]


def reset_col(column, df):
    for col in cols[cols.index(column):]:
        df[col] = protected_data[col]

    return df


def generalization(experimental_data):
    a = b = h = w = n = y = 0
    save_a = save_b = save_h = save_w = save_n = save_y = 0
    max = initial_fk['Count'][0]
    for a in age:
        experimental_data['Age'] = pd.cut(protected_data['Age'], bins=list(range(-1, 2025, a)))
        experimental_data = reset_col('born', experimental_data)
        b = h = w = n = y = 0
        experimental_data['fk'] = experimental_data.groupby(qi_k, dropna=False)['college'].transform('size')
        fk_1 = ((experimental_data['fk']) == 1).sum()
        if fk_1 < max:
            max = fk_1
            save_a = a; save_b = b; save_h = h; save_w = w; save_n = n; save_y = y

        for b in born:
            experimental_data['born'] = pd.cut(protected_data['born'], bins=list(range(-1, 2025, b)))
            experimental_data = reset_col('height', experimental_data)
            h = w = n = y = 0
            experimental_data['fk'] = experimental_data.groupby(qi_k, dropna=False)['college'].transform('size')
            fk_1 = ((experimental_data['fk']) == 1).sum()
            if fk_1 < max:
                max = fk_1
                save_a = a; save_b = b; save_h = h; save_w = w; save_n = n; save_y = y

            for h in hg:
                experimental_data['height'] = pd.cut(protected_data['height'], bins=list(range(-1, 2025, h)))
                experimental_data = reset_col('weight', experimental_data)
                w = n = y = 0
                experimental_data['fk'] = experimental_data.groupby(qi_k, dropna=False)['college'].transform('size')
                fk_1 = ((experimental_data['fk']) == 1).sum()
                if fk_1 < max:
                    max = fk_1
                    save_a = a; save_b = b; save_h = h; save_w = w; save_n = n; save_y = y

                for w in wg:
                    experimental_data['weight'] = pd.cut(protected_data['weight'], bins=list(range(-1, 2025, w)))
                    experimental_data = reset_col('NBA_Years', experimental_data)
                    n = y = 0
                    experimental_data['fk'] = experimental_data.groupby(qi_k, dropna=False)['college'].transform('size')
                    fk_1 = ((experimental_data['fk']) == 1).sum()
                    if fk_1 < max:
                        max = fk_1
                        save_a = a; save_b = b; save_h = h; save_w = w; save_n = n; save_y = y

                    for n in nbay:
                        experimental_data['NBA_Years'] = pd.cut(protected_data['NBA_Years'],
                                                                bins=list(range(-1, 2025, n)))
                        experimental_data = reset_col('Year', experimental_data)
                        y = 0
                        experimental_data['fk'] = experimental_data.groupby(qi_k, dropna=False)['college'].transform(
                            'size')
                        fk_1 = ((experimental_data['fk']) == 1).sum()
                        if fk_1 < max:
                            max = fk_1
                            save_a = a; save_b = b; save_h = h; save_w = w; save_n = n; save_y = y

                        for y in years:
                            experimental_data['Year'] = pd.cut(protected_data['Year'], bins=list(range(-1, 2025, y)))
                            experimental_data['fk'] = experimental_data.groupby(qi_k, dropna=False)[
                                'college'].transform('size')
                            fk_1 = ((experimental_data['fk']) == 1).sum()
                            if fk_1 < max:
                                max = fk_1
                                save_a = a; save_b = b; save_h = h; save_w = w; save_n = n; save_y = y

    experimental_data['Age'] = pd.cut(protected_data['Age'], bins=list(range(-1, 2025, save_a)))
    experimental_data['born'] = pd.cut(protected_data['born'], bins=list(range(-1, 2025, save_b)))
    experimental_data['height'] = pd.cut(protected_data['height'], bins=list(range(-1, 2025, save_h)))
    experimental_data['weight'] = pd.cut(protected_data['weight'], bins=list(range(-1, 2025, save_w)))
    experimental_data['NBA_Years'] = pd.cut(protected_data['NBA_Years'], bins=list(range(-1, 2025, save_n)))
    experimental_data['Year'] = pd.cut(protected_data['Year'], bins=list(range(-1, 2025, save_y)))
    experimental_data['fk'] = experimental_data.groupby(qi_k, dropna=False)['college'].transform('size')
    experimental_data = convert_type(experimental_data, str, cols)

    return experimental_data


# %% Initiate generalization
protected_data_G = generalization(protected_data_G)
((protected_data_G['fk']) == 1).sum()

del protected_data_G['fk']

protected_data_G.to_csv('NBA_Data/Protected_NBA_Gen.csv', index=False)

# %% SUPPRESSION & NOISE (SN)
protected_data_SN = protected_data_S.copy()

stats_eq = pd.DataFrame()

# Set parameters for Laplace function implementation
sensitivity = 1


def Laplace(ep):
    beta = sensitivity / ep
    # Gets random laplacian noise for all values
    laplacian_noise = np.random.laplace(0, beta, 1)

    return laplacian_noise


def apply_epsilon(col, df, eq):
    ints = ['G', 'GS', 'MP', 'ORB', 'DRB', 'TRB', 'PTS']
    ep = 0.5
    for group_name, df_group in eq:
        for row_index, row in df_group.iterrows():
            df[col][row_index] = df[col][row_index] + Laplace(ep)
    if col in ints:
        df[col] = df[col].apply(lambda x: format(x, '.0f'))
    else:
        df[col] = df[col].apply(lambda x: format(x, '.1f'))
    df[col] = df[col].astype('float64')
    return df


# how many equivalence classes
eq = protected_data_SN.groupby(qi_k)

for col in qi_ep:
    apply_epsilon(col, protected_data_SN, eq)

protected_data_SN[qi_ep].min()
protected_data_SN[qi_ep[1:]] = np.where(protected_data_SN[qi_ep[1:]] < 0, 0, protected_data_SN[qi_ep[1:]])
protected_data_SN['G'] = np.where(protected_data_SN['G'] == -0.0, 1, protected_data_SN['G'])
protected_data_SN['MP'] = np.where(protected_data_SN['MP'] == 0.0, 1, protected_data_SN['MP'])
protected_data_SN[qi_ep[1:]] = np.where(protected_data_SN[qi_ep[1:]] == -0.0, 0, protected_data_SN[qi_ep[1:]])

protected_data_SN.to_csv('NBA_Data/Protected_NBA_SupNoi.csv', index=False)

# %% # SUPPRESSION & GENERALIZATION (SG)
protected_data_SG = protected_data_S.copy()

protected_data_SG = generalization(protected_data_SG)
((protected_data_SG['fk']) == 1).sum()

del protected_data_SG['fk']

protected_data_SG.to_csv('NBA_Data/Protected_NBA_SupGen.csv', index=False)

# %% GENERALIZATION & NOISE (GN)
protected_data_GN = protected_data_G.copy()

# how many equivalence classes
eq = protected_data_GN.groupby(qi_k)

for col in qi_ep:
    apply_epsilon(col, protected_data_GN, eq)

protected_data_GN[qi_ep].min()
protected_data_GN[qi_ep[1:]] = np.where(protected_data_GN[qi_ep[1:]] < 0, 0, protected_data_GN[qi_ep[1:]])
protected_data_GN['G'] = np.where(protected_data_GN['G'] == 0.0, 1, protected_data_GN['G'])
protected_data_GN['MP'] = np.where(protected_data_GN['MP'] == 0.0, 1, protected_data_GN['MP'])
protected_data_GN[qi_ep[1:]] = np.where(protected_data_GN[qi_ep[1:]] == -0.0, 0, protected_data_GN[qi_ep[1:]])

protected_data_GN.to_csv('NBA_Data/Protected_NBA_GenNoi.csv', index=False)

# %% SUPPRESSION & GENERALIZATION & NOISE (SGN)
protected_data_SGN = protected_data_SG.copy()

# how many equivalence classes
eq = protected_data_SGN.groupby(qi_k)

for col in qi_ep:
    apply_epsilon(col, protected_data_SGN, eq)

protected_data_SGN[qi_ep].min()
protected_data_SGN[qi_ep[1:]] = np.where(protected_data_SGN[qi_ep[1:]] < 0, 0, protected_data_SGN[qi_ep[1:]])
protected_data_SGN['G'] = np.where(protected_data_SGN['G'] == 0.0, 1, protected_data_SGN['G'])
protected_data_SGN['MP'] = np.where(protected_data_SGN['MP'] == 0.0, 1, protected_data_SGN['MP'])
protected_data_SGN[qi_ep[1:]] = np.where(protected_data_SGN[qi_ep[1:]] == -0.0, 0, protected_data_SGN[qi_ep[1:]])

protected_data_SGN.to_csv('NBA_Data/Protected_NBA_SupGenNoi.csv', index=False)
