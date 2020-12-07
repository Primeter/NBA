import pandas as pd
import matplotlib.pyplot as plt
import recordlinkage
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


# %% Re-identification with record linkage
def record_linkage(df_protect, df_init, block_left, block_right, gen_col):
    indexer = recordlinkage.Index()
    if block_left is None:
        indexer.full()
    if block_left is not None:
        indexer.block(left_on=block_left, right_on=block_right)
    candidates = indexer.index(df_protect, df_init)
    # print(len(candidates))
    compare = recordlinkage.Compare()
    if all(df_protect[gen_col].dtypes == object):
        compare.string('height', 'height', threshold=0.9, label='height')
        compare.string('weight', 'weight', threshold=0.9, label='weight')
        # compare.string('born', 'born', threshold=0.9, label='born')
        compare.string('Year', 'Year', threshold=0.9, label='Year')
        compare.string('Age', 'Age', threshold=0.9, label='Age')
        compare.string('NBA_Years', 'NBA_Years', threshold=0.9, label='NBA_Years')
        if block_left == 'birth_city':
            compare.string('born', 'born', threshold=0.9, label='born')

    if not all(df_protect[gen_col].dtypes == object):
        compare.numeric('height', 'height', label='height')
        compare.numeric('weight', 'weight', label='weight')
        compare.numeric('Year', 'Year', label='Year')
        compare.numeric('Age', 'Age', label='Age')
        compare.numeric('NBA_Years', 'NBA_Years', label='NBA_Years')
        if block_left == 'born':
            compare.string('birth_city', 'birth_city', threshold=0.9, label='birth_city')

    if all(df_protect[['year_start', 'year_end']].dtypes == float):
        compare.numeric('year_start', 'year_start', label='year_start')
        compare.numeric('year_end', 'year_end', label='year_end')

    if not all(df_protect[['year_start', 'year_end']].dtypes == float):
        compare.string('year_start', 'year_start', threshold=0.9, label='year_start')
        compare.string('year_end', 'year_end', threshold=0.9, label='year_end')

    compare.string('college', 'college', threshold=0.9, label='college')
    compare.string('birth_state', 'birth_state', threshold=0.9, label='birth_state')
    compare.numeric('PER', 'PER', label='PER')
    compare.numeric('G', 'G', label='G')
    compare.numeric('GS', 'GS', label='GS')
    compare.numeric('MP', 'MP', label='MP')
    compare.numeric('ORB%', 'ORB%', label='ORB%')
    compare.numeric('DRB%', 'DRB%', label='DRB%')
    compare.numeric('TRB%', 'TRB%', label='TRB%')
    compare.numeric('ORB', 'ORB', label='ORB')
    compare.numeric('DRB', 'DRB', label='DRB')
    compare.numeric('TRB', 'TRB', label='TRB')
    compare.numeric('PTS', 'PTS', label='PTS')

    return compare.compute(candidates, df_protect, df_init)


# %% Convert to string the numerical data which are in range categories - to record linkage
initial_data_S = initial_data.copy()
initial_data_G = initial_data.copy()
initial_data_SG = initial_data.copy()

sup_cols = ['college', 'birth_city', 'year_start', 'year_end', 'birth_state']
gen_cols = list(set(qi_k) - set(sup_cols))

initial_data_S = convert_type(initial_data_S, str, sup_cols)
initial_data_G = convert_type(initial_data_G, str, gen_cols)
initial_data_SG = convert_type(initial_data_SG, str, qi_k)

initial_data_S = convert_type(initial_data_S, np.int32, gen_cols)
protected_data_S = convert_type(protected_data_S, np.int32, gen_cols)
protected_data_SN = convert_type(protected_data_SN, np.int32, gen_cols)

# %% Initiate record linkage to suppression
linkage_S = record_linkage(initial_data_S, protected_data_S, 'born', 'born', gen_cols)

# Get the linkage score
potential_matches_S = linkage_S[linkage_S.sum(axis=1) > 1].reset_index()
potential_matches_S['Score'] = potential_matches_S.loc[:, 'height':'PTS'].sum(axis=1)
potential_matches_S = potential_matches_S[potential_matches_S['Score'] == 10]


# %% Initiate record linkage to generalization
initial_data_G['year_start'] = initial_data_G['year_start'].fillna(-1)
initial_data_G['year_end'] = initial_data_G['year_end'].fillna(-1)
initial_data_G = convert_type(initial_data_G, np.int32, ['year_start', 'year_end'])
initial_data_G['year_start'] = np.where(initial_data_G['year_start'] == -1, np.nan, initial_data_G['year_start'])
initial_data_G['year_end'] = np.where(initial_data_G['year_end'] == -1, np.nan, initial_data_G['year_start'])

protected_data_G['year_start'] = protected_data_G['year_start'].fillna(-1)
protected_data_G['year_end'] = protected_data_G['year_end'].fillna(-1)
protected_data_G = convert_type(protected_data_G, np.int32, ['year_start', 'year_end'])
protected_data_G['year_start'] = np.where(protected_data_G['year_start'] == -1, np.nan, protected_data_G['year_start'])
protected_data_G['year_end'] = np.where(protected_data_G['year_end'] == -1, np.nan, protected_data_G['year_start'])

linkage_G = record_linkage(initial_data_G, protected_data_G, None, None, gen_cols)

# Get the linkage score
potential_matches_G = linkage_G[linkage_G.sum(axis=1) > 1].reset_index()
potential_matches_G['Score'] = potential_matches_G.loc[:, 'height':'PTS'].sum(axis=1)
potential_matches_G = potential_matches_G[potential_matches_G['Score'] >= 11]
potential_matches_G[potential_matches_G['Score'] == 15].count()


# %% Initiate record linkage to suppression and noise
linkage_SN = record_linkage(initial_data_S, protected_data_SN, 'born', 'born', gen_cols)

# Get the linkage score
potential_matches_SN = linkage_SN[linkage_SN.sum(axis=1) > 1].reset_index()
potential_matches_SN['Score'] = potential_matches_SN.loc[:, 'height':'PTS'].sum(axis=1)
potential_matches_SN = potential_matches_SN[potential_matches_SN['Age'] == 1]
potential_matches_SN = potential_matches_SN[(potential_matches_SN['height'] >= 0.8) &
                                                  (potential_matches_SN['weight'] >= 0.8)]
potential_matches_SN = potential_matches_SN[potential_matches_SN['Year'] == 1]
potential_matches_SN = potential_matches_SN[potential_matches_SN['NBA_Years'] == 1]
potential_matches_SN[potential_matches_SN['Score'] > 6].count()

# %% Initiate record linkage to suppression and generalization
linkage_SG = record_linkage(initial_data_SG, protected_data_SG, None, None, gen_cols)

# Get the linkage score
potential_matches_SG = linkage_SG[linkage_SG.sum(axis=1) > 1].reset_index()
potential_matches_SG['Score'] = potential_matches_SG.loc[:, 'height':'PTS'].sum(axis=1)
potential_matches_SG[potential_matches_SG['Score'] >= 11].count()
potential_matches_SG = potential_matches_SG[potential_matches_SG['Score'] >= 11]
dups = potential_matches_SG[potential_matches_SG.level_1.duplicated()]

# %% Initiate record linkage to generalization and noise
protected_data_GN['year_start'] = protected_data_GN['year_start'].fillna(-1)
protected_data_GN['year_end'] = protected_data_GN['year_end'].fillna(-1)
protected_data_GN = convert_type(protected_data_GN, np.int32, ['year_start', 'year_end'])
protected_data_GN['year_start'] = np.where(protected_data_GN['year_start'] == -1, np.nan, protected_data_GN['year_start'])
protected_data_GN['year_end'] = np.where(protected_data_GN['year_end'] == -1, np.nan, protected_data_GN['year_start'])

linkage_GN = record_linkage(protected_data_GN, initial_data_G, None, None, gen_cols)

# Get the linkage score
potential_matches_GN = linkage_GN[linkage_GN.sum(axis=1) > 1].reset_index()
potential_matches_GN['Score'] = potential_matches_GN.loc[:, 'height':'PTS'].sum(axis=1)
potential_matches_GN[potential_matches_GN['Score'] >= 12].count()
potential_matches_GN = potential_matches_GN[potential_matches_GN['Score'] >= 3.9]
potential_matches_GN = potential_matches_GN[
    (potential_matches_GN.loc[:, 'year_start':'birth_state'].sum(axis=1)
     >= 1) | (potential_matches_GN.loc[:, 'PER':'PTS'].sum(axis=1) >= 9)]

inds = potential_matches_GN.groupby(['level_0'])['Score'].transform(max) >= 12

ind = potential_matches_GN[inds].groupby(['level_0']).Score.idxmax()

df_inds = potential_matches_GN[inds]

idx = df_inds[df_inds.Score < 12]
idx2 = idx.index.get_level_values(-1)  # get single index
idx1 = potential_matches_GN.index.get_level_values(-1)

potential_matches_GN = potential_matches_GN[~idx1.isin(idx2)]

potential_matches_GN['sum'] = potential_matches_GN.iloc[:, 8:11].sum(axis=1)
potential_matches_GN = potential_matches_GN.loc[
    potential_matches_GN.groupby('level_0')['sum'].transform(max) == potential_matches_GN['sum']]

dup = potential_matches_GN.groupby('level_0').size().reset_index(name='Count')
dup[dup.Count == 1].count()
dup[dup.Count == 2].count()
dups = potential_matches_GN[potential_matches_GN.level_1.duplicated()]

# %% Initiate record linkage to suppression, generalization and noise
linkage_SGN = record_linkage(protected_data_SGN, initial_data_SG, None, None, gen_cols)

# Get the linkage score
potential_matches_SGN = linkage_SGN[linkage_SGN.sum(axis=1) > 1].reset_index()
potential_matches_SGN['Score'] = potential_matches_SGN.loc[:, 'height':'PTS'].sum(axis=1)
potential_matches_SGN['Score'].max()

potential_matches_SGN['max'] = potential_matches_SGN.loc[:, 'PER':'PTS'].max(axis=1)
potential_matches_SGN = potential_matches_SGN.loc[potential_matches_SGN.groupby('level_0')['max']
                                                            .transform(max) == potential_matches_SGN['max']]

dups = potential_matches_SGN[potential_matches_SGN.level_1.duplicated()]


