import base64
import pandas as pd
import hashlib
import matplotlib.pyplot as plt
import collections
from scipy.stats import entropy
import recordlinkage
import numpy as np

# %% Load data
initial_data = pd.read_csv('NBA_Data/PlayersStats_SevereInjuries_clean2.csv')

del initial_data['Unnamed: 0']

initial_data.info()


# %% Convert attributes type
def convert_type(df, type, cols):
    for column in cols:
        df[column] = df[column].astype(type)

    return df


# %% Append the statistics on each generalisation
def stats_add_datapoints(idx, col, df1, df2):
    return df1.append(pd.Series({'min': df2['fk'].min(), 'max': df2['fk'].max(), 'avg': df2['fk'].mean(),
                                 'fk=1': ((df2['fk']) == 1).sum(), name=idx))


# %% Convert float64 to integer64 (pandas read into float by default)
columns = ['height', 'weight', 'born', 'year_start', 'year_end', 'Year', 'Age', 'NBA_Years', 'SevereInjury']

initial_data = convert_type(initial_data, 'Int64', columns)

# %% How many players by season?
season_players = initial_data.groupby(['Year', 'Player']).size().reset_index(name='Count') \
    .groupby('Year').size()

# %% Birth country - too many uniques
country = initial_data.groupby(['Player', 'birth_state']).size().reset_index(name='Count')
countryn = country.groupby('birth_state').size()

# %% College
college = initial_data.groupby(['Player', 'collage']).size().reset_index(name='Count')
college = collage.groupby('collage').size().reset_index(name='Count')
college[collage['Count'] == 1].count()  # 69 unique universities

# %% Plot height
height = initial_data.groupby(['Player', 'height']).size().reset_index(name='Count')
plt.hist(height['height'], color='gray', edgecolor='white', bins=20)
plt.ylabel('Number of players')
plt.xlabel('Height')
plt.show()

# %% Plot weight
weight = initial_data.groupby(['Player', 'weight']).size().reset_index(name='Count')
plt.hist(weight['weight'], color='gray', edgecolor='white', bins=20)
plt.ylabel('Number of players')
plt.xlabel('Weight')
plt.show()

# %% Plot PER
per = initial_data.groupby(['Player', 'PER']).size().reset_index(name='Count')
plt.hist(per['PER'], color='gray', edgecolor='white', bins=20)
plt.ylabel('Number of players')
plt.xlabel('PER')
plt.show()

######################
#     PSEUDONYMS     #
######################
# %% New dataframe to work on transformations
protected_data = initial_data.copy()

# %% NOT USED!!
# Hash function on Player
key = b'catsareliquid'

protected_data['Player'] = protected_data['Player'] \
    .apply(lambda x: base64.b64encode(hashlib.blake2b(x.encode('UTF-8'), key=key, digest_size=32).digest()))

# Hash function on Team
protected_data['Tm'] = protected_data['Tm'] \
    .apply(lambda x: base64.b64encode(hashlib.blake2b(x.encode('UTF-8'), key=key, digest_size=32).digest()))

# Reverse
name = initial_data['Player'][0]
reverse = base64.b64encode(hashlib.blake2b(name.encode('UTF-8'), key=key, digest_size=32).digest())
reverse_data = protected_data[protected_data['Player'] == reverse]

######################
#       MASKING      #
######################

# %% Remove explicit identifiers
del protected_data['Player']
del protected_data['Tm']

# %% Quasi-identifiers set
qi_k = ['height', 'weight', 'collage', 'born', 'birth_city', 'birth_state', 'year_start', 'year_end',
        'Year', 'Age', 'NBA_Years']

qi_ep = ['PER', 'G', 'GS', 'MP', 'ORB%', 'DRB%', 'TRB%', 'ORB', 'DRB', 'TRB', 'PTS']

# %% Privacy risk with all QI less 'PER' because it is float
protected_data['fk'] = protected_data.groupby(qi_k, dropna=False)['collage'].transform('size')

initial_fk = protected_data.groupby('fk').size().reset_index(name='Count')

# %% SUPPRESSION
# Suppress attribute
protected_data_S = protected_data.copy()
protected_data_S['collage'] = '*'
protected_data_S['birth_city'] = '*'
protected_data_S['year_start'] = '*'
protected_data_S['year_end'] = '*'
protected_data_S['birth_state'] = '*'
# protected_data_S['Player'] = '*'
# protected_dataS['Tm'] = '*'

protected_data_S['fk'] = protected_data_S.groupby(qi_k, dropna=False)['collage'].transform('size')

fk = protected_data_S.groupby('fk').size().reset_index(name='Count')

df_columns = protected_data.columns.values.tolist()
tgt = ['SevereInjury']
cols_to_remove = list(set(df_columns) - set(qi_k) - set(qi_ep) - set(tgt))

protected_data_S_clean = protected_data_S.copy()
protected_data_S_clean.drop(protected_data_S_clean[cols_to_remove], axis=1, inplace=True)


# GENERALIZATION
# %% Create extra dataframes
protected_data_G = protected_data.copy()
# Columns that will be transformed
cols = ['Age', 'born', 'height', 'weight', 'NBA_Years', 'Year']


def df_reset(copy_df):
    # Dataframe copy to experiment several bins
    df_exp = copy_df.copy()
    # Create new dataframe to register the min, max, avg of fk, counts of fk=1 and the entropy of each transformation
    df_stats = pd.DataFrame()
    for col in cols:
        df_stats = stats_add_datapoints(col, col, df_stats, copy_df)
    return df_stats, df_exp


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


def generalization(stats, experimental_data):
    a = b = h = w = n = y = 0
    for a in age:
        experimental_data['Age'] = pd.cut(protected_data['Age'], bins=list(range(-1, 2025, a)))
        experimental_data = reset_col('born', experimental_data)
        b = h = w = n = y = 0
        experimental_data['fk'] = experimental_data.groupby(qi_k, dropna=False)['collage'].transform('size')
        idx = 'age' + str(a) + '_born' + str(b) + '_h' + str(h) + '_w' + str(w) + '_nbay' + str(n) \
              + '_years' + str(y)
        stats = stats_add_datapoints(idx, 'Age', stats, experimental_data)

        for b in born:
            experimental_data['born'] = pd.cut(protected_data['born'], bins=list(range(-1, 2025, b)))
            experimental_data = reset_col('height', experimental_data)
            h = w = n = y = t = 0
            experimental_data['fk'] = experimental_data.groupby(qi_k, dropna=False)['collage'].transform('size')
            idx = 'age' + str(a) + '_born' + str(b) + '_h' + str(h) + '_w' + str(w) + '_nbay' + str(n) \
                  + '_years' + str(y)
            stats = stats_add_datapoints(idx, 'born', stats, experimental_data)

            for h in hg:
                experimental_data['height'] = pd.cut(protected_data['height'], bins=list(range(-1, 2025, h)))
                experimental_data = reset_col('weight', experimental_data)
                w = n = y = t = 0
                experimental_data['fk'] = experimental_data.groupby(qi_k, dropna=False)['collage'].transform('size')
                idx = 'age' + str(a) + '_born' + str(b) + '_h' + str(h) + '_w' + str(w) + '_nbay' + str(n) \
                      + '_years' + str(y)
                stats = stats_add_datapoints(idx, 'height', stats, experimental_data)

                for w in wg:
                    experimental_data['weight'] = pd.cut(protected_data['weight'], bins=list(range(-1, 2025, w)))
                    experimental_data = reset_col('NBA_Years', experimental_data)
                    n = y = t = 0
                    experimental_data['fk'] = experimental_data.groupby(qi_k, dropna=False)['collage'].transform('size')
                    idx = 'age' + str(a) + '_born' + str(b) + '_h' + str(h) + '_w' + str(w) + '_nbay' + str(n) \
                          + '_years' + str(y)
                    stats = stats_add_datapoints(idx, 'weight', stats, experimental_data)

                    for n in nbay:
                        experimental_data['NBA_Years'] = pd.cut(protected_data['NBA_Years'],
                                                                bins=list(range(-1, 2025, n)))
                        experimental_data = reset_col('Year', experimental_data)
                        y = t = 0
                        experimental_data['fk'] = experimental_data.groupby(qi_k, dropna=False)['collage'].transform(
                            'size')
                        idx = 'age' + str(a) + '_born' + str(b) + '_h' + str(h) + '_w' + str(w) + '_nbay' + str(n) \
                              + '_years' + str(y)
                        stats = stats_add_datapoints(idx, 'NBA_Years', stats, experimental_data)

                        for y in years:
                            experimental_data['Year'] = pd.cut(protected_data['Year'], bins=list(range(-1, 2025, y)))
                            t = 0
                            experimental_data['fk'] = experimental_data.groupby(qi_k, dropna=False)[
                                'collage'].transform(
                                'size')
                            idx = 'age' + str(a) + '_born' + str(b) + '_h' + str(h) + '_w' + str(w) + '_nbay' + str(n) \
                                  + '_years' + str(y)
                            stats = stats_add_datapoints(idx, 'Year', stats, experimental_data)

    return stats, experimental_data


# %% Initiate generalization
stats_G, experimental_data_G = df_reset(protected_data)

stats_G, experimental_data_G = generalization(stats_G, experimental_data_G)

# %% Check the fk
stats_G['fk=1'].min()

sorted(stats_G['fk=1'].unique())

fk_G = stats_G.groupby('fk=1').size().reset_index(name='Count')

stats_G[stats_G['fk=1'] == 725]

# %% Save the best parameters
a = 6
b = 6
h = 10
w = 10
n = 6
y = 6
protected_data_G['Age'] = pd.cut(protected_data['Age'], bins=list(range(-1, 2025, a)))
protected_data_G['born'] = pd.cut(protected_data['born'], bins=list(range(-1, 2025, b)))
protected_data_G['height'] = pd.cut(protected_data['height'], bins=list(range(-1, 2025, h)))
protected_data_G['weight'] = pd.cut(protected_data['weight'], bins=list(range(-1, 2025, w)))
protected_data_G['NBA_Years'] = pd.cut(protected_data['NBA_Years'], bins=list(range(-1, 2025, n)))
protected_data_G['Year'] = pd.cut(protected_data['Year'], bins=list(range(-1, 2025, y)))
protected_data_G['fk'] = protected_data_G.groupby(qi_k, dropna=False)['collage'].transform('size')
protected_data_G = convert_type(protected_data_G, str, cols)

fk = protected_data_G.groupby('fk').size().reset_index(name='Count')

protected_data_G_clean = protected_data_G.copy()
protected_data_G_clean.drop(protected_data_G_clean[cols_to_remove], axis=1, inplace=True)


# %% SUPPRESSION & NOISE (SN)
experimental_data_SN = protected_data_S.copy()
protected_data_SN = protected_data_S.copy()

# Distribution of the quasi-identifiers for £-differential privacy
game = protected_data_SN.groupby('G').size()  # int, >1
game_started = protected_data_SN.groupby('GS').size()  # int, >0
min_played = protected_data_SN.groupby('MP').size()  # int, >1
orb_per = protected_data_SN.groupby('ORB%').size()  # float, >0
drb_per = protected_data_SN.groupby('DRB%').size()  # float, >0
trb_per = protected_data_SN.groupby('TRB%').size()  # float, >0
orb = protected_data_SN.groupby('ORB').size()  # int , >0
drb = protected_data_SN.groupby('DRB').size()  # int, >0
trb = protected_data_SN.groupby('TRB').size()  # int, >0
points = protected_data_SN.groupby('PTS').size()  # int, >0

stats_eq = pd.DataFrame()

# Set parameters for Laplace function implementation
sensitivity = 1

def Laplace(ep):
    beta = sensitivity / ep
    # Gets random laplacian noise for all values
    laplacian_noise = np.random.laplace(0, beta, 1)

    return laplacian_noise

def save_best_epsilon(col, df, eq):
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
    save_best_epsilon(col, protected_data_SN, eq)

protected_data_SN[qi_ep].min()
protected_data_SN[qi_ep[1:]] = np.where(protected_data_SN[qi_ep[1:]] < 0, 0, protected_data_SN[qi_ep[1:]])
protected_data_SN['G'] = np.where(protected_data_SN['G'] == -0.0, 1, protected_data_SN['G'])
protected_data_SN['MP'] = np.where(protected_data_SN['MP'] == 0.0, 1, protected_data_SN['MP'])
protected_data_SN[qi_ep[1:]] = np.where(protected_data_SN[qi_ep[1:]] == -0.0, 0, protected_data_SN[qi_ep[1:]])

per_noise = protected_data_SN.groupby(['PER']).size().reset_index(name='Count')
game_noise = protected_data_SN.groupby('G').size()
game_started_noise = protected_data_SN.groupby('GS').size()
min_played_noise = protected_data_SN.groupby('MP').size()
orb_per_noise = protected_data_SN.groupby('ORB%').size()
drb_per_noise = protected_data_SN.groupby('DRB%').size()
trb_per_noise = protected_data_SN.groupby('TRB%').size()
orb_noise = protected_data_SN.groupby('ORB').size()
drb_noise = protected_data_SN.groupby('DRB').size()
trb_noise = protected_data_SN.groupby('TRB').size()
points_noise = protected_data_SN.groupby('PTS').size()

fk = protected_data_SN.groupby('fk').size().reset_index(name='Count')

protected_data_SN_clean = protected_data_SN.copy()
protected_data_SN_clean.drop(protected_data_SN_clean[cols_to_remove], axis=1, inplace=True)


# %% # SUPPRESSION & GENERALIZATION (SG)
protected_data_SG = protected_data_S.copy()

stats_SG, experimental_data_SG = df_reset(protected_data_S)

stats_SG, experimental_data_SG = generalization(stats_SG, experimental_data_SG)

# %% Check the fk
fk_SG = stats_SG.groupby('fk=1').size().reset_index(name='Count')

# %% Save the best parameters
a = 6
b = 6
h = 10
w = 10
n = 6
y = 6
protected_data_SG['Age'] = pd.cut(protected_data_S['Age'], bins=list(range(-1, 2025, a)))
protected_data_SG['born'] = pd.cut(protected_data_S['born'], bins=list(range(-1, 2025, b)))
protected_data_SG['height'] = pd.cut(protected_data_S['height'], bins=list(range(-1, 2025, h)))
protected_data_SG['weight'] = pd.cut(protected_data_S['weight'], bins=list(range(-1, 2025, w)))
protected_data_SG['NBA_Years'] = pd.cut(protected_data_S['NBA_Years'], bins=list(range(-1, 2025, n)))
protected_data_SG['Year'] = pd.cut(protected_data_S['Year'], bins=list(range(-1, 2025, y)))
#protected_data_SG['total_years'] = pd.cut(protected_data_S['total_years'], bins=list(range(-1, 2025, t)))
protected_data_SG['fk'] = protected_data_SG.groupby(qi_k, dropna=False)['collage'].transform('size')
protected_data_SG = convert_type(protected_data_SG, str, cols)

fk = protected_data_SG.groupby('fk').size().reset_index(name='Count')

protected_data_SG_clean = protected_data_SG.copy()
protected_data_SG_clean.drop(protected_data_SG_clean[cols_to_remove], axis=1, inplace=True)


# %% GENERALIZATION & NOISE (GN)
experimental_data_GN = protected_data_G.copy()
protected_data_GN = protected_data_G.copy()

# how many equivalence classes
eq = protected_data_GN.groupby(qi_k)

for col in qi_ep:
    save_best_epsilon(col, protected_data_GN, eq)

protected_data_GN[qi_ep].min()
protected_data_GN[qi_ep[1:]] = np.where(protected_data_GN[qi_ep[1:]] < 0, 0, protected_data_GN[qi_ep[1:]])
protected_data_GN['G'] = np.where(protected_data_GN['G'] == 0.0, 1, protected_data_GN['G'])
protected_data_GN['MP'] = np.where(protected_data_GN['MP'] == 0.0, 1, protected_data_GN['MP'])
protected_data_GN[qi_ep[1:]] = np.where(protected_data_GN[qi_ep[1:]] == -0.0, 0, protected_data_GN[qi_ep[1:]])

fk = protected_data_GN.groupby('fk').size().reset_index(name='Count')

protected_data_GN_clean = protected_data_GN.copy()
protected_data_GN_clean.drop(protected_data_GN_clean[cols_to_remove], axis=1, inplace=True)


# %% SUPPRESSION & GENERALIZATION & NOISE (SGN)
experimental_data_SGN = protected_data_SG.copy()
protected_data_SGN = protected_data_SG.copy()

# how many equivalence classes
eq = protected_data_SGN.groupby(qi_k)

for col in qi_ep:
    save_best_epsilon(col, protected_data_SGN, eq)

protected_data_SGN[qi_ep].min()
protected_data_SGN[qi_ep[1:]] = np.where(protected_data_SGN[qi_ep[1:]] < 0, 0, protected_data_SGN[qi_ep[1:]])
protected_data_SGN['G'] = np.where(protected_data_SGN['G'] == 0.0, 1, protected_data_SGN['G'])
protected_data_SGN['MP'] = np.where(protected_data_SGN['MP'] == 0.0, 1, protected_data_SGN['MP'])
protected_data_SGN[qi_ep[1:]] = np.where(protected_data_SGN[qi_ep[1:]] == -0.0, 0, protected_data_SGN[qi_ep[1:]])

fk = protected_data_SGN.groupby('fk').size().reset_index(name='Count')

rmv = ['Player', 'Tm']
cols_to_remove = list(set(cols_to_remove) - set(rmv))

protected_data_SGN_clean = protected_data_SGN.copy()
protected_data_SGN_clean.drop(protected_data_SGN_clean[cols_to_remove], axis=1, inplace=True)



# %% Re-identification with record linkage
def record_linkage(df_protect, df_init, block_left, block_right, thr, gen_col):
    indexer = recordlinkage.Index()
    if block_left is None:
        indexer.full()
    if block_left is not None:
        indexer.block(left_on=block_left, right_on=block_right)
    candidates = indexer.index(df_protect, df_init)
    # print(len(candidates))
    compare = recordlinkage.Compare()
    if all(df_protect[gen_col].dtypes == object):
        compare.string('height', 'height', threshold=thr, label='height')
        compare.string('weight', 'weight', threshold=thr, label='weight')
        # compare.string('born', 'born', threshold=thr, label='born')
        compare.string('Year', 'Year', threshold=thr, label='Year')
        compare.string('Age', 'Age', threshold=thr, label='Age')
        compare.string('NBA_Years', 'NBA_Years', threshold=thr, label='NBA_Years')
        compare.string('total_years', 'total_years', threshold=thr, label='total_years')
        if block_left == 'birth_city':
            compare.string('born', 'born', threshold=thr, label='born')

    if not all(df_protect[gen_col].dtypes == object):
        compare.numeric('height', 'height', label='height')
        compare.numeric('weight', 'weight', label='weight')
        compare.numeric('Year', 'Year', label='Year')
        compare.numeric('Age', 'Age', label='Age')
        compare.numeric('NBA_Years', 'NBA_Years', label='NBA_Years')
        compare.numeric('total_years', 'total_years', label='total_years')
        if block_left == 'born':
            compare.string('birth_city', 'birth_city', threshold=thr, label='birth_city')

    if all(df_protect[['year_start', 'year_end']].dtypes == float):
        compare.numeric('year_start', 'year_start', label='year_start')
        compare.numeric('year_end', 'year_end', label='year_end')

    if not all(df_protect[['year_start', 'year_end']].dtypes == float):
        compare.string('year_start', 'year_start', threshold=thr, label='year_start')
        compare.string('year_end', 'year_end', threshold=thr, label='year_end')

    compare.string('collage', 'collage', threshold=thr, label='collage')
    compare.string('birth_state', 'birth_state', threshold=thr, label='birth_state')
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
# The noisy columns doesn't need type transformation
initial_data_S = initial_data.copy()
initial_data_G = initial_data.copy()
initial_data_SG = initial_data.copy()

sup_cols = ['collage', 'birth_city', 'year_start', 'year_end', 'birth_state']
gen_cols = list(set(qi_k) - set(sup_cols))

initial_data_S = convert_type(initial_data_S, str, sup_cols)
initial_data_G = convert_type(initial_data_G, str, gen_cols)
initial_data_SG = convert_type(initial_data_SG, str, qi_k)

initial_data_S = convert_type(initial_data_S, np.int32, gen_cols)
protected_data_S = convert_type(protected_data_S, np.int32, gen_cols)
protected_data_SN = convert_type(protected_data_SN, np.int32, gen_cols)

# %% Initiate record linkage to suppression
threshold = 0.9

features_S09 = record_linkage(initial_data_S, protected_data_S, 'born', 'born', thr, gen_cols)


# Get the linkage score
potential_matches_S09 = [features_S09.sum(axis=1) > 1].reset_index()
potential_matches_S009['Score'] = potential_matches_S09.loc[:, 'height':'PTS'].sum(axis=1)
potential_matches_S009 = potential_matches_S09[potential_matches_S09['Score'] == 10]


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

features_G09 = record_linkage(initial_data_G, protected_data_G, None, None, thr, gen_cols)

# Get the linkage score
potential_matches_G090 = [features_G09.sum(axis=1) > 1].reset_index()
potential_matches_G090['Score'] = potential_matches_G090.loc[:, 'height':'PTS'].sum(axis=1)
potential_matches_G090 = potential_matches_G090[potential_matches_G090['Score'] >= 11]
potential_matches_G090[potential_matches_G090['Score'] == 15].count()



# %% Initiate record linkage to suppression and noise
features_SN09 = record_linkage(initial_data_S, protected_data_SN, 'born', 'born', thr, gen_cols)

# Get the linkage score
potential_matches_SN00 = f['features_SN0.7'][f['features_SN0.7'].sum(axis=1) > 1].reset_index()
potential_matches_SN090['Score'] = potential_matches_SN090.loc[:, 'height':'PTS'].sum(axis=1)
potential_matches_SN090 = potential_matches_SN090[potential_matches_SN090['Age'] == 1]
potential_matches_SN090 = potential_matches_SN090[(potential_matches_SN090['height'] >= 0.8) &
                                                  (potential_matches_SN090['weight'] >= 0.8)]
potential_matches_SN090 = potential_matches_SN090[potential_matches_SN090['Year'] == 1]
potential_matches_SN090 = potential_matches_SN090[potential_matches_SN090['NBA_Years'] == 1]
potential_matches_SN090[potential_matches_SN090['Score'] > 6].count()

# %% Initiate record linkage to suppression and generalization
features_SG09 = record_linkage(initial_data_SG, protected_data_SG, None, None, thr, gen_cols)

# Get the linkage score
potential_matches_SG090 = [features_SG09.sum(axis=1) > 1].reset_index()
potential_matches_SG090['Score'] = potential_matches_SG090.loc[:, 'height':'PTS'].sum(axis=1)
potential_matches_SG090[potential_matches_SG090['Score'] >= 11].count()
potential_matches_SG090 = potential_matches_SG090[potential_matches_SG090['Score'] >= 11]
dups = potential_matches_SG090[potential_matches_SG070.level_1.duplicated()]

# %% Initiate record linkage to generalization and noise
features_GN09 = record_linkage(protected_data_GN, initial_data_G, None, None, thr, gen_cols)

# Get the linkage score
potential_matches_GN090 = [features_GN09.sum(axis=1) > 1].reset_index()
potential_matches_GN090['Score'] = potential_matches_GN090.loc[:, 'height':'PTS'].sum(axis=1)
potential_matches_GN090[potential_matches_GN090['Score'] >= 12].count()
potential_matches_GN090 = potential_matches_GN090[potential_matches_GN090['Score'] >= 3.9]
potential_matches_GN090 = potential_matches_GN090[
    (potential_matches_GN090.loc[:, 'year_start':'birth_state'].sum(axis=1)
     >= 1) | (potential_matches_GN090.loc[:, 'PER':'PTS'].sum(axis=1) >= 9)]

inds = potential_matches_GN090.groupby(['level_0'])['Score'].transform(max) >= 12

ind = potential_matches_GN090[inds].groupby(['level_0']).Score.idxmax()

df_inds = potential_matches_GN090[inds]

idx = df_inds[df_inds.Score < 12]
idx2 = idx.index.get_level_values(-1)  # get single index
idx1 = potential_matches_GN090.index.get_level_values(-1)

potential_matches_GN090 = potential_matches_GN090[~idx1.isin(idx2)]

potential_matches_GN090['sum'] = potential_matches_GN090.iloc[:, 8:11].sum(axis=1)
potential_matches_GN090 = potential_matches_GN090.loc[
    potential_matches_GN090.groupby('level_0')['sum'].transform(max) == potential_matches_GN090['sum']]

dup = potential_matches_GN090.groupby('level_0').size().reset_index(name='Count')
dup[dup.Count == 1].count()
dup[dup.Count == 2].count()
dups = potential_matches_GN090[potential_matches_GN090.level_1.duplicated()]

# %% Initiate record linkage to suppression, generalization and noise
features_SGN09 = record_linkage(protected_data_SGN, initial_data_SG, None, None, thr, gen_cols)

# Get the linkage score
potential_matches_SGN090 = [features_SGN09.sum(axis=1) > 1].reset_index()
potential_matches_SGN090['Score'] = potential_matches_SGN090.loc[:, 'height':'PTS'].sum(axis=1)
potential_matches_SGN090['Score'].max()

potential_matches_SGN090['max'] = potential_matches_SGN090.loc[:, 'PER':'PTS'].max(axis=1)
potential_matches_SGN090 = potential_matches_SGN090.loc[potential_matches_SGN090.groupby('level_0')['max']
                                                            .transform(max) == potential_matches_SGN090['max']]

dups = potential_matches_SGN090[potential_matches_SGN090.level_1.duplicated()]

