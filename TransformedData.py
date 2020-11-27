import os
import pandas as pd
import numpy as np

player_data = pd.read_csv('NBA_Data/player_data.csv')
players = pd.read_csv('NBA_Data/Players.csv')
season = pd.read_csv('NBA_Data/Seasons_Stats.csv')
injuries = pd.read_csv('NBA_Data/injuries_2010-2020.csv')

del players['Unnamed: 0']
del season['Unnamed: 0']

#########################
#       ANALYSIS        #
#########################

player_data.shape
players.shape
season.shape
injuries.shape

season['Player'].describe()
players['Player'].describe()
player_data['name'].describe()

player_data[(~player_data.name.isin(season.Player))]
player_data[(~player_data.name.isin(players.Player))]
players[(~players.Player.isin(season.Player))]

acquired = injuries.groupby(['Acquired']).size().reset_index(name='Count')
notes = injuries.groupby(['Notes']).size().reset_index(name='Count')
acquired = injuries.groupby(['Acquired', 'Notes']).size().reset_index(name='Count')

player_data_name = player_data.groupby(['name']).size().reset_index(name='Count')
season_name = season.groupby(['Player']).size().reset_index(name='Count')

injuries.info()

check_acq_relq = injuries.loc[(injuries['Relinquished'] == 'Al Horford') | (injuries['Acquired'] == 'Al Horford')]

##########################
#       Processing       #
##########################

# Get data over 2010
# player_data = player_data.loc[player_data['year_start'] >= 2010]
season = season.loc[season['Year'] >= 2010]

# Get players born after 1977 on players dataset
player_data.loc[player_data['name'] == 'Chris Johnson']
players.loc[players['Player'] == 'Chris Johnson']
players = players.loc[players['born'] >= 1977]

# Fix born on A.J. Price
players['born'] = np.where(players['Player'] == 'A.J. Price', 1986, players['born'])

# Remove space at the end of the name
player_data['name'] = player_data['name'].str.rstrip()
season['Player'] = season['Player'].str.rstrip()
players['Player'] = players['Player'].str.rstrip()

# Create birth year attribute in player_data
player_data['born'] = player_data['birth_date'].apply(lambda x: str(x)[-4:])

# Select year_start, year_end and college from player_data
player_data.rename(columns={'name': 'Player'}, inplace=True)
player_data_reduce = player_data[['Player', 'year_start', 'year_end', 'college', 'born']]

# Add players info to the season_stats
player_data_reduce['born'][player_data_reduce['born'] == 'nan'] = -1
player_data_reduce['born'] = player_data_reduce['born'].astype(int)

players['born'][players['born'].isnull()] = -1
players['born'] = players['born'].astype(int)

merge_data = players.merge(player_data_reduce, on=['Player', 'born'], how='left')

merge_data = merge_data[merge_data['born'] != -1]


# Add year_start (by players name and born) to season_stats
merge_data = pd.merge(merge_data, season, how='left')

merge_data.info()

merge_data['NBA_Years'] = merge_data['Year'] - merge_data['year_start']

merge_data['NBA_Years'].describe()

# Remove empty attribute
del merge_data['blank2']
del merge_data['blanl']

# Delete the rows with no information on the season year
merge_data = merge_data[merge_data['Year'].notnull()].reset_index(drop=True)

player_teams = merge_data.groupby(['Player', 'Tm']).size().reset_index(name='Count')
check_years_played = merge_data.loc[merge_data['Player'] == 'A.J. Price']  # data[2010-2015], reality:[2009-2017]

# WORK ON INJURIES
# Drop row with information about retirement
injuries = injuries.drop(injuries.index[6039]).reset_index(drop=True)

# Assign new attribute with player's name
injuries['Player'] = injuries['Relinquished']
injuries['Player'].fillna(injuries['Acquired'], inplace=True)

# Simplifying the notes
injuries['Case'] = np.nan
injuries['Case'] = np.where(injuries['Relinquished'].isna(), 'returned to lineup', injuries['Case'])

del injuries['Acquired']

for i in range(0, len(injuries)):
    if 'out indefinitely' in injuries['Notes'][i]:
        injuries['Case'][i] = 'DNP'
    elif 'DTD' in injuries['Notes'][i]:
        injuries['Case'][i] = 'DTD'
    elif 'out for season' in injuries['Notes'][i]:
        injuries['Case'][i] = 'out for season'
    elif 'returned to lineup' in injuries['Notes'][i]:
        injuries['Case'][i] = 'returned to lineup'
    elif 'out for seaon' in injuries['Notes'][i]:
        injuries['Case'][i] = 'out for season'
    elif '(out) indefinitely' in injuries['Notes'][i]:
        injuries['Case'][i] = 'DNP'
    elif 'out indefinitley' in injuries['Notes'][i]:
        injuries['Case'][i] = 'DNP'
    elif 'out indefiinitely' in injuries['Notes'][i]:
        injuries['Case'][i] = 'DNP'
    elif 'out indefinteily' in injuries['Notes'][i]:
        injuries['Case'][i] = 'DNP'
    elif 'DNP' in injuries['Notes'][i]:
        injuries['Case'][i] = 'DNP'

# Remove 'returned to lineup': did play
injuries = injuries[~(injuries['Case'].isin(['returned to lineup']))].reset_index(drop=True)

# Assign new attribute with the season year
injuries['Date'] = pd.to_datetime(injuries['Date'])
injuries['Year'] = injuries['Date'].apply(lambda x: x.year if x.month < 10 else x.year + 1)

# Case study
cases_nan = injuries[injuries['Case'] == 'nan']
steve = injuries.loc[injuries['Player'] == 'Steve Nash']

# Replace nan by DNP
injuries['Case'] = np.where(injuries['Case'] == 'nan', 'DNP', injuries['Case'])

# Order by Player and year
injuries = injuries.sort_values(by=['Player', 'Date'])

# How many days pass between the injuries?
injuries['Days'] = injuries.groupby(['Year', 'Player']).Date.apply(lambda x: x - x.shift(1))
injuries['Days'] = injuries['Days'].apply(lambda x: x.days)
injuries['Days'] = injuries['Days'].fillna(0)
# There are games every two days (minimal). Consider 7 for lack of information
injuries['Days'] = injuries['Days'].apply(lambda x: x if x <= 7 else 0)

# Flag to facilitate the calculation of maximum days with injury
injuries['flag'] = np.where((injuries['Days'] != 0) & (injuries['Case'] != 'DTD'), 1, 0)

# Get max days with injuries
gb = injuries.groupby((injuries['flag'] != injuries['flag'].shift()).cumsum())
injuries['DaysSum'] = gb['Days'].cumsum()
injuries['DaysSum'] = np.where(injuries.flag == 0, 0, injuries.DaysSum)
injuries['MaxDays'] = injuries.groupby(['Year', 'Player']).DaysSum.transform('max')

# Assign severe injury for all season if player is 'out for season'
injuries['SevereInjury'] = (injuries.groupby(['Year', 'Player']).Case
                            .transform(lambda x: [1] * len(x) if x.str.contains('out for season').any() else 0))

# Assign severe injury if player did not play >15 days in each season
injuries['SevereInjury'] = np.where((injuries.MaxDays >= 15) & (injuries.SevereInjury != 1), 1, injuries.SevereInjury)

# Case study
ab = injuries.loc[(injuries['Player'] == 'O.J. Mayo') | (injuries['Player'] == 'Ario Chalmers') | (
            injuries['Player'] == 'Mike Miller')
                  | (injuries['Player'] == 'Danilo Gallinari')
                  | (injuries['Player'] == 'Kobe Bryant')]

# Leasao grave a partir do 15º dia ou considerar tudo como lesao??

# Clean Team
tm = merge_data.groupby('Tm').size().reset_index(name='Count')
team = injuries.groupby('Team').size().reset_index(name='Count')

injuries['Tm'] = injuries['Team']
injuries['Tm'] = np.where(injuries['Tm'] == '76ers', 'PHI', injuries['Tm'])
injuries['Tm'] = np.where(injuries['Tm'] == 'Blazers', 'POR', injuries['Tm'])
#injuries['Tm'] = np.where(injuries['Tm'] == 'Bobcats', '', injuries['Tm'])
injuries['Tm'] = np.where(injuries['Tm'] == 'Bucks', 'MIL', injuries['Tm'])
injuries['Tm'] = np.where(injuries['Tm'] == 'Bulls', 'CHI', injuries['Tm'])
injuries['Tm'] = np.where(injuries['Tm'] == 'Cavaliers', 'CLE', injuries['Tm'])
injuries['Tm'] = np.where(injuries['Tm'] == 'Celtics', 'BOS', injuries['Tm'])
injuries['Tm'] = np.where(injuries['Tm'] == 'Clippers', 'LAC', injuries['Tm'])
injuries['Tm'] = np.where(injuries['Tm'] == 'Grizzlies', 'MEM', injuries['Tm'])
injuries['Tm'] = np.where(injuries['Tm'] == 'Hawks', 'ATL', injuries['Tm'])
injuries['Tm'] = np.where(injuries['Tm'] == 'Heat', 'MIA', injuries['Tm'])
injuries['Tm'] = np.where(injuries['Tm'] == 'Hornets', 'CHA', injuries['Tm'])
injuries['Tm'] = np.where(injuries['Tm'] == 'Jazz', 'UTA', injuries['Tm'])
injuries['Tm'] = np.where(injuries['Tm'] == 'Kings', 'SAC', injuries['Tm'])
injuries['Tm'] = np.where(injuries['Tm'] == 'Knicks', 'NYK', injuries['Tm'])
injuries['Tm'] = np.where(injuries['Tm'] == 'Lakers', 'LAL', injuries['Tm'])
injuries['Tm'] = np.where(injuries['Tm'] == 'Magic', 'ORL', injuries['Tm'])
injuries['Tm'] = np.where(injuries['Tm'] == 'Mavericks', 'DAL', injuries['Tm'])
injuries['Tm'] = np.where(injuries['Tm'] == 'Nets', 'BKN', injuries['Tm'])
injuries['Tm'] = np.where(injuries['Tm'] == 'Nuggets', 'DEN', injuries['Tm'])
injuries['Tm'] = np.where(injuries['Tm'] == 'Pacers', 'IND', injuries['Tm'])
injuries['Tm'] = np.where(injuries['Tm'] == 'Pelicans', 'NOP', injuries['Tm'])
injuries['Tm'] = np.where(injuries['Tm'] == 'Pistons', 'DET', injuries['Tm'])
injuries['Tm'] = np.where(injuries['Tm'] == 'Raptors', 'TOR', injuries['Tm'])
injuries['Tm'] = np.where(injuries['Tm'] == 'Rockets', 'HOU', injuries['Tm'])
injuries['Tm'] = np.where(injuries['Tm'] == 'Spurs', 'SAS', injuries['Tm'])
injuries['Tm'] = np.where(injuries['Tm'] == 'Suns', 'PHX', injuries['Tm'])
injuries['Tm'] = np.where(injuries['Tm'] == 'Thunder', 'OKC', injuries['Tm'])
injuries['Tm'] = np.where(injuries['Tm'] == 'Timberwolves', 'MIN', injuries['Tm'])
injuries['Tm'] = np.where(injuries['Tm'] == 'Warriors', 'GSW', injuries['Tm'])
injuries['Tm'] = np.where(injuries['Tm'] == 'Wizards', 'WAS', injuries['Tm'])

# Group injuries by year
injuries_reduce = injuries.loc[injuries.groupby(['Player', 'Year', 'Tm']).Year.idxmin()]

abb = injuries_reduce.loc[(injuries_reduce['Player'] == 'O.J. Mayo') | (injuries_reduce['Player'] == 'Ario Chalmers') | (
            injuries_reduce['Player'] == 'Mike Miller')
                  | (injuries_reduce['Player'] == 'Danilo Gallinari')
                          | (injuries_reduce['Player'] == 'Kobe Bryant')]

# Get the needed data to merge
injuries_reduce = injuries_reduce[['Tm', 'Player', 'Year', 'SevereInjury']]

# Merge previous data with injuries
final_data = pd.merge(merge_data, injuries_reduce, on=['Player', 'Year', 'Tm'], how='left')

# Transform SevereInjury in multiclass
final_data['SevereInjury'] = np.where(final_data['SevereInjury'] == 1, 3, final_data['SevereInjury'])
final_data['SevereInjury'] = np.where(final_data['SevereInjury'] == 0, 2, final_data['SevereInjury'])
final_data['SevereInjury'] = np.where(final_data['SevereInjury'].isnull(), 1, final_data['SevereInjury'])

sevinj = final_data.groupby('SevereInjury').size().reset_index(name='Count')

# Delete college
del final_data['college']

final_data['NBA_Years'] = final_data['NBA_Years'].fillna(final_data['Year'].sub(final_data.groupby('Player')['Year'].transform('first')))

# Keep distinct players (there are players in many teams in the same season)
final_data = final_data.loc[final_data.groupby(['Player', 'Year'])['SevereInjury'].idxmax()]


# G-Games; GS-Game Started; MP-Minutes Played per game ; PER-Player Efficiency Rating ; TS%-True Shooting percentage;
# 3PAr-3Point Attempt Rate; FTr-Free Trows Rate; ORB%-Offensive Rebounds per game percentage;
# DRB%-Defensive Rebounds percentage; TRB%-Total Rebounds per game; AST%-Assists percentage; STL%-Steals percentage;
# BLK%-Blocks percentage; TOV%-Turnovers percentage; USG%-Usage percentage; OWS-offensive win shares;
# DWS-Defensive Win shares; WS-Win Shares; WS/48- Win shares every 48 min; OBPM-Offensive box plus/minus;
# DBPM-Defensive box plus/minus; BPM-Box plus/minus; VORP-Value over replacement player;
# FG-Field Goals per game; FGA-Field Goals Attempts per game; FG%-Field Goal percentage; 3P-3Point Field goals per game
# 3PA-3 Point Field Goals Attempted; 3P%-3 Point Field Goal Percentage; 2P-2 Point Field Goal;
# 2P%-2 Point Field Goal Percentage; 2P%-2 Point Field Goal Percentage; eFG%-Effective goal percentage;
# FT-Free Throws per game; FTA-Free Throws Attempts per game; FT%-Free Throws percentage; ORB-Offensive Rebounds;
# DRB-Defensive Rebounds; TRB-Total rebound; AST-Assists per game; STL-Steal per game; BLK-Block per game;
# TOV-Turnover per game; PF-Personal fouls per game; PTS-Point per game
