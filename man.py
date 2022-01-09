import pandas as pd
df=pd.read_csv('mm.csv')

state = df.groupby('ped_id')
#print(state.get_group(657))

df2 =  df[df['ped_id']==657]
print(df2)
print(df2.groupby('ped_id')['frame'].transform('count'))
print(df2.groupby('ped_id')['frame'].transform('count') < 2)
# print(df.groupby('ped_id')['frame'].get_group(657))

# short_tracklets_ix = df.index[
#             df.groupby('ped_id')['frame'].transform('count') < 2]
# df.drop(short_tracklets_ix, inplace=True)

# state = df.groupby('ped_id')
# print(pd.__version__)

# print(state.get_group(657))