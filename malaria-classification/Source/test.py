import pandas as pd

data = pd.read_csv('csv/blackbg/KBase.csv')
len = len(next(zip(*data)))
print(len)