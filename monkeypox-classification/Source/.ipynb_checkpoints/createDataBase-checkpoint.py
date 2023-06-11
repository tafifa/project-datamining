import texture as t
import pandas as pd


def createDB():
  # MONKEYPOX
  dfO = t.getDataFrame('../Data/archive/Fold1/Fold1/Fold1/Train/Monkeypox/*.jpg')

  # NON-MONKEYPOX
  dfM = t.getDataFrame('../Data/archive/Fold1/Fold1/Fold1/Train/Others/*.jpg')

  df_concat = pd.concat([dfO, dfM], ignore_index=True)
  df_concat.to_csv('csv/KBase.csv')

if __name__ == "__main__":
  createDB()