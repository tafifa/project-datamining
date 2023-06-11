import texture as t
import pandas as pd
import time


def createDB():
  start_time = time.time()
  # MONKEYPOX
  dfO = t.getDataFrame('../Data/archive/Fold1/Fold1/Fold1/Train/Monkeypox/*.jpg')

  # NON-MONKEYPOX
  dfM = t.getDataFrame('../Data/archive/Fold1/Fold1/Fold1/Train/Others/*.jpg')
  # print(dfO, dfM)

  df_concat = pd.concat([dfO, dfM], ignore_index=True)
  df_concat.to_csv('./csv/KBase.csv')
  print("Time elapsed for create database: {:.4f}s\n".format(time.time() - start_time))

if __name__ == "__main__":
  createDB()