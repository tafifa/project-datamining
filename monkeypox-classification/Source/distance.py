import texture as t
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import time
import os
import glob

def distanceComparison(pathDir, metricOpt):
  start_time = time.time()

  data = pd.read_csv('csv/KBase.csv')
  x = np.array(data[['contrast', 'correlation', 'homogeneity', 'energy']])
  y = np.array(data[['label']]).ravel()

  knn = KNeighborsClassifier(metric=metricOpt, n_neighbors=1)
  knn.fit(x, y)

  path = glob.glob(pathDir)
  valM = 0
  valO = 0

  for item in path:
    # print(item)
    filename, contrast, correlation, homogeneity, energy = t.getData(item)
    inputTest = np.array([contrast, correlation, homogeneity, energy]).reshape(1, -1)
    # print(inputTest)
    result = knn.predict(inputTest).reshape(1, -1)
    # print(result)
    if result == "Monkeypox":
      valM += 1
    else:
      valO += 1
    
  if __name__ != "__main__":
    print(f"Result from {os.path.basename(os.path.dirname(pathDir))}:")
    print('\tPositive =', valM, '\n\tNegative =', valO)
    print("\nTime elapsed for distance comparison: {:.4f}s".format(time.time() - start_time))

  return valM, valO

if __name__ == "__main__":
    distanceComparison('../Data/archive/Original Images/Original Images/Monkey Pox/M01_01.jpg', 'euclidean')