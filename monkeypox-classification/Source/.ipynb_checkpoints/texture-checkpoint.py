import cv2
import numpy as np
import glob
import skimage.feature as feature
import pandas as pd
import os
from pathlib import Path

def getData(pathDir):
  path = glob.glob(pathDir)
  filename = []
  images = []
  contrast = []
  correlation = []
  homogeneity = []
  energy = []

  for imagepath in path:
    image_spot = cv2.imread(imagepath,cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image_spot, cv2.COLOR_BGR2GRAY)
    name = Path(imagepath).stem
    filename.append(name)
    images.append(gray)

  for item in images:
    graycom = feature.graycomatrix(item, [1], [3*np.pi/4], levels=256)
    contrast.append(feature.graycoprops(graycom, 'contrast').item())
    correlation.append(feature.graycoprops(graycom, 'correlation').item())
    homogeneity.append(feature.graycoprops(graycom, 'homogeneity').item())
    energy.append(feature.graycoprops(graycom, 'energy').item())

  return filename, contrast, correlation, homogeneity, energy

def getDataFrame(path):
  filename, contrast, correlation, homogeneity, energy = getData(path)
  header = ['filename', 'contrast', 'correlation', 'homogeneity', 'energy']
  df = pd.DataFrame(list(zip(filename, contrast, correlation, homogeneity, energy)), columns=header)
  df = df.assign(label=os.path.basename(os.path.dirname(path)))

  if __name__ == "__main__":
    print(df)

  return df

