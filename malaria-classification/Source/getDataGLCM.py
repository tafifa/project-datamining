from skimage.feature import graycomatrix, graycoprops
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import cv2
import glob
import os
import shutil
import pandas as pd
from pathlib import Path

properties = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']

def show(img):
	cv2.imshow("test img", img)

	cv2.waitKey(0)
	cv2.destroyAllWindows()

def getglcm(img):
	agls = [0, np.pi/4, np.pi/2, 3*np.pi/4]
	glcm = graycomatrix(img,
		     							distances=[1],
											angles = agls,
											levels = 256,
											symmetric = True,
											normed = True)
	
	# print(glcm)
	
	feature = []
	glcm_props = [propery for name in properties for propery in graycoprops(glcm, name)[0]]
	for item in glcm_props:
		feature.append(item)
	
	return feature

def getData(pathDir):
	path = glob.glob(pathDir)
	imgs = []
	i = 0
    
	for item in path:
		if i == 2399:
				break
		# print(item)
		i += 1
		img  = cv2.imread(item, cv2.IMREAD_COLOR)
		gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


		_, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

		# Set the black regions to white in the original image
		img[mask == 0] = [255, 255, 255]  # Set black pixels to white (BGR value)
		gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

		# show(gray)
		
		imgs.append(gray)

	glcm_all_agls = []
	for img in imgs: 
		glcm_all_agls.append(getglcm(img))
	
	# print(glcm_all_agls)
	
	return glcm_all_agls

def getDataFrame(path):	
	columns = []
	angles = ['0', '45', '90','135']
	for name in properties :
			for ang in angles:
					columns.append(name + "_" + ang)
        
	# columns.append("label")

	# print(len(columns))

	glcm_all_agls = getData(path)
	
	glcm_df = pd.DataFrame(glcm_all_agls, columns=columns)
	glcm_df = glcm_df.assign(label=os.path.basename(os.path.dirname(path)))

	return glcm_df

def createDatabase():
	dfO = getDataFrame('../Data/cell_images/Parasitized/*')

  # UNINFECTED
	dfM = getDataFrame('../Data/cell_images/Uninfected/*')
	# print(dfO, dfM)

	df_concat = pd.concat([dfO, dfM], ignore_index=True)
	df_concat.to_csv('csv/KBase.csv')

def distanceComparison(pathDir, metricOpt):
	data = pd.read_csv('csv/KBase.csv')
	# len = len(next(zip(*data)))
	# print(len)
	x = np.array(data.iloc[:, 0:24])
	y = np.array(data.iloc[:, 25]).ravel()

	# print(x,y)

	knn = KNeighborsClassifier(metric=metricOpt, n_neighbors=1)
	knn.fit(x, y)

	# print(knn)

	path = glob.glob(pathDir)
	i = 0
    
	for item in path:
		glcmData = getData(item)
		inputTest = np.array(glcmData).reshape(1, -1)
		# print(inputTest)
		result = knn.predict(inputTest).reshape(1, -1)
		filename = Path(item).stem

		# if result == 'Parasitized':
		print('hasil ' , result, filename)
		# 	shutil.copy(item, '../Data/cell_images/dummy/Parasitized/')

if __name__ == '__main__':
	# createDatabase()
	# dir = '../Data/cell_images/dummy/Uninfected/*.png'
	# print(getData(dir))
	createDatabase()
	# distanceComparison(dir, 'canberra')
	
'''

the problem is glcm program calculate the background too, and it affect in the knowledge base
it has been tested with white and black background and return different value

new objective 19/6/2023 13.59
- create database with 2400 data images or more
- then find the true positive and false negative parasitized and uninfected folder
- then save true positive and false negative to dummy folder in each folder
- create database again with same amount of images following the smallest total images
- follow step 2 and 3 respectively, again and again until distance comparison make a good predict
- if it fail, you can try from step 1 with bigger amount data images
- if it fail again, you must consider with thresolding, feature selection, angles selection, and maybe coding review from first
- expectation finishing it will be 1-2 days

'''