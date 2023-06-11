# import python file from directory
import createDataBase as cDB
import texture as t
import distance as d

### analyze distance comparison for image from path
# cDB.createDB()

pathTest = '../Data/archive/Original Images/Original Images/Others/*.jpg'
metric = 'euclidean'

positive, negative = d.distanceComparison(pathTest, metric)

print("\nHasil Prediksi adalah","Positif" if positive > negative else "Negatif")