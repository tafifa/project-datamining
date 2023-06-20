import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

# Read the image
image = cv2.imread('../Data/cell_images/dummy/test/C5NThinF_IMG_20150609_122020_cell_103_U.png')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


### WHITE BACKGROUND
# Threshold the image to create a binary mask
_, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

# Set the black regions to white in the original image
image[mask == 0] = [255, 255, 255]  # Set black pixels to white (BGR value)

# ### BLACK BACKGROUND
# # Threshold the image to create a binary mask
# _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

# # Set the black regions to white in the original image
# image[mask == 0] = [0, 0, 0]  # Set black pixels to white (BGR value)

#### MASKED IMAGE
# # Apply a threshold to create a binary mask for the black regions
# _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

# # Invert the mask to select non-black regions
# image = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))

# # Set the black regions to white in the masked image
# image[mask == 0] = [255, 255, 255]  # Set black pixels to white (BGR value)

# Calculate texture features using GLCM
distances = [1]  # List of distances for co-occurrence matrix
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # List of angles for co-occurrence matrix

# gray_masked = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

# Compute GLCM matrix
glcm = graycomatrix(image, distances, angles)

cv2.imshow('Masked Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Calculate texture properties from the GLCM matrix
contrast = graycoprops(glcm, 'contrast')
dissimilarity = graycoprops(glcm, 'dissimilarity')
homogeneity = graycoprops(glcm, 'homogeneity')
energy = graycoprops(glcm, 'energy')
correlation = graycoprops(glcm, 'correlation')

# Display the calculated texture features
print('Contrast:', contrast)
print('Dissimilarity:', dissimilarity)
print('Homogeneity:', homogeneity)
print('Energy:', energy)
print('Correlation:', correlation)

# Display the resulting masked image
# cv2.imshow('Masked Image', masked_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

'''
MASKED IMAGE
Contrast: [[ 646.61612426 1205.95858135  749.1359447  1252.5483631 ]]
Dissimilarity: [[2.53574951 4.72924934 2.93778802 4.91195437]]
Homogeneity: [[0.99005604 0.98145421 0.98847944 0.98073773]]
Energy: [[0.74315321 0.73927783 0.74267574 0.73881155]]
Correlation: [[0.97729108 0.95739186 0.97363953 0.95574675]]

BLACK BACKGROUND
Contrast: [[233.55396669 447.45419974 282.37804477 457.81338183]]
Dissimilarity: [[2.83815472 4.41848545 3.13874259 4.53505291]]
Homogeneity: [[0.57175675 0.53419681 0.5672242  0.52975316]]
Energy: [[0.32056631 0.31238642 0.31858698 0.31202578]]
Correlation: [[0.97486631 0.95156683 0.96955697 0.9504475 ]]

WHITE BACKGROUND
Contrast: [[124.93587004 216.91559193 131.72547729 231.39266424]]
Dissimilarity: [[2.4122014  3.51441248 2.54794821 3.64712853]]
Homogeneity: [[0.5717573  0.53419796 0.56722494 0.5297543 ]]
Energy: [[0.32056631 0.31238642 0.31858698 0.31202578]]
Correlation: [[0.97854802 0.96256034 0.97734755 0.96006204]]
'''