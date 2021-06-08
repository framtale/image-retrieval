from numpy.core.defchararray import array
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from PIL import Image
from imutils import build_montages
import numpy as np
import pickle
import cv2
import sys

# We use the euclidean_distance as our default metric
def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

def perform_search(query_image, index, maxResults=10):
	results = []
	for i in range(0, len(index["features"])):
		# Compute the euclidean distance between our query features
		# and the features for the current image in our index, then
		# update our results list with a 2-tuple consisting of the
		# computed distance and the index of the image
		distance = euclidean_distance(query_image, index["features"][i])
		results.append((distance, i))
	# Sort the results and grab the top ones
	results = sorted(results)[:maxResults]
	return results

# Again we use the MNIST dataset as default
print("[INFO] loading MNIST dataset...")
((trainX, _), (testX, _)) = mnist.load_data()

# Add a channel dimension to every image in the dataset, then scale
# the pixel intensities to the range [0, 1]
trainX = np.expand_dims(trainX, axis=-1)
testX = np.expand_dims(testX, axis=-1)
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

# Load the autoencoder model and index from disk
print("[INFO] loading autoencoder and index...")
autoencoder = load_model("autoencoder.h5")
index = pickle.loads(open("feature_vectors.pickle", "rb").read())

# Create the encoder model which consists of *just* the encoder
# portion of the autoencoder
encoder = Model(inputs=autoencoder.input,
	outputs=autoencoder.get_layer("encoded").output)

# Compute the feature vector of our input image
features = encoder.predict(testX)

# Randomly sample a set of testing query image indexes
queryIdxs = list(range(0, testX.shape[0]))
queryIdxs = np.random.choice(queryIdxs, size=10,
	replace=False)
 
# Iterate over the testing indexes
for i in queryIdxs:
	# Take the features for the current image, find all similar
	# images in our dataset, and then initialize our list of result
	# images
	queryFeatures = features[i]
	results = perform_search(queryFeatures, index, maxResults=225)
	images = []

	for (d, j) in results:
		# Grab the result image, convert it back to the previous range
		# and then update the images list
		image = (trainX[j] * 255).astype("uint8")
		image = np.dstack([image] * 3)
		images.append(image)
	# Display the query image
	query = (testX[i] * 255).astype("uint8")
	cv2.imshow("Query", query)
	# Build a montage from the results and display it
	montage = build_montages(images, (28, 28), (15, 15))[0]
	cv2.imshow("Results", montage)
	cv2.waitKey(0)
