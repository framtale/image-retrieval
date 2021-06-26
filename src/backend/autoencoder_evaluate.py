from numpy.core.defchararray import array
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import statistics
import numpy as np
import pickle
import cv2
import sys
import os

# We use the euclidean distance as our default metric
def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

# We use cosine similarity for comparison
def cosine_similarity(a, b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

def perform_search_euclidean(query_image, index, maxResults=10):
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

def perform_search_cosine(query_image, index, maxResults=10):
	results = []
	for i in range(0, len(index["features"])):
		# Compute the cosine similarity between our query features
		# and the features for the current image in our index, then
		# update our results list with a 2-tuple consisting of the
		# computed distance and the index of the image
		distance = cosine_similarity(query_image, index["features"][i])
		results.append((distance, i))
	# Sort the results and grab the top ones
	results = sorted(results)[:maxResults]
	return results

# Again we use the MNIST dataset as default
print("[INFO] loading MNIST dataset...")
((trainX, trainY), (testX, testY)) = mnist.load_data()

# Add a channel dimension to every image in the dataset, then scale
# the pixel intensities to the range [0, 1]
trainX = np.expand_dims(trainX, axis=-1)
testX = np.expand_dims(testX, axis=-1)
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

# We just take the first 500 entries
testX = testX[:500]

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

if not os.path.isfile("euclidean_error.pickle"):
	euclidean_error = []
	print("[INFO] Performing euclidean evaluation...")
	for i in tqdm(range(testX.shape[0])):
		queryfeatures = features[i]
		results = perform_search_euclidean(queryfeatures, index, maxResults=1000)
		label = testY[i]
		error_rate = 0
		for result in results:
			if trainY[result[1]] != label:
				error_rate += 1
		euclidean_error.append(error_rate)

	with open("euclidean_error.pickle", "wb") as epickle:
		pickle.dump(euclidean_error, epickle)

if not os.path.isfile("cosine_error.pickle"):
	cosine_error = []
	print("[INFO] Performing cosine evaluation...")
	for i in tqdm(range(testX.shape[0])):
		queryfeatures = features[i]
		results = perform_search_cosine(queryfeatures, index, maxResults=1000)
		label = testY[i]
		error_rate = 0
		for result in results:
			if trainY[result[1]] != label:
				error_rate += 1
		cosine_error.append(error_rate)

	with open("cosine_error.pickle", "wb") as cpickle:
		pickle.dump(cosine_error, cpickle)

with open("euclidean_error.pickle", "rb") as epickle:
	euclidean_error = pickle.load(epickle)

with open("cosine_error.pickle", "rb") as cpickle:
	cosine_error = pickle.load(cpickle)

euclidean_average = sum(euclidean_error)/len(euclidean_error)
euclidean_median = statistics.median(euclidean_error)
print("euclidean average is {}".format(euclidean_average))
print("euclidean median is {}".format(euclidean_median))

cosine_average = sum(cosine_error)/len(cosine_error)
cosine_median = statistics.median(cosine_error)
print("cosine average is {}".format(cosine_average))
print("cosine median is {}".format(cosine_median))

plt.plot(range(len(testX)), euclidean_error, ".")
plt.savefig("euclidean_error.png")

plt.plot(range(len(testX)), cosine_error, ".")
plt.savefig("cosine_error.png")
print(sum(cosine_error)/len(cosine_error))
