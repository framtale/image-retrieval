from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
import numpy as np
import pickle
import json

# We use the MNIST dataset as default again
print("[INFO] loading MNIST training split...")
((trainX, _), (_testX, _)) = mnist.load_data()

# Add a channel dimension to every image in the training split, then
# scale the pixel intensities to the range [0, 1]
trainX = np.expand_dims(trainX, axis=-1)
trainX = trainX.astype("float32") / 255.0

print("[INFO] loading autoencoder model...")
autoencoder = load_model("./autoencoder.h5")

# Create the encoder model which consists of *just* the encoder
# portion of the autoencoder
encoder = Model(inputs=autoencoder.input,
	outputs=autoencoder.get_layer("encoded").output)

# We have to compute the feature vectors here
print("[INFO] encoding images...")
features = encoder.predict(trainX)

# Construct a dictionary that maps the index of the MNIST training
# image to its corresponding vector representation
indexes = list(range(0, trainX.shape[0]))

data = {"indexes": indexes, "features": features}

# Write the data dictionary to disk
print("[INFO] saving index...")
file = open("feature_vectors.pickle", "wb")
file.write(pickle.dumps(data))
file.close()