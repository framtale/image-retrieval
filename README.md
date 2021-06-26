# image-retrieval

This repository provides a brief demonstration to a content-based image retrieval engine which was done due to the proseminar of "feature matching".

# Usage
We recommend to run this repository on Gitpod. Thus, you just have to prefix the URL of this repository given as following:
```
https://gitpod.io/#https://github.com/framtale/image-retrieval
```

After that, you have to install several python dependencies for this project. To make it more comfortable, we added a requirements file. Therefore you have to run the following commands.
```
pip install requirements.txt
sudo apt-get update
sudo apt-get install ffmpeg libsm6 libxext6  -y
```

However, we supplied a command line interface for you, so that you don't need to run each python script one after another. The cli is written in Rust and can be compiled as:
```
cd src/cli/target/release
cargo build --release
```

Finally, you can run the given executable using two predefined flags.

If you just want to search for sample images, you can simply run:
```
./cbir -s
```

Additionally, if you would like to evaluate the performance and plot the accuracy, you can enter the following command:
```
./cbir -p
```
