use std::{
    path::{Path, PathBuf},
    process::Command
};

use structopt::StructOpt;

#[derive(StructOpt)]
#[structopt(about = "utility to perform reverse image search")]
struct SearchEngine {
    /// Query this image to find similar images
    img_path: PathBuf,
}

fn main() -> std::io::Result<()> {
    // We have to check whether we already trained our autoencoder.
    // If this isn't the case, we have to train it and compute the feature vectors
    if !Path::new("autoencoder.h5").exists() {
        Command::new("python3")
                .arg("./../../../backend/train_autoencoder.py")
                .status()?;

        Command::new("python3")
                .arg("./../../../backend/compute_feature_vectors.py")
                .status()?;
    }

    Command::new("python3")
            .arg("./../../../backend/image_search.py")
            .status()?;
    Ok(())
}
