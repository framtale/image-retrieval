use std::{
    path::Path,
    process::Command
};

use structopt::StructOpt;

#[derive(StructOpt)]
#[structopt(name= "CBIR", about = "utility to perform reverse image search")]
struct SearchEngine {
    /// Set this flag to perforiming image search for a sample set of images
    #[structopt(short, long)]
    search: bool,
    /// Plot the overall accuracy
    #[structopt(short, long)]
    plot: bool,
}

fn main() -> std::io::Result<()> {
    // We have to check whether we already trained our autoencoder.
    // If this isn't the case, we have to train it and compute the feature vectors
    if !Path::new("./../../../backend/autoencoder.h5").exists() {
        Command::new("python3")
                .arg("./../../../backend/train_autoencoder.py")
                .status()?;

        Command::new("python3")
                .arg("./../../../backend/compute_feature_vectors.py")
                .status()?;
    }

    let cbir = SearchEngine::from_args();
    
    if cbir.search {
        Command::new("python3")
                .arg("./../../../backend/image_search.py")
                .status()?;
    }
    if cbir.plot {
        Command::new("python3")
                .arg("./../../../backend/autoencoder_evaluate.py")
                .status()?;
    }

    Ok(())

}
