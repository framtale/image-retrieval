use std::{
    path::PathBuf,
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
    let args = SearchEngine::from_args();

    Command::new("python3")
            .args(&["../autoencoder.py", args.img_path.to_str().unwrap()])
            .status()?;

    Ok(())
}
