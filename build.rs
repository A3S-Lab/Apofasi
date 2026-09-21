//! Compile the MLX forward shim when the `mlx` feature is enabled.
//!
//! `mlx-sys` builds MLX from source and needs `xcrun metal`. This crate links
//! a prebuilt `libmlx` (the Python wheel layout: `include/` + `lib/libmlx.dylib`
//! + `lib/mlx.metallib`) so Apple Silicon inference does not require that compiler.

use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=MLX_ROOT");
    println!("cargo:rerun-if-changed=native/mlx_forward.cc");
    if std::env::var("CARGO_FEATURE_MLX").is_err() {
        return;
    }
    let root = mlx_root();
    let include = root.join("include");
    let lib = root.join("lib");
    if !include.join("mlx/mlx.h").is_file() || !lib.join("libmlx.dylib").is_file() {
        panic!(
            "MLX_ROOT ({}) must contain include/mlx/mlx.h and lib/libmlx.dylib",
            root.display()
        );
    }

    cc::Build::new()
        .cpp(true)
        .file("native/mlx_forward.cc")
        .include(&include)
        .flag("-std=c++20")
        .warnings(false)
        .opt_level(3)
        .compile("apofasi_mlx");

    println!("cargo:rustc-link-search=native={}", lib.display());
    println!("cargo:rustc-link-lib=dylib=mlx");
    let rpath = lib.display().to_string();
    println!("cargo:rustc-link-arg=-Wl,-rpath,{rpath}");
}

fn mlx_root() -> PathBuf {
    if let Ok(root) = std::env::var("MLX_ROOT") {
        if !root.is_empty() {
            return PathBuf::from(root);
        }
    }
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default());
    let scratch = manifest.join("../../.scratch");
    if let Some(found) = discover_mlx(&scratch) {
        return found;
    }
    panic!("set MLX_ROOT to an MLX package directory (include/mlx/mlx.h and lib/libmlx.dylib)");
}

fn discover_mlx(scratch: &std::path::Path) -> Option<PathBuf> {
    let entries = std::fs::read_dir(scratch).ok()?;
    for entry in entries.flatten() {
        let Ok(pythons) = std::fs::read_dir(entry.path().join("lib")) else {
            continue;
        };
        for python in pythons.flatten() {
            let cand = python.path().join("site-packages/mlx");
            if cand.join("include/mlx/mlx.h").is_file() && cand.join("lib/libmlx.dylib").is_file() {
                return Some(cand);
            }
        }
    }
    None
}
