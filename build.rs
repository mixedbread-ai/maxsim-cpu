use std::env;
use std::path::PathBuf;

fn main() {
    // Tell cargo to rerun this script if libxsmm changes
    println!("cargo:rerun-if-changed=build.rs");
    
    // Platform-specific linking
    #[cfg(all(target_os = "linux", not(feature = "use-libxsmm")))]
    {
        // On Linux without libxsmm, we need to explicitly link system OpenBLAS
        println!("cargo:rustc-link-lib=openblas");
    }
    
    // Only link libxsmm if the feature is enabled
    if cfg!(feature = "use-libxsmm") {
        // Look for LIBXSMM_DIR or LIBXSMM_LIB_DIR
        let libxsmm_dir = env::var("LIBXSMM_DIR")
            .or_else(|_| env::var("LIBXSMM_LIB_DIR").map(|lib_dir| {
                // If we have LIBXSMM_LIB_DIR, assume parent dir is LIBXSMM_DIR
                PathBuf::from(&lib_dir).parent().unwrap().to_string_lossy().to_string()
            }))
            .expect("LIBXSMM_DIR or LIBXSMM_LIB_DIR must be set when using use-libxsmm feature");
        
        let libxsmm_path = PathBuf::from(libxsmm_dir);
        
        // Add library search path
        println!("cargo:rustc-link-search=native={}/lib", libxsmm_path.display());
        
        // Force static linking of libxsmm
        println!("cargo:rustc-link-lib=static=xsmm");
        println!("cargo:rustc-link-lib=static=xsmmext");
        // Don't use xsmmnoblas - we want the BLAS version!
        
        // Link against BLAS (OpenBLAS or system BLAS)
        println!("cargo:rustc-link-lib=openblas");
        
        // Link with standard libraries that libxsmm needs
        println!("cargo:rustc-link-lib=dl");
        println!("cargo:rustc-link-lib=m");
        println!("cargo:rustc-link-lib=pthread");
        println!("cargo:rustc-link-lib=stdc++");
        println!("cargo:rustc-link-lib=gomp");  // OpenMP support
        
        // Tell cargo to rerun if libxsmm libs change
        println!("cargo:rerun-if-changed={}/lib", libxsmm_path.display());
    }
}