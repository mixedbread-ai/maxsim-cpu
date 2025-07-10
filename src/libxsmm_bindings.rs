use libc::{c_char, c_float, c_int, c_void};

// Type aliases matching libxsmm
type LibxsmmBlasint = c_int;  // LP64: 32-bit int

// Opaque types for JIT kernels
pub type LibxsmmKernel = *const c_void;
pub type LibxsmmGemmFunction = unsafe extern "C" fn(
    a: *const c_void,
    b: *const c_void,
    c: *mut c_void,
    pa: *const c_void,
    pb: *const c_void,
    pc: *const c_void,
);

// Kernel flags
pub const LIBXSMM_GEMM_FLAG_NONE: c_int = 0;
pub const LIBXSMM_GEMM_FLAG_TRANS_A: c_int = 1;
pub const LIBXSMM_GEMM_FLAG_TRANS_B: c_int = 2;

// Data types
pub const LIBXSMM_DATATYPE_F32: c_int = 1;

// Prefetch strategies
pub const LIBXSMM_PREFETCH_NONE: c_int = 0;
pub const LIBXSMM_PREFETCH_AL2: c_int = 1;
pub const LIBXSMM_PREFETCH_BL2_VIA_C: c_int = 2;
pub const LIBXSMM_PREFETCH_AL2_AHEAD: c_int = 4;
pub const LIBXSMM_PREFETCH_AL2BL2_VIA_C: c_int = LIBXSMM_PREFETCH_AL2 | LIBXSMM_PREFETCH_BL2_VIA_C;

// GEMM shape structure
#[repr(C)]
pub struct LibxsmmGemmShape {
    pub m: c_int,
    pub n: c_int,
    pub k: c_int,
    pub lda: c_int,
    pub ldb: c_int,
    pub ldc: c_int,
}

// Function bindings
#[link(name = "xsmm")]
extern "C" {
    // Initialize and finalize
    pub fn libxsmm_init();
    pub fn libxsmm_finalize();
    
    // Dispatch to get JIT kernel
    pub fn libxsmm_dispatch_gemm(
        gemm_shape: *const LibxsmmGemmShape,
        gemm_flags: c_int,
        prefetch_flags: c_int,
        dtype: c_int,
    ) -> *const c_void;  // Returns function pointer or null
    
    // Direct GEMM call
    pub fn libxsmm_sgemm(
        transa: *const c_char,
        transb: *const c_char,
        m: *const LibxsmmBlasint,
        n: *const LibxsmmBlasint,
        k: *const LibxsmmBlasint,
        alpha: *const c_float,
        a: *const c_float,
        lda: *const LibxsmmBlasint,
        b: *const c_float,
        ldb: *const LibxsmmBlasint,
        beta: *const c_float,
        c: *mut c_float,
        ldc: *const LibxsmmBlasint,
    );
    
    // The old dispatch API doesn't exist anymore, so we'll use the direct BLAS interface
}


pub unsafe fn xsmm_sgemm(
    transa: u8,
    transb: u8,
    m: i32,
    n: i32,
    k: i32,
    alpha: f32,
    a: *const f32,
    lda: i32,
    b: *const f32,
    ldb: i32,
    beta: f32,
    c: *mut f32,
    ldc: i32,
) {
    let transa_char = transa as c_char;
    let transb_char = transb as c_char;
    let m_blasint = m as LibxsmmBlasint;
    let n_blasint = n as LibxsmmBlasint;
    let k_blasint = k as LibxsmmBlasint;
    let lda_blasint = lda as LibxsmmBlasint;
    let ldb_blasint = ldb as LibxsmmBlasint;
    let ldc_blasint = ldc as LibxsmmBlasint;
    
    libxsmm_sgemm(
        &transa_char,
        &transb_char,
        &m_blasint,
        &n_blasint,
        &k_blasint,
        &alpha,
        a,
        &lda_blasint,
        b,
        &ldb_blasint,
        &beta,
        c,
        &ldc_blasint,
    );
}