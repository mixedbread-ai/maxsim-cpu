//! MaxSim - High-performance BLAS GEMM → per-row max → sum.
//!
//! Requires either:
//! - x86_64 with AVX2 (Intel Haswell 2013+, AMD Excavator 2015+)
//! - ARM64/AArch64 (Apple Silicon, AWS Graviton, etc.)

#[cfg(not(any(
    all(target_arch = "x86_64", target_feature = "avx2"),
    target_arch = "aarch64"
)))]

use blas::sgemm;
use numpy::{PyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::cell::RefCell;

#[cfg(feature = "use-libxsmm")]
use libc::c_void;

#[cfg(feature = "use-libxsmm")]
mod libxsmm_bindings;


// Thread-local buffers to avoid repeated allocations
thread_local! {
    static SIMILARITY_BUFFER: RefCell<Vec<f32>> = RefCell::new(Vec::new());
    static TEMP_BUFFER: RefCell<Vec<f32>> = RefCell::new(Vec::new());
    static BATCH_BUFFER: RefCell<Vec<f32>> = RefCell::new(Vec::with_capacity(1024 * 1024));
}

// Initialize libxsmm only once
#[cfg(feature = "use-libxsmm")]
use std::sync::Once;

#[cfg(feature = "use-libxsmm")]
static LIBXSMM_INIT: Once = Once::new();

// SIMD module with platform-specific implementations.
// TODO: AVX512?
mod simd {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    
    #[cfg(target_arch = "aarch64")]
    use std::arch::aarch64::*;
    
    /// Find max w/ AVX2 and prefetching.
    #[cfg(target_arch = "x86_64")]
    #[inline]
    pub fn simd_max_avx2(slice: &[f32]) -> f32 {
        if slice.len() < 8 {
            return slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        }
        
        unsafe {
            // Use 4 vectors for better ILP (Instruction Level Parallelism)
            let mut max_vec0 = _mm256_set1_ps(f32::NEG_INFINITY);
            let mut max_vec1 = _mm256_set1_ps(f32::NEG_INFINITY);
            let mut max_vec2 = _mm256_set1_ps(f32::NEG_INFINITY);
            let mut max_vec3 = _mm256_set1_ps(f32::NEG_INFINITY);
            
            let mut i = 0;
            
            // Process 32 elements at a time (4x8) for better ILP
            while i + 32 <= slice.len() {
                // Prefetch next cache line
                _mm_prefetch(slice.as_ptr().add(i + 64) as *const i8, _MM_HINT_T0);
                
                let data0 = _mm256_loadu_ps(slice.as_ptr().add(i));
                let data1 = _mm256_loadu_ps(slice.as_ptr().add(i + 8));
                let data2 = _mm256_loadu_ps(slice.as_ptr().add(i + 16));
                let data3 = _mm256_loadu_ps(slice.as_ptr().add(i + 24));
                
                max_vec0 = _mm256_max_ps(max_vec0, data0);
                max_vec1 = _mm256_max_ps(max_vec1, data1);
                max_vec2 = _mm256_max_ps(max_vec2, data2);
                max_vec3 = _mm256_max_ps(max_vec3, data3);
                
                i += 32;
            }
            
            // Process remaining groups of 8
            while i + 8 <= slice.len() {
                let data = _mm256_loadu_ps(slice.as_ptr().add(i));
                max_vec0 = _mm256_max_ps(max_vec0, data);
                i += 8;
            }
            
            // Combine the 4 vectors
            max_vec0 = _mm256_max_ps(max_vec0, max_vec1);
            max_vec2 = _mm256_max_ps(max_vec2, max_vec3);
            max_vec0 = _mm256_max_ps(max_vec0, max_vec2);
            
            // Horizontal max within the final vector
            let high = _mm256_extractf128_ps(max_vec0, 1);
            let low = _mm256_castps256_ps128(max_vec0);
            let max128 = _mm_max_ps(high, low);
            
            let shuffled = _mm_shuffle_ps(max128, max128, 0b01001110);
            let max64 = _mm_max_ps(max128, shuffled);
            let shuffled2 = _mm_shuffle_ps(max64, max64, 0b00000001);
            let final_max = _mm_max_ps(max64, shuffled2);
            
            let mut result = _mm_cvtss_f32(final_max);
            
            // Handle remaining elements
            for j in i..slice.len() {
                result = result.max(slice[j]);
            }
            
            result
        }
    }
    
    /// Find max w/ ARM NEON.
    #[cfg(target_arch = "aarch64")]
    #[inline]
    pub fn simd_max_avx2(slice: &[f32]) -> f32 {
        if slice.len() < 4 {
            return slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        }
        
        unsafe {
            // Initialize 4 vectors for better ILP
            let mut max_vec0 = vdupq_n_f32(f32::NEG_INFINITY);
            let mut max_vec1 = vdupq_n_f32(f32::NEG_INFINITY);
            let mut max_vec2 = vdupq_n_f32(f32::NEG_INFINITY);
            let mut max_vec3 = vdupq_n_f32(f32::NEG_INFINITY);
            
            let mut i = 0;
            
            // Process 16 elements at a time (4x4)
            while i + 16 <= slice.len() {
                let data0 = vld1q_f32(slice.as_ptr().add(i));
                let data1 = vld1q_f32(slice.as_ptr().add(i + 4));
                let data2 = vld1q_f32(slice.as_ptr().add(i + 8));
                let data3 = vld1q_f32(slice.as_ptr().add(i + 12));
                
                max_vec0 = vmaxq_f32(max_vec0, data0);
                max_vec1 = vmaxq_f32(max_vec1, data1);
                max_vec2 = vmaxq_f32(max_vec2, data2);
                max_vec3 = vmaxq_f32(max_vec3, data3);
                
                i += 16;
            }
            
            // Process remaining groups of 4
            while i + 4 <= slice.len() {
                let data = vld1q_f32(slice.as_ptr().add(i));
                max_vec0 = vmaxq_f32(max_vec0, data);
                i += 4;
            }
            
            // Combine the 4 vectors
            max_vec0 = vmaxq_f32(max_vec0, max_vec1);
            max_vec2 = vmaxq_f32(max_vec2, max_vec3);
            max_vec0 = vmaxq_f32(max_vec0, max_vec2);
            
            // Horizontal max within the final vector
            let max_pair = vmaxq_f32(max_vec0, vextq_f32(max_vec0, max_vec0, 2));
            let max_val = vmaxq_f32(max_pair, vextq_f32(max_pair, max_pair, 1));
            let mut result = vgetq_lane_f32(max_val, 0);
            
            // Handle remaining elements
            for j in i..slice.len() {
                result = result.max(slice[j]);
            }
            
            result
        }
    }
    
    /// Fallback-ish. Really not great.
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    #[inline]
    pub fn simd_max_avx2(slice: &[f32]) -> f32 {
        slice.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    }
}

// MaxSim algorithm.
mod algorithm {
    use super::*;
    use crate::simd::simd_max_avx2;
    use blas::sgemm;
    
    /// Process a single variable-length document directly
    fn process_single_doc(
        q: &[f32],           // [q_len * dim]
        doc: &[f32],         // [doc_len * dim]
        q_len: usize,
        doc_len: usize,
        dim: usize,
    ) -> f32 {
        // Use thread-local buffer to avoid allocations
        SIMILARITY_BUFFER.with(|buffer| {
            let mut buffer = buffer.borrow_mut();
            buffer.resize(q_len * doc_len, 0.0);
            
            // Compute Q × D^T
            unsafe {
                sgemm(
                    b'T', b'N',
                    doc_len as i32,
                    q_len as i32,
                    dim as i32,
                    1.0,
                    doc, dim as i32,
                    q, dim as i32,
                    0.0,
                    buffer.as_mut_slice(), doc_len as i32,
                );
            }
            
            // Find max for each query and sum
            let mut score = 0.0f32;
            for qi in 0..q_len {
                let start = qi * doc_len;
                let query_sims = &buffer[start..start + doc_len];
                score += simd_max_avx2(query_sims);
            }
            
            score
        })
    }
    
    /// Fused GEMM+reduction with document tiling
    pub fn maxsim_fused_doc_tiles(
        q: &[f32],           // [q_len * dim]
        d: &[f32],           // [n_docs * d_len * dim]
        q_len: usize,
        d_len: usize,
        dim: usize,
    ) -> Vec<f32> {
        let n_docs = d.len() / (d_len * dim);
        
        // For macOS/ARM, use more efficient processing strategy
        #[cfg(target_arch = "aarch64")]
        {
            // Process documents in parallel without excessive tiling
            // ARM has unified memory architecture, so tiling is not anywhere near as important.
            (0..n_docs).into_par_iter().map(|doc_idx| {
                let doc_offset = doc_idx * d_len * dim;
                let doc_data = &d[doc_offset..doc_offset + d_len * dim];
                
                // Process in smaller blocks to fit in L2 cache
                let block_size = 64; // Claude says this is the best value to fit in cache for most Apple chips.
                let mut max_vals = vec![f32::NEG_INFINITY; q_len];
                
                for block_start in (0..d_len).step_by(block_size) {
                    let block_end = (block_start + block_size).min(d_len);
                    let actual_block_size = block_end - block_start;
                    
                    // Compute similarities for this block
                    let mut block_sims = vec![0.0f32; q_len * actual_block_size];
                    let block_data = &doc_data[block_start * dim..block_end * dim];
                    
                    unsafe {
                        sgemm(
                            b'T', b'N',
                            actual_block_size as i32,
                            q_len as i32,
                            dim as i32,
                            1.0,
                            block_data, dim as i32,
                            q, dim as i32,
                            0.0,
                            &mut block_sims, actual_block_size as i32,
                        );
                    }
                    
                    // Update max values using NEON
                    for qi in 0..q_len {
                        let base_idx = qi * actual_block_size;
                        let query_sims = &block_sims[base_idx..base_idx + actual_block_size];
                        let max_val = simd_max_avx2(query_sims);
                        max_vals[qi] = max_vals[qi].max(max_val);
                    }
                }
                
                // Sum max values
                max_vals.iter().sum()
            }).collect()
        }
        
        #[cfg(not(target_arch = "aarch64"))]
        {
            let mut results = vec![0.0f32; n_docs];
            
            // x86 tiling strategy
            let doc_tile_size = match d_len {
                512 => 128,
                1024 => 64,
                2048 => 32,
                4096 => 16,
                _ => 32,
            };
            
            for doc_tile_start in (0..n_docs).step_by(doc_tile_size) {
                let doc_tile_end = (doc_tile_start + doc_tile_size).min(n_docs);
                let tile_docs = doc_tile_end - doc_tile_start;
                let tile_tokens = tile_docs * d_len;
                
                let mut tile_sims = vec![0.0f32; q_len * tile_tokens];
                let tile_d_start = doc_tile_start * d_len * dim;
                let tile_d_end = doc_tile_end * d_len * dim;
                let tile_d = &d[tile_d_start..tile_d_end];
                
                unsafe {
                    sgemm(
                        b'T', b'N',
                        tile_tokens as i32,
                        q_len as i32,
                        dim as i32,
                        1.0,
                        tile_d, dim as i32,
                        q, dim as i32,
                        0.0,
                        &mut tile_sims, tile_tokens as i32,
                    );
                }
                
                let tile_results: Vec<f32> = (0..tile_docs).into_par_iter().map(|tile_doc_idx| {
                    let doc_start = tile_doc_idx * d_len;
                    let mut score = 0.0f32;
                    
                    for qi in 0..q_len {
                        let base_idx = doc_start + qi * tile_tokens;
                        let doc_sims = &tile_sims[base_idx..base_idx + d_len];
                        let max_val = simd_max_avx2(doc_sims);
                        score += max_val;
                    }
                    
                    score
                }).collect();
                
                for (i, &score) in tile_results.iter().enumerate() {
                    results[doc_tile_start + i] = score;
                }
            }
            
            results
        }
    }
    
    pub fn maxsim_ultra_adaptive(
        q: &[f32],           // [q_len * dim]
        d: &[f32],           // [n_docs * d_len * dim]
        q_len: usize,
        d_len: usize,
        dim: usize,
    ) -> Vec<f32> {
        #[cfg(feature = "use-libxsmm")]
        {
            crate::libxsmm::maxsim_libxsmm_clean(q, d, q_len, d_len, dim)
        }
        
        #[cfg(not(feature = "use-libxsmm"))]
        {
            maxsim_fused_doc_tiles(q, d, q_len, d_len, dim)
        }
    }
    
    /// Process variable-length documents with optimized batching
    pub fn maxsim_variable_length(
        q: &[f32],                                    // [q_len * dim]
        doc_infos: Vec<(usize, usize, &[f32])>,     // [(doc_idx, doc_len, doc_data)]
        q_len: usize,
        dim: usize,
    ) -> Vec<f32> {
        #[cfg(feature = "use-libxsmm")]
        {
            return crate::libxsmm::maxsim_libxsmm_variable(q, doc_infos, q_len, dim);
        }
        
        #[cfg(not(feature = "use-libxsmm"))]
        {
            let n_docs = doc_infos.len();
            let mut results = vec![0.0f32; n_docs];
            
            // Fast path: if all documents have similar lengths, process in one batch
            let (min_len, max_len) = doc_infos.iter()
                .map(|(_, len, _)| *len)
                .fold((usize::MAX, 0), |(min, max), len| (min.min(len), max.max(len)));
            
            if max_len as f32 / min_len as f32 <= 1.2 && n_docs >= 50 {
                // All documents have similar lengths - process in single batch
                return BATCH_BUFFER.with(|buffer| {
                    let mut buffer = buffer.borrow_mut();
                    let required_size = n_docs * max_len * dim;
                    buffer.resize(required_size, 0.0);
                    buffer.fill(0.0);
                    
                    // Fill all documents
                    for (idx, (_, doc_len, doc_data)) in doc_infos.iter().enumerate() {
                        let src_size = doc_len * dim;
                        let dst_offset = idx * max_len * dim;
                        buffer[dst_offset..dst_offset + src_size]
                            .copy_from_slice(&doc_data[..src_size]);
                    }
                    
                    // Process all at once
                    let batch_results = maxsim_fused_doc_tiles(
                        q, &buffer[..required_size], q_len, max_len, dim
                    );
                    
                    // Results are already in correct order
                    let mut final_results = vec![0.0f32; n_docs];
                    for (idx, (doc_idx, _, _)) in doc_infos.iter().enumerate() {
                        final_results[*doc_idx] = batch_results[idx];
                    }
                    final_results
                });
            }
            
            // Sort documents by length for better batching
            let mut sorted_indices: Vec<usize> = (0..n_docs).collect();
            sorted_indices.sort_by_key(|&i| doc_infos[i].1);
            
            // Process in larger batches with adaptive sizing
            let target_batch_size = 128; // Larger batches for better GEMM efficiency
            let mut i = 0;
            
            while i < n_docs {
                // Find batch end - include docs within 20% length difference
                let base_len = doc_infos[sorted_indices[i]].1;
                let max_acceptable_len = (base_len as f32 * 1.2) as usize;
                
                let mut batch_end = i + 1;
                while batch_end < n_docs && batch_end < i + target_batch_size {
                    if doc_infos[sorted_indices[batch_end]].1 > max_acceptable_len {
                        break;
                    }
                    batch_end += 1;
                }
                
                let batch_size = batch_end - i;
                
                if batch_size == 1 {
                    // Single document
                    let idx = sorted_indices[i];
                    let (doc_idx, doc_len, doc_data) = &doc_infos[idx];
                    results[*doc_idx] = process_single_doc(q, doc_data, q_len, *doc_len, dim);
                } else if batch_size >= 32 {
                    // Large batch - worth the overhead of batched processing
                    // Check if all documents in batch have exactly the same length
                    let first_len = doc_infos[sorted_indices[i]].1;
                    let all_same_length = sorted_indices[i..batch_end]
                        .iter()
                        .all(|&idx| doc_infos[idx].1 == first_len);
                    
                    if all_same_length {
                        // Super optimized path - no padding needed!
                        let batch_results = BATCH_BUFFER.with(|buffer| {
                            let mut buffer = buffer.borrow_mut();
                            let required_size = batch_size * first_len * dim;
                            buffer.resize(required_size, 0.0);
                            
                            // Copy documents contiguously
                            for (batch_idx, &sorted_idx) in sorted_indices[i..batch_end].iter().enumerate() {
                                let (_, _, doc_data) = &doc_infos[sorted_idx];
                                let dst_offset = batch_idx * first_len * dim;
                                buffer[dst_offset..dst_offset + first_len * dim]
                                    .copy_from_slice(&doc_data[..first_len * dim]);
                            }
                            
                            // Process with no wasted computation
                            maxsim_fused_doc_tiles(
                                q, &buffer[..required_size], q_len, first_len, dim
                            )
                        });
                        
                        // Copy results back
                        for (batch_idx, &sorted_idx) in sorted_indices[i..batch_end].iter().enumerate() {
                            let (doc_idx, _, _) = &doc_infos[sorted_idx];
                            results[*doc_idx] = batch_results[batch_idx];
                        }
                    } else {
                        // Batch processing with padding
                        let max_len = sorted_indices[i..batch_end]
                            .iter()
                            .map(|&idx| doc_infos[idx].1)
                            .max()
                            .unwrap();
                        
                        let batch_results = BATCH_BUFFER.with(|buffer| {
                            let mut buffer = buffer.borrow_mut();
                            let required_size = batch_size * max_len * dim;
                            
                            // Resize buffer if needed
                            buffer.resize(required_size, 0.0);
                            
                            // Fill batch - only clear padding areas
                            for (batch_idx, &sorted_idx) in sorted_indices[i..batch_end].iter().enumerate() {
                                let (_, doc_len, doc_data) = &doc_infos[sorted_idx];
                                let src_size = doc_len * dim;
                                let dst_offset = batch_idx * max_len * dim;
                                
                                // Copy actual data
                                buffer[dst_offset..dst_offset + src_size]
                                    .copy_from_slice(&doc_data[..src_size]);
                                
                                // Clear only the padding area
                                if *doc_len < max_len {
                                    let padding_start = dst_offset + src_size;
                                    let padding_end = dst_offset + max_len * dim;
                                    buffer[padding_start..padding_end].fill(0.0);
                                }
                            }
                            
                            // Process batch with optimized kernel
                            maxsim_fused_doc_tiles(
                                q, &buffer[..required_size], q_len, max_len, dim
                            )
                        });
                        
                        // Copy results back to original positions
                        for (batch_idx, &sorted_idx) in sorted_indices[i..batch_end].iter().enumerate() {
                            let (doc_idx, _, _) = &doc_infos[sorted_idx];
                            results[*doc_idx] = batch_results[batch_idx];
                        }
                    }
                } else {
                    // Small batch - process with standard approach
                    // Batch processing with reused buffer
                    let max_len = sorted_indices[i..batch_end]
                        .iter()
                        .map(|&idx| doc_infos[idx].1)
                        .max()
                        .unwrap();
                    
                    let batch_results = BATCH_BUFFER.with(|buffer| {
                        let mut buffer = buffer.borrow_mut();
                        let required_size = batch_size * max_len * dim;
                        
                        // Resize buffer if needed
                        buffer.resize(required_size, 0.0);
                        
                        // Fill batch - only clear padding areas
                        for (batch_idx, &sorted_idx) in sorted_indices[i..batch_end].iter().enumerate() {
                            let (_, doc_len, doc_data) = &doc_infos[sorted_idx];
                            let src_size = doc_len * dim;
                            let dst_offset = batch_idx * max_len * dim;
                            
                            // Copy actual data
                            buffer[dst_offset..dst_offset + src_size]
                                .copy_from_slice(&doc_data[..src_size]);
                            
                            // Clear only the padding area
                            if *doc_len < max_len {
                                let padding_start = dst_offset + src_size;
                                let padding_end = dst_offset + max_len * dim;
                                buffer[padding_start..padding_end].fill(0.0);
                            }
                        }
                        
                        // Process batch with optimized kernel
                        maxsim_fused_doc_tiles(
                            q, &buffer[..required_size], q_len, max_len, dim
                        )
                    });
                    
                    // Copy results back to original positions
                    for (batch_idx, &sorted_idx) in sorted_indices[i..batch_end].iter().enumerate() {
                        let (doc_idx, _, _) = &doc_infos[sorted_idx];
                        results[*doc_idx] = batch_results[batch_idx];
                    }
                }
                
                i = batch_end;
            }
            
            results
        }
    }
}

// libxsmm -- the true magic (feature-gated)
#[cfg(feature = "use-libxsmm")]
mod libxsmm {
    use super::*;
    
    /// Clean libxsmm implementation
    pub fn maxsim_libxsmm_clean(
        q: &[f32],           // [q_len * dim]
        d: &[f32],           // [n_docs * d_len * dim]
        q_len: usize,
        d_len: usize,
        dim: usize,
    ) -> Vec<f32> {
        // Initialize libxsmm
        LIBXSMM_INIT.call_once(|| {
            unsafe { libxsmm_bindings::libxsmm_init(); }
        });

        let n_docs = d.len() / (d_len * dim);
        
        // Try to keep tiles in L2 cache  
        let block_size = 64;

        // Process documents in parallel
        (0..n_docs).into_par_iter().map(|doc_idx| {
            let doc_offset = doc_idx * d_len * dim;
            let doc_data = &d[doc_offset..doc_offset + d_len * dim];
            
            // max values for each query
            let mut max_vals = vec![f32::NEG_INFINITY; q_len];
            
            // Process document tokens in blocks
            for t in (0..d_len).step_by(block_size) {
                let actual_block_size = block_size.min(d_len - t);
                
                // Workspace for GEMM output
                let mut c = vec![0.0f32; q_len * actual_block_size];
                
                // Compute Q × D_block^T using libxsmm
                // For column-major BLAS: C = A*B computes C^T = B^T * A^T
                // We want C = Q * D^T, so we compute C^T = D * Q^T\
                // This was a complete mess to figure out.
                // But it works now and it's correct.
                unsafe {
                    libxsmm_bindings::xsmm_sgemm(
                        b'T',                               // transa: D block transposed
                        b'N',                               // transb: Q not transposed
                        actual_block_size as i32,           // M: cols of original C
                        q_len as i32,                       // N: rows of original C
                        dim as i32,                         // K: dimension
                        1.0,                                // alpha
                        doc_data.as_ptr().add(t * dim),     // A: doc block
                        dim as i32,                         // lda
                        q.as_ptr(),                         // B: queries
                        dim as i32,                         // ldb
                        0.0,                                // beta
                        c.as_mut_ptr(),                     // C: output
                        actual_block_size as i32,           // ldc
                    );
                }
                
                // Update max values - C is column-major from libxsmm
                // Since we computed C^T = D * Q^T, element C[qi,ti] is at qi * actual_block_size + ti
                for qi in 0..q_len {
                    for ti in 0..actual_block_size {
                        let idx = qi * actual_block_size + ti;
                        max_vals[qi] = max_vals[qi].max(c[idx]);
                    }
                }
            }
            
            // Sum all max values
            max_vals.iter().sum()
        }).collect()
    }
    
    /// Process variable-length documents with libxsmm
    pub fn maxsim_libxsmm_variable(
        q: &[f32],                                    // [q_len * dim]
        doc_infos: Vec<(usize, usize, &[f32])>,     // [(doc_idx, doc_len, doc_data)]
        q_len: usize,
        dim: usize,
    ) -> Vec<f32> {
        // Initialize libxsmm
        LIBXSMM_INIT.call_once(|| {
            unsafe { libxsmm_bindings::libxsmm_init(); }
        });
        
        let n_docs = doc_infos.len();
        
        // Process documents in parallel, each with its actual length
        let mut results = vec![0.0f32; n_docs];
        let results_vec: Vec<(usize, f32)> = doc_infos.into_par_iter().map(|(doc_idx, doc_len, doc_data)| {
            // max values for each query
            let mut max_vals = vec![f32::NEG_INFINITY; q_len];
            
            // Try to keep tiles in L2 cache  
            let block_size = 64;
            
            // Process document tokens in blocks
            for t in (0..doc_len).step_by(block_size) {
                let actual_block_size = block_size.min(doc_len - t);
                
                // Workspace for GEMM output
                let mut c = vec![0.0f32; q_len * actual_block_size];
                
                unsafe {
                    libxsmm_bindings::xsmm_sgemm(
                        b'T',                               // transa: D block transposed
                        b'N',                               // transb: Q not transposed
                        actual_block_size as i32,           // M: cols of original C
                        q_len as i32,                       // N: rows of original C
                        dim as i32,                         // K: dimension
                        1.0,                                // alpha
                        doc_data.as_ptr().add(t * dim),     // A: doc block
                        dim as i32,                         // lda
                        q.as_ptr(),                         // B: queries
                        dim as i32,                         // ldb
                        0.0,                                // beta
                        c.as_mut_ptr(),                     // C: output
                        actual_block_size as i32,           // ldc
                    );
                }
                
                // Update max values
                for qi in 0..q_len {
                    for ti in 0..actual_block_size {
                        let idx = qi * actual_block_size + ti;
                        max_vals[qi] = max_vals[qi].max(c[idx]);
                    }
                }
            }
            
            // Sum all max values
            (doc_idx, max_vals.iter().sum())
        }).collect();
        
        // Place results in correct order
        for (doc_idx, score) in results_vec {
            results[doc_idx] = score;
        }
        
        results
    }
}

// from here onwards, we're back in the safety of python land.
#[pymodule]
fn maxsim_cpu(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(maxsim_scores, m)?)?;
    m.add_function(wrap_pyfunction!(maxsim_scores_variable, m)?)?;
    Ok(())
}

#[pyfunction]
fn maxsim_scores<'py>(
    py: Python<'py>,
    query: PyReadonlyArray2<f32>,     // [q_len, dim]
    docs:  PyReadonlyArray3<f32>,     // [n_docs, d_len, dim]
) -> PyResult<&'py PyArray1<f32>> {
    // shape checks
    let q_shape = query.shape();
    let d_shape = docs.shape();
    if q_shape[1] != d_shape[2] {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("Dimension mismatch: query dim {} vs docs dim {}", q_shape[1], d_shape[2]),
        ));
    }

    let q_slice = query.as_slice()?;
    let d_slice = docs.as_slice()?;

    // release the GIL for computation
    let scores = py.allow_threads(|| {
        algorithm::maxsim_ultra_adaptive(
            q_slice,
            d_slice,
            q_shape[0],
            d_shape[1],
            q_shape[1],
        )
    });

    Ok(PyArray1::from_vec(py, scores))
}

#[pyfunction]
fn maxsim_scores_variable<'py>(
    py: Python<'py>,
    query: PyReadonlyArray2<f32>,           // [q_len, dim]
    docs: Vec<PyReadonlyArray2<f32>>,      // List of [d_len_i, dim] arrays
) -> PyResult<&'py PyArray1<f32>> {
    // Validate inputs
    let q_shape = query.shape();
    let q_dim = q_shape[1];
    
    // Check dimension consistency and collect document info
    let mut doc_infos: Vec<(usize, usize, &[f32])> = Vec::with_capacity(docs.len());
    for (i, doc) in docs.iter().enumerate() {
        let d_shape = doc.shape();
        if d_shape[1] != q_dim {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Dimension mismatch at doc {}: query dim {} vs doc dim {}", 
                        i, q_dim, d_shape[1]),
            ));
        }
        doc_infos.push((i, d_shape[0], doc.as_slice()?));
    }
    
    let q_slice = query.as_slice()?;
    let q_len = q_shape[0];
    
    // Release the GIL for computation
    let scores = py.allow_threads(|| {
        algorithm::maxsim_variable_length(
            q_slice,
            doc_infos,
            q_len,
            q_dim,
        )
    });
    
    Ok(PyArray1::from_vec(py, scores))
}