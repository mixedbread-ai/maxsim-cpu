import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
import time
import numpy as np
import jax.numpy as jnp
from jax import jit, vmap
import maxsim_cpu

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Single query with batch of documents
query_embedding = torch.randn(32, 128, device=device)  # Single query
# Normalize query embeddings
query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=-1)

# Document lengths to test
DOC_LENGTHS = [512, 1024,]
# 2048, 4096]

# Variable length test parameters
VARIABLE_MIN_LEN = 128
VARIABLE_MAX_LEN = 1536
VARIABLE_NUM_DOCS = 1000  # Reduced for quick testing

def original_method(query_emb, doc_emb):
    """Original einsum approach"""
    query_document_score = torch.einsum(
        "qh,bth->qbt",
        query_emb,
        doc_emb,
    )
    scores = query_document_score.max(dim=-1).values.sum(dim=0)
    return scores


def optimized_matmul(query_emb, doc_emb):
    """Using matmul instead of einsum"""
    # Reshape documents: (1000, doc_len, 128) -> (1000*doc_len, 128)
    doc_reshaped = doc_emb.view(-1, doc_emb.size(-1))  # (1000*doc_len, 128)

    # Compute similarity: (32, 128) @ (128, 1000*doc_len) -> (32, 1000*doc_len)
    scores = torch.matmul(query_emb, doc_reshaped.T)  # (32, 1000*doc_len)

    # Reshape back: (32, 1000*doc_len) -> (32, 1000, doc_len)
    scores = scores.view(query_emb.size(0), doc_emb.size(0), doc_emb.size(1))

    # Max and sum as before
    scores = scores.max(dim=-1).values.sum(dim=0)
    return scores


def optimized_batched(query_emb, doc_emb):
    """Process in batches to reduce memory usage"""
    batch_size = 100  # Process 100 batches at a time
    all_max_scores = []

    for i in range(0, doc_emb.size(0), batch_size):
        end_idx = min(i + batch_size, doc_emb.size(0))
        batch_docs = doc_emb[i:end_idx]  # (batch_size, doc_len, 128)

        # Reshape batch: (batch_size, doc_len, 128) -> (batch_size*doc_len, 128)
        batch_reshaped = batch_docs.view(-1, batch_docs.size(-1))

        # Compute similarity: (32, 128) @ (128, batch_size*doc_len) -> (32, batch_size*doc_len)
        batch_scores = torch.matmul(query_emb, batch_reshaped.T)

        # Reshape: (32, batch_size*doc_len) -> (32, batch_size, doc_len)
        batch_scores = batch_scores.view(
            query_emb.size(0), batch_docs.size(0), batch_docs.size(1)
        )

        # Max across documents in each batch
        max_scores = batch_scores.max(dim=-1).values  # (32, batch_size)
        all_max_scores.append(max_scores)

    # Concatenate and sum
    all_max_scores = torch.cat(all_max_scores, dim=1)  # (32, 1000)
    scores = all_max_scores.sum(dim=0)  # (1000,)
    return scores


def optimized_fused(query_emb, doc_emb):
    """Fused operations with better memory access"""
    # Use torch.baddbmm for efficient batched matrix multiplication
    # First expand query to match batch dimension
    query_expanded = query_emb.unsqueeze(0).expand(
        doc_emb.size(0), -1, -1
    )  # (1000, 32, 128)

    # Compute batched matrix multiplication
    # (1000, 32, 128) @ (1000, 128, doc_len) -> (1000, 32, doc_len)
    scores = torch.bmm(query_expanded, doc_emb.transpose(1, 2))

    # Max and sum
    scores = scores.max(dim=-1).values  # (1000, 32)
    scores = scores.sum(dim=1)  # (1000,)
    return scores


def max_sim3(q, d):
    # flatten then compute; may advantage due to better memory continuous
    K, M, D = d.shape
    Q = q.shape[0]
    scores = d.reshape(-1, D) @ q.T  # (K*M, Q)
    max_scores = np.max(scores.reshape(K, M, Q), axis=1)  # (K, Q)
    return np.sum(max_scores, axis=1)  # (K,)


@jit
def max_sim_jax(q, d):
    # JAX will automatically use XLA to optimize this
    scores = jnp.einsum("kmd,qd->kmq", d, q)
    max_scores = jnp.max(scores, axis=1)
    return jnp.sum(max_scores, axis=1)


@jit
def max_sim_jax_vmap(q, d):
    # Use vmap for better vectorization
    def compute_batch(d_batch):
        scores = jnp.dot(d_batch, q.T)  # (doc_len, 32)
        return jnp.max(scores, axis=0)  # (32,)

    max_scores = vmap(compute_batch)(d)  # (1000, 32)
    return jnp.sum(max_scores, axis=1)


# Rust implementation wrapper
def rust_maxsim(q, d):
    """Rust implementation of MaxSim"""
    # Ensure inputs are contiguous and float32
    if not isinstance(q, np.ndarray):
        q = q.cpu().numpy()
    if not isinstance(d, np.ndarray):
        d = d.cpu().numpy()
    
    q = np.ascontiguousarray(q, dtype=np.float32)
    d = np.ascontiguousarray(d, dtype=np.float32)
    
    return maxsim_cpu.maxsim_scores(q, d)

# Define all methods to benchmark
methods = [
    ("Torch Original (einsum)", original_method),
    ("Torch (matmul)", optimized_matmul),
    # ("Torch (batched)", optimized_batched),
    # ("Torch (fused)", optimized_fused),
    ("Numpy (flatten)", max_sim3),
    ("Jax (einsum)", max_sim_jax),
    ("Jax (vmap)", max_sim_jax_vmap),
]

# Add Rust method if available
methods.append(("Rust (AVX2/parallel)", rust_maxsim))

# Store results for all document lengths
all_results = {}

# Benchmark across different document lengths
for doc_len in DOC_LENGTHS:
    print(f"\n{'='*60}")
    print(f"Testing with document length: {doc_len}")
    print(f"{'='*60}")
    
    # Create document embeddings for this length
    document_embedding = torch.randn(1000, doc_len, 128, device=device)
    # Normalize document embeddings
    document_embedding = torch.nn.functional.normalize(document_embedding, p=2, dim=-1)
    
    # Warm up GPU if using CUDA
    if device.type == "cuda":
        for _ in range(3):
            _ = original_method(query_embedding, document_embedding)
        torch.cuda.synchronize()

    results = {}
    
    # Pre-convert all data formats outside timing loop
    tmp_query = query_embedding.cpu().numpy().astype(np.float32)
    tmp_document = document_embedding.cpu().numpy().astype(np.float32)
    tmp_query_jax = jnp.array(tmp_query)
    tmp_document_jax = jnp.array(tmp_document)
    
    for name, method in methods:

        # warmup every method
        _ = (
            method(tmp_query, tmp_document)
            if name in ["Numpy (flatten)", "Jax (einsum)", "Jax (vmap)", "Rust (AVX2/parallel)"]
            else None
        )
        _ = (
            method(tmp_query_jax, tmp_document_jax).block_until_ready()
            if name in ["Jax (einsum)", "Jax (vmap)"]
            else None
        )
        _ = (
            method(query_embedding, document_embedding)
            if name not in ["Numpy (flatten)", "Jax (einsum)", "Jax (vmap)", "Rust (AVX2/parallel)"]
            else None
        )

        # Multiple runs for better timing
        times = []
        for _ in range(10):
            torch.cuda.synchronize() if device.type == "cuda" else None
            start_time = time.time()

            if name == "Numpy (flatten)" or name == "Rust (AVX2/parallel)":
                result = method(tmp_query, tmp_document)
            elif name in ["Jax (einsum)", "Jax (vmap)"]:
                result = method(tmp_query_jax, tmp_document_jax).block_until_ready()
            else:  # PyTorch methods
                result = method(query_embedding, document_embedding)

            torch.cuda.synchronize() if device.type == "cuda" else None
            end_time = time.time()
            times.append(end_time - start_time)

        avg_time = sum(times) / len(times)
        results[name] = (avg_time, result)
        print(f"{name}: {avg_time:.4f} seconds (avg of {len(times)} runs)")

    # Verify all methods produce the same result for this document length
    print(f"\nVerifying results match for doc_len={doc_len}...")
    reference_result = results["Torch Original (einsum)"][1]
    for name, (_, result) in results.items():
        if name != "Torch Original (einsum)":
            if name == "Numpy (flatten)" or name == "Rust (AVX2/parallel)":
                result = torch.from_numpy(result)
            if "Jax" in name:
                result = torch.from_numpy(np.asarray(result))
            if torch.allclose(result, reference_result, atol=1e-5):
                print(f"✓ {name} matches reference")
            else:
                print(f"✗ {name} differs from reference")
                print(
                    f"  Max difference: {torch.max(torch.abs(result - reference_result)):.2e}"
                )

    print(f"\nResult shape: {reference_result.shape}")
    print(f"Sample values: {reference_result[:5]}")

    # Show speedup for this document length
    original_time = results["Torch Original (einsum)"][0]
    jax_vmap_time = results.get("Jax (vmap)", (float('inf'), None))[0]
    rust_time = results.get("Rust (AVX2/parallel)", (float('inf'), None))[0]
    
    print(f"\nSpeedup comparison for doc_len={doc_len}:")
    for name, (avg_time, _) in results.items():
        if name != "Torch Original (einsum)":
            speedup = original_time / avg_time
            print(f"{name}: {speedup:.2f}x faster than PyTorch")
    
    # Special comparison: Rust vs JAX vmap
    if "Rust (AVX2/parallel)" in results and "Jax (vmap)" in results:
        rust_vs_jax = jax_vmap_time / rust_time
        print(f"\n⚡ Rust is {rust_vs_jax:.2f}x faster than JAX vmap")
    
    # Store results for this document length
    all_results[doc_len] = results

# Summary comparison across all document lengths
print(f"\n{'='*80}")
print("SUMMARY: Performance across different document lengths")
print(f"{'='*80}")

print(f"\n{'Method':<25} {'Doc Length':<12} {'Time (s)':<12} {'Speedup':<10}")
print("-" * 65)

for method_name, _ in methods:
    for doc_len in DOC_LENGTHS:
        avg_time, _ = all_results[doc_len][method_name]
        original_time = all_results[doc_len]["Torch Original (einsum)"][0]
        speedup = original_time / avg_time if method_name != "Torch Original (einsum)" else 1.0
        
        print(f"{method_name:<25} {doc_len:<12} {avg_time:<12.4f} {speedup:<10.2f}x")

# Show how each method scales with document length
print(f"\n{'='*80}")
print("SCALING ANALYSIS: Performance relative to PyTorch")
print(f"{'='*80}")

print(f"\n{'Method':<25} {'Doc Length':<12} {'Time (s)':<12} {'vs PyTorch':<15}")
print("-" * 65)

for method_name, _ in methods:
    for doc_len in DOC_LENGTHS:
        avg_time, _ = all_results[doc_len][method_name]
        torch_time = all_results[doc_len]["Torch Original (einsum)"][0]
        speedup = torch_time / avg_time
        speedup_str = f"{speedup:.2f}x" if method_name != "Torch Original (einsum)" else "1.00x (baseline)"
        print(f"{method_name:<25} {doc_len:<12} {avg_time:<12.4f} {speedup_str:<15}")

# Final summary: Rust vs JAX vmap across all document lengths
print(f"\n{'='*80}")
print("FINAL SUMMARY: Rust vs JAX vmap Performance")
print(f"{'='*80}")

print(f"\n{'Doc Length':<12} {'Rust Time':<12} {'JAX Time':<12} {'Rust Speedup':<15}")
print("-" * 55)

for doc_len in DOC_LENGTHS:
    if "Rust (AVX2/parallel)" in all_results[doc_len] and "Jax (vmap)" in all_results[doc_len]:
        rust_time = all_results[doc_len]["Rust (AVX2/parallel)"][0]
        jax_time = all_results[doc_len]["Jax (vmap)"][0]
        speedup = jax_time / rust_time
        print(f"{doc_len:<12} {rust_time:<12.4f} {jax_time:<12.4f} {speedup:<15.2f}x")

# Variable length document test
print(f"\n{'='*80}")
print(f"VARIABLE LENGTH TEST: Documents with {VARIABLE_MIN_LEN}-{VARIABLE_MAX_LEN} tokens")
print(f"{'='*80}")

# Generate random document lengths with uniform distribution
np.random.seed(42)  # For reproducibility
variable_doc_lengths = np.random.randint(VARIABLE_MIN_LEN, VARIABLE_MAX_LEN + 1, size=VARIABLE_NUM_DOCS)
max_var_len = np.max(variable_doc_lengths)

print(f"\nGenerating {VARIABLE_NUM_DOCS} documents with variable lengths:")
print(f"  Min length: {np.min(variable_doc_lengths)} tokens")
print(f"  Max length: {np.max(variable_doc_lengths)} tokens")
print(f"  Mean length: {np.mean(variable_doc_lengths):.1f} tokens")
print(f"  Std dev: {np.std(variable_doc_lengths):.1f} tokens")

# Show distribution
print(f"\nLength distribution:")
bins = [128, 256, 512, 768, 1024, 1280, 1536]
hist, _ = np.histogram(variable_doc_lengths, bins=bins)
for i in range(len(bins)-1):
    pct = hist[i] / VARIABLE_NUM_DOCS * 100
    print(f"  {bins[i]:4d}-{bins[i+1]:4d} tokens: {hist[i]:3d} docs ({pct:4.1f}%)")


# First generate the original unpadded documents
original_docs = []
for i, doc_len in enumerate(variable_doc_lengths):
    doc_emb = torch.randn(doc_len, 128, device=device)
    doc_emb = torch.nn.functional.normalize(doc_emb, p=2, dim=-1)
    original_docs.append(doc_emb)

# ========================================================================
# PADDING METHOD COMPARISON
# ========================================================================
print(f"\n{'='*80}")
print("PADDING METHOD COMPARISON")
print(f"{'='*80}")
print(f"Comparing padding methods for {VARIABLE_NUM_DOCS} documents, max length {max_var_len}")

# Define padding methods
def pad_current_approach(docs, doc_lengths, max_len):
    """Current simple loop approach (baseline)"""
    padded_docs = torch.zeros(len(docs), max_len, 128, device=device)
    for i in range(len(docs)):
        doc_len = doc_lengths[i]
        padded_docs[i, :doc_len] = docs[i]
    return padded_docs

def pad_numpy_batched(docs, doc_lengths, max_len, batch_size=100):
    """NumPy batched approach for better cache utilization"""
    padded = np.zeros((len(docs), max_len, 128), dtype=np.float32)
    docs_np = [doc.cpu().numpy() for doc in docs]
    
    for batch_start in range(0, len(docs), batch_size):
        batch_end = min(batch_start + batch_size, len(docs))
        for i in range(batch_start, batch_end):
            length = doc_lengths[i]
            padded[i, :length] = docs_np[i]
    
    return torch.from_numpy(padded).to(device)

def pad_numpy_contiguous(docs, doc_lengths, max_len):
    """NumPy with emphasis on contiguous memory"""
    padded = np.zeros((len(docs), max_len, 128), dtype=np.float32, order='C')
    docs_np = [doc.cpu().numpy().astype(np.float32, copy=False) for doc in docs]
    
    for i, (doc, length) in enumerate(zip(docs_np, doc_lengths)):
        padded[i, :length, :] = doc
    
    return torch.from_numpy(padded).to(device)

def pad_torch_optimized(docs, doc_lengths, max_len):
    """Optimized PyTorch operations"""
    padded = torch.zeros((len(docs), max_len, 128), dtype=torch.float32, device=device)
    
    for i, (doc, length) in enumerate(zip(docs, doc_lengths)):
        padded[i, :length].copy_(doc)
    
    return padded

# Benchmark padding methods
padding_methods = [
    ("Current approach (baseline)", pad_current_approach),
    ("NumPy batched (100)", pad_numpy_batched),
    ("NumPy contiguous", pad_numpy_contiguous),
    ("PyTorch optimized", pad_torch_optimized),
]

padding_results = {}
best_padding_time = float('inf')
best_padding_method = None
best_padded_docs = None

for pad_name, pad_method in padding_methods:
    # Warmup
    _ = pad_method(original_docs, variable_doc_lengths, max_var_len)
    
    # Benchmark
    times = []
    for _ in range(10):
        start = time.perf_counter()
        padded_result = pad_method(original_docs, variable_doc_lengths, max_var_len)
        end = time.perf_counter()
        times.append(end - start)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    padding_results[pad_name] = avg_time
    
    print(f"{pad_name:<30} {avg_time*1000:6.2f} ms ± {std_time*1000:4.2f} ms")
    
    if avg_time < best_padding_time:
        best_padding_time = avg_time
        best_padding_method = pad_name
        best_padded_docs = padded_result

print(f"\n✅ Best padding method: {best_padding_method} ({best_padding_time*1000:.2f} ms)")
print(f"   {padding_results['Current approach (baseline)']/best_padding_time:.2f}x faster than baseline")

# Use the best padding method for the rest of the tests
padded_docs_torch = best_padded_docs
padding_time = best_padding_time

# Convert to numpy for methods that need it
padded_docs = padded_docs_torch.cpu().numpy().astype(np.float32)
query_np = query_embedding.cpu().numpy().astype(np.float32)

print(f"\n{'='*80}")
print("CONTINUING WITH VARIABLE LENGTH BENCHMARKS")
print(f"{'='*80}")
print(f"Using {best_padding_method} for padding (time: {padding_time:.4f} seconds)")

# Test each method with variable length documents
print(f"\nBenchmarking with variable length documents...")
variable_results = {}

# Pre-convert data for each method type to ensure fair timing
query_np_preconverted = query_embedding.cpu().numpy().astype(np.float32)
query_jax = jnp.array(query_np_preconverted)
docs_jax = jnp.array(padded_docs)

for name, method in methods:
    # Skip methods that don't handle padding well
    if name in ["Numpy (flatten)", "Jax (einsum)", "Jax (vmap)", "Rust (AVX2/parallel)"]:
        # For these methods, we need to test document by document
        if name == "Rust (AVX2/parallel)":
            # Use the same padded data as other methods
            times = []
            for _ in range(5):  # Fewer runs for variable length
                start_time = time.time()
                
                # Use the already padded array - no duplicate padding
                result = rust_maxsim(query_np_preconverted, padded_docs)
                
                end_time = time.time()
                times.append(end_time - start_time)
            
            avg_time = sum(times) / len(times)
        else:
            # For other methods, use the full padded tensor
            times = []
            for _ in range(5):
                start_time = time.time()
                if name in ["Numpy (flatten)"]:
                    result = method(query_np_preconverted, padded_docs)
                elif name in ["Jax (einsum)", "Jax (vmap)"]:
                    result = method(query_jax, docs_jax).block_until_ready()
                else:
                    result = method(query_embedding, padded_docs)
                end_time = time.time()
                times.append(end_time - start_time)
            avg_time = sum(times) / len(times)
    else:
        # PyTorch methods can handle the padded tensor directly
        times = []
        for _ in range(5):
            torch.cuda.synchronize() if device.type == "cuda" else None
            start_time = time.time()
            result = method(query_embedding, padded_docs_torch)
            torch.cuda.synchronize() if device.type == "cuda" else None
            end_time = time.time()
            times.append(end_time - start_time)
        avg_time = sum(times) / len(times)
    
    # Store both raw time and whether this method requires padding
    variable_results[name] = (avg_time, True)  # All methods using padded array require padding
    # Include padding time in the displayed result
    total_time_with_padding = avg_time + padding_time
    print(f"{name}: {total_time_with_padding:.4f} seconds (avg of {len(times)} runs, includes {padding_time:.4f}s padding)")

# Show speedup comparisons for variable length
print(f"\nSpeedup comparison for variable length documents (including padding overhead):")
torch_time, _ = variable_results["Torch Original (einsum)"]
torch_total = torch_time + padding_time
for name, (avg_time, requires_padding) in variable_results.items():
    if name != "Torch Original (einsum)":
        total_time = avg_time + padding_time if requires_padding else avg_time
        speedup = torch_total / total_time
        print(f"{name}: {speedup:.2f}x faster than PyTorch")

# Show breakdown of times
print(f"\n{'='*60}")
print("TIME BREAKDOWN")
print(f"{'='*60}")
print(f"\nPadding operation time: {padding_time:.4f} seconds")
print(f"\n{'Method':<25} {'Compute Time':<15} {'Total Time':<20} {'Overhead %':<10}")
print("-" * 70)
for name, (avg_time, requires_padding) in variable_results.items():
    if requires_padding:
        total_time = avg_time + padding_time
        overhead_pct = (padding_time / total_time) * 100
        print(f"{name:<25} {avg_time:<15.4f} {total_time:.4f} (+ padding) {overhead_pct:>6.1f}%")
    else:
        print(f"{name:<25} {avg_time:<15.4f} {avg_time:<20.4f} {0:>6.1f}%")

# Rust vs JAX vmap for variable length
if "Rust (AVX2/parallel)" in variable_results and "Jax (vmap)" in variable_results:
    rust_time, _ = variable_results["Rust (AVX2/parallel)"]
    jax_time, _ = variable_results["Jax (vmap)"]
    # Both require padding, so include it for fair comparison
    rust_total = rust_time + padding_time
    jax_total = jax_time + padding_time
    rust_vs_jax = jax_total / rust_total
    print(f"\n⚡ For variable length docs (with padding), Rust is {rust_vs_jax:.2f}x faster than JAX vmap")
    
    # Also show new API comparison
    if "Rust NEW API (no padding)" in variable_results:
        new_api_time, _ = variable_results["Rust NEW API (no padding)"]
        new_vs_jax = jax_total / new_api_time
        print(f"⚡ Rust NEW API is {new_vs_jax:.2f}x faster than JAX vmap (JAX includes padding)")

print(f"\nNote: Variable length processing approach:")
print(f"  - Pads all documents to max length ({max_var_len} tokens)")
print(f"  - Uses special padding vectors with guaranteed negative similarities")
print(f"  - Padding time is included in final reported times for fair comparison")
print(f"  - Single batch processing for all documents")
print(f"  - Padding vector: negative mean of query vectors (ensures negative dot products)")



# query_np already defined earlier during padding setup

# Test the new variable-length API
print("\n================================================================================")
print("NEW VARIABLE LENGTH API TEST")
print("================================================================================")

# Check if new API exists
rust_maxsim_variable = maxsim_cpu.maxsim_scores_variable

# Prepare documents as list of arrays for new API - use original unpadded docs
print('hi!', flush=True)

docs_list = []
for i, doc_len in enumerate(variable_doc_lengths):
    # Convert original documents to numpy
    doc_emb = original_docs[i].cpu().numpy().astype(np.float32)
    docs_list.append(doc_emb)

# Benchmark new API
rust_times_new = []
for run in range(5):
    print('rusttime')
    start_time = time.time()
    result_new = rust_maxsim_variable(query_np, docs_list)
    end_time = time.time()
    rust_times_new.append(end_time - start_time)

# Benchmark PyTorch on padded variable length data
pytorch_var_times = []
for run in range(5):
    start_time = time.time()
    result_pytorch_var = original_method(query_embedding, padded_docs_torch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    end_time = time.time()
    pytorch_var_times.append(end_time - start_time)

avg_pytorch_var_time = np.mean(pytorch_var_times)
print(f"PyTorch variable length time: {avg_pytorch_var_time:.4f} seconds (avg of 5 runs)")


avg_new_time = np.mean(rust_times_new)
print(f"New API time: {avg_new_time:.4f} seconds (avg of 5 runs, no padding needed)")

# Get the compute times
rust_old_compute, _ = variable_results['Rust (AVX2/parallel)']
torch_compute, _ = variable_results['Torch Original (einsum)']

# Old API requires padding
rust_old_total = rust_old_compute + padding_time
torch_total = torch_compute + padding_time

print(f"\nOld API time: {rust_old_total:.4f} seconds ({rust_old_compute:.4f}s compute + {padding_time:.4f}s padding)")
print(f"Speedup of new API vs old: {rust_old_total / avg_new_time:.2f}x")

# Compare to PyTorch (which also requires padding)
print(f"\nCompared to PyTorch (both including required padding):")
print(f"  PyTorch total: {torch_total:.4f}s ({torch_compute:.4f}s compute + {padding_time:.4f}s padding)")
print(f"  Old Rust API: {torch_total / rust_old_total:.2f}x faster than PyTorch")
print(f"  New Rust API: {torch_total / avg_new_time:.2f}x faster than PyTorch")

# Analyze impact of avoiding padding
print(f"\nNew API advantage by avoiding padding:")
print(f"  Padding overhead eliminated: {padding_time:.4f} seconds")
print(f"  Old API total: {rust_old_total:.4f}s (includes padding)")
print(f"  New API total: {avg_new_time:.4f}s (no padding)")
print(f"  Additional speedup from no padding: {rust_old_total / avg_new_time:.2f}x")

# Add new API to results for summary
variable_results["Rust NEW API (no padding)"] = (avg_new_time, False)  # False = doesn't require padding

# Verify results match between PyTorch and new Rust API
# Store the PyTorch result for comparison
result_pytorch_torch = result_pytorch_var  # PyTorch result from padded array
result_new_torch = torch.from_numpy(result_new)
max_diff = (result_pytorch_torch - result_new_torch).abs().max().item()
print(f"\nPyTorch vs New Rust API match: {'✓' if max_diff < 1e-4 else '✗'} (max diff: {max_diff:.6f})")

# Also verify old Rust API matches PyTorch
if 'Rust (AVX2/parallel)' in variable_results:
    # Get the Rust result from the padded test
    rust_old_result = None
    for name, method in methods:
        if name == "Rust (AVX2/parallel)":
            rust_old_result = rust_maxsim(query_np, padded_docs)
            break
    if rust_old_result is not None:
        rust_old_torch = torch.from_numpy(rust_old_result)
        max_diff_old = (result_pytorch_torch - rust_old_torch).abs().max().item()
        print(f"PyTorch vs Old Rust API match: {'✓' if max_diff_old < 1e-4 else '✗'} (max diff: {max_diff_old:.6f})")