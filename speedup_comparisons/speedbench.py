import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
import time
import numpy as np
import jax.numpy as jnp
from jax import jit, vmap
import maxsim_cpu
import matplotlib.pyplot as plt
import subprocess
import datetime
import platform
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='MaxSim CPU Benchmark')
parser.add_argument('--cores', default=32, help='Number of cores (for labeling only) or "mac" for Mac benchmarks')
parser.add_argument('--iterations', type=int, default=50, help='Number of iterations')
args = parser.parse_args()

# Handle cores argument
if args.cores == 'mac':
    is_mac_benchmark = True
    cores_display = None
else:
    is_mac_benchmark = False
    cores_display = int(args.cores)

# Get CPU information
def get_cpu_info():
    # Check if we're on macOS
    if platform.system() == 'Darwin':
        try:
            # Use sysctl to get the chip name on macOS
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                    capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except:
            pass
        
        # Fallback: try to determine Apple Silicon model
        try:
            # Check if it's Apple Silicon
            result = subprocess.run(['uname', '-m'], capture_output=True, text=True)
            if result.returncode == 0 and 'arm64' in result.stdout:
                # Try to get more specific chip info
                system_profiler = subprocess.run(['system_profiler', 'SPHardwareDataType'], 
                                                 capture_output=True, text=True)
                if system_profiler.returncode == 0:
                    for line in system_profiler.stdout.split('\n'):
                        if 'Chip:' in line:
                            return line.split(':', 1)[1].strip()
                        elif 'Processor Name:' in line:
                            return line.split(':', 1)[1].strip()
                
                # If we can't get specific info, at least indicate it's Apple Silicon
                return "Apple Silicon"
        except:
            pass
    
    # Original Linux code
    try:
        result = subprocess.run(['lscpu'], capture_output=True, text=True)
        lscpu_output = result.stdout
        
        cpu_name = None
        
        for line in lscpu_output.split('\n'):
            if 'Model name:' in line:
                cpu_name = line.split(':', 1)[1].strip()
                break
        
        return cpu_name or "Unknown CPU"
    except:
        return platform.processor() or "Unknown CPU"

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Get CPU info
cpu_name = get_cpu_info()
print(f"CPU: {cpu_name}")
if not is_mac_benchmark:
    print(f"Running with {cores_display} cores (set by taskset)")

# Single query with batch of documents
query_embedding = torch.randn(32, 128, device=device)
query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=-1)

# Document lengths to test
DOC_LENGTHS = [512, 1024, 2048]

# Number of iterations for timing
ITERATIONS = args.iterations

# Number of documents in batch
NUM_DOCS = 1000

# Variable length test parameters
VARIABLE_MIN_LEN = 128
VARIABLE_MAX_LEN = 1536
VARIABLE_NUM_DOCS = 1000

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
    doc_reshaped = doc_emb.view(-1, doc_emb.size(-1))
    scores = torch.matmul(query_emb, doc_reshaped.T)
    scores = scores.view(query_emb.size(0), doc_emb.size(0), doc_emb.size(1))
    scores = scores.max(dim=-1).values.sum(dim=0)
    return scores


def max_sim3(q, d):
    K, M, D = d.shape
    Q = q.shape[0]
    scores = d.reshape(-1, D) @ q.T
    max_scores = np.max(scores.reshape(K, M, Q), axis=1)
    return np.sum(max_scores, axis=1)


@jit
def max_sim_jax(q, d):
    scores = jnp.einsum("kmd,qd->kmq", d, q)
    max_scores = jnp.max(scores, axis=1)
    return jnp.sum(max_scores, axis=1)


@jit
def max_sim_jax_vmap(q, d):
    def compute_batch(d_batch):
        scores = jnp.dot(d_batch, q.T)
        return jnp.max(scores, axis=0)

    max_scores = vmap(compute_batch)(d)
    return jnp.sum(max_scores, axis=1)


def rust_maxsim(q, d):
    """Rust implementation of MaxSim"""
    if not isinstance(q, np.ndarray):
        q = q.cpu().numpy()
    if not isinstance(d, np.ndarray):
        d = d.cpu().numpy()
    
    q = np.ascontiguousarray(q, dtype=np.float32)
    d = np.ascontiguousarray(d, dtype=np.float32)
    
    return maxsim_cpu.maxsim_scores(q, d)

# Define all methods to benchmark
methods = [
    ("PyTorch (baseline)", original_method),
    ("PyTorch (matmul)", optimized_matmul),
    ("NumPy", max_sim3),
    ("JAX", max_sim_jax),
    ("JAX (vmap)", max_sim_jax_vmap),
    ("maxsim-cpu (Rust)", rust_maxsim),
]

def run_variable_length_benchmark(query_emb, query_np):
    """Run variable length benchmark - real world scenario with padding overhead"""
    print(f"\n{'='*80}")
    print(f"VARIABLE LENGTH TEST: Documents with {VARIABLE_MIN_LEN}-{VARIABLE_MAX_LEN} tokens")
    print(f"Real-world scenario: Padding overhead is counted as real cost")
    print(f"{'='*80}")
    
    # Generate random document lengths
    np.random.seed(42)
    variable_doc_lengths = np.random.randint(VARIABLE_MIN_LEN, VARIABLE_MAX_LEN + 1, size=VARIABLE_NUM_DOCS)
    max_var_len = np.max(variable_doc_lengths)
    avg_len = np.mean(variable_doc_lengths)
    
    print(f"\nGenerating {VARIABLE_NUM_DOCS} documents with variable lengths:")
    print(f"  Min length: {np.min(variable_doc_lengths)} tokens")
    print(f"  Max length: {max_var_len} tokens")
    print(f"  Mean length: {avg_len:.1f} tokens")
    print(f"  Padding overhead: {(max_var_len/avg_len - 1)*100:.0f}% wasted computation for padded methods")
    
    # Generate original documents
    original_docs = []
    for i, doc_len in enumerate(variable_doc_lengths):
        doc_emb = torch.randn(doc_len, 128, device=device)
        doc_emb = torch.nn.functional.normalize(doc_emb, p=2, dim=-1)
        original_docs.append(doc_emb)
    
    results = {}
    
    # Test Rust NEW API (no padding needed)
    if hasattr(maxsim_cpu, 'maxsim_scores_variable'):
        print("\nTesting Rust Variable API (no padding)...")
        docs_list = [doc.cpu().numpy().astype(np.float32) for doc in original_docs]
        
        times = []
        for _ in range(min(10, ITERATIONS)):
            start_time = time.time()
            result_rust = maxsim_cpu.maxsim_scores_variable(query_np, docs_list)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        docs_per_sec = VARIABLE_NUM_DOCS / avg_time
        results["Rust Variable API"] = {
            'avg_time': avg_time,
            'docs_per_sec': docs_per_sec,
            'needs_padding': False
        }
        print(f"Rust Variable API: {avg_time:.4f}s ({docs_per_sec:.0f} docs/sec)")
    
    # Now test methods that require padding
    print("\nPadding documents for other methods...")
    start_pad = time.time()
    padded_docs = torch.zeros(len(original_docs), max_var_len, 128, device=device)
    for i in range(len(original_docs)):
        doc_len = variable_doc_lengths[i]
        padded_docs[i, :doc_len] = original_docs[i]
    padding_time = time.time() - start_pad
    print(f"Padding time: {padding_time:.4f}s")
    
    # Convert padded data for different methods
    padded_np = padded_docs.cpu().numpy().astype(np.float32)
    query_jax = jnp.array(query_np)
    padded_jax = jnp.array(padded_np)
    
    # Test each method on padded data
    for name, method in methods:
        if name == "maxsim-cpu (Rust)":
            # Test Rust on padded data for comparison
            name = "Rust (padded)"
        
        print(f"\nTesting {name}...")
        times = []
        
        for _ in range(min(10, ITERATIONS)):
            start_time = time.time()
            
            if "NumPy" in name or "Rust" in name:
                result = method(query_np, padded_np)
            elif "JAX" in name:
                result = method(query_jax, padded_jax).block_until_ready()
            else:  # PyTorch methods
                result = method(query_emb, padded_docs)
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        total_time = avg_time + padding_time
        docs_per_sec = VARIABLE_NUM_DOCS / total_time
        
        results[name] = {
            'avg_time': avg_time,
            'total_time': total_time,
            'docs_per_sec': docs_per_sec,
            'needs_padding': True
        }
        print(f"{name}: {total_time:.4f}s total ({avg_time:.4f}s compute + {padding_time:.4f}s padding)")
        print(f"  = {docs_per_sec:.0f} docs/sec")
    
    return results, avg_len

def run_benchmark():
    """Run benchmark"""
    print(f"\n{'='*80}")
    print(f"MAXSIM BENCHMARK: Reranking {NUM_DOCS} Documents")
    if not is_mac_benchmark:
        print(f"Running with {cores_display} Logical Cores")
    print(f"{'='*80}")
    
    results = {}
    
    # Benchmark across different document lengths
    for doc_len in DOC_LENGTHS:
        print(f"\n{'='*60}")
        print(f"Testing with document length: {doc_len} tokens")
        print(f"{'='*60}")
        
        # Create document embeddings for this length
        document_embedding = torch.randn(NUM_DOCS, doc_len, 128, device=device)
        document_embedding = torch.nn.functional.normalize(document_embedding, p=2, dim=-1)
        
        doc_results = {}
        
        # Pre-convert all data formats outside timing loop
        tmp_query = query_embedding.cpu().numpy().astype(np.float32)
        tmp_document = document_embedding.cpu().numpy().astype(np.float32)
        tmp_query_jax = jnp.array(tmp_query)
        tmp_document_jax = jnp.array(tmp_document)
        
        for name, method in methods:
            # warmup every method
            if name in ["NumPy", "maxsim-cpu (Rust)"]:
                _ = method(tmp_query, tmp_document)
            elif "JAX" in name:
                _ = method(tmp_query_jax, tmp_document_jax).block_until_ready()
            else:
                _ = method(query_embedding, document_embedding)

            # Multiple runs for better timing
            times = []
            for _ in range(ITERATIONS):
                start_time = time.time()

                if name in ["NumPy", "maxsim-cpu (Rust)"]:
                    result = method(tmp_query, tmp_document)
                elif "JAX" in name:
                    result = method(tmp_query_jax, tmp_document_jax).block_until_ready()
                else:  # PyTorch methods
                    result = method(query_embedding, document_embedding)

                end_time = time.time()
                times.append(end_time - start_time)

            avg_time = sum(times) / len(times)
            std_time = np.std(times)
            docs_per_sec = NUM_DOCS / avg_time
            
            doc_results[name] = {
                'avg_time': avg_time,
                'std_time': std_time,
                'docs_per_sec': docs_per_sec,
                'result': result
            }
            
            print(f"{name:<20}: {avg_time:.4f}s ± {std_time:.4f}s ({docs_per_sec:>7,.0f} docs/sec)")

        results[doc_len] = doc_results
    
    # Add variable length results
    query_np = query_embedding.cpu().numpy().astype(np.float32)
    var_results, avg_var_len = run_variable_length_benchmark(query_embedding, query_np)
    
    results['variable'] = var_results
    results['variable_avg_len'] = avg_var_len
    
    return results

def create_performance_plot(results_dict):
    """Create performance plot showing docs/second across document lengths"""
    plt.figure(figsize=(14, 8))
    
    # Include variable in x-axis
    x_labels = [str(dl) for dl in DOC_LENGTHS] + ['Variable\n(128-1536)']
    x_pos = np.arange(len(x_labels))
    
    # Color scheme
    colors = {
        "PyTorch (baseline)": "#1f77b4",
        "PyTorch (matmul)": "#17becf",
        "NumPy": "#ff7f0e",
        "JAX": "#d62728",
        "JAX (vmap)": "#e377c2",
        "maxsim-cpu (Rust)": "#2ca02c",
        "Rust (padded)": "#90EE90",
        "Rust Variable API": "#006400"
    }
    
    # Plot results for each method
    for method_name in [m[0] for m in methods]:
        docs_per_sec_values = []
        
        # Special handling for maxsim-cpu (Rust)
        if method_name == "maxsim-cpu (Rust)":
            # Fixed length performance
            for doc_len in DOC_LENGTHS:
                if doc_len in results_dict and method_name in results_dict[doc_len]:
                    docs_per_sec_values.append(results_dict[doc_len][method_name]['docs_per_sec'])
                else:
                    docs_per_sec_values.append(0)
            
            # For variable length, use Rust Variable API performance
            if 'variable' in results_dict and "Rust Variable API" in results_dict['variable']:
                docs_per_sec_values.append(results_dict['variable']["Rust Variable API"]['docs_per_sec'])
            else:
                docs_per_sec_values.append(0)
                
            # Plot continuous line for maxsim-cpu
            if any(d > 0 for d in docs_per_sec_values):
                plt.plot(x_pos, docs_per_sec_values, marker='o', linewidth=2.5, 
                        markersize=10, label=method_name, color=colors.get(method_name, 'gray'))
                
                # Add special marker for Variable API point
                if docs_per_sec_values[-1] > 0:
                    plt.plot([len(DOC_LENGTHS)], [docs_per_sec_values[-1]], marker='s', 
                            markersize=12, color=colors.get(method_name, 'gray'))
                    
                    # Add annotation for impressive performance
                    plt.annotate(f'{docs_per_sec_values[-1]:,.0f}', 
                                xy=(len(DOC_LENGTHS), docs_per_sec_values[-1]), 
                                xytext=(len(DOC_LENGTHS), docs_per_sec_values[-1] * 1.1),
                                ha='center', fontsize=10, fontweight='bold', color='darkgreen')
        else:
            # Other methods
            for doc_len in DOC_LENGTHS:
                if doc_len in results_dict and method_name in results_dict[doc_len]:
                    docs_per_sec_values.append(results_dict[doc_len][method_name]['docs_per_sec'])
                else:
                    docs_per_sec_values.append(0)
            
            # Add variable result for non-Rust methods
            if 'variable' in results_dict and method_name in results_dict['variable']:
                docs_per_sec_values.append(results_dict['variable'][method_name]['docs_per_sec'])
            else:
                docs_per_sec_values.append(0)
            
            if any(d > 0 for d in docs_per_sec_values):
                plt.plot(x_pos, docs_per_sec_values, marker='o', linewidth=2.5, 
                        markersize=10, label=method_name, color=colors.get(method_name, 'gray'))
    
    plt.xlabel('Document Length (tokens)', fontsize=14)
    plt.ylabel('Documents Processed per Second', fontsize=14)
    if is_mac_benchmark:
        plt.title(f'MaxSim Performance: Reranking {NUM_DOCS} Documents\n{cpu_name}', fontsize=16)
    else:
        plt.title(f'MaxSim Performance: Reranking {NUM_DOCS} Documents\n{cores_display} Logical Cores on {cpu_name}', fontsize=16)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.xticks(x_pos, x_labels)
    
    plt.tight_layout()
    if is_mac_benchmark:
        plt.savefig(f'maxsim_performance_mac.png', dpi=300, bbox_inches='tight')
    else:
        plt.savefig(f'maxsim_performance_{cores_display}cores.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_speedup_plot(results_dict):
    """Create speedup plot relative to PyTorch baseline"""
    plt.figure(figsize=(14, 8))
    
    baseline = "PyTorch (baseline)"
    
    # Include variable in x-axis
    x_labels = [str(dl) for dl in DOC_LENGTHS] + ['Variable\n(128-1536)']
    x_pos = np.arange(len(x_labels))
    
    # Add baseline reference line
    plt.axhline(y=1, color='red', linestyle='--', alpha=0.8, 
                label=f'Baseline: {baseline}', linewidth=3)
    
    # Add text to indicate baseline
    plt.text(0.02, 1.02, 'BASELINE = 1.0x', transform=plt.gca().get_yaxis_transform(), 
             color='red', fontweight='bold', fontsize=11, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Color scheme
    colors = {
        "PyTorch (matmul)": "#17becf",
        "NumPy": "#ff7f0e",
        "JAX": "#d62728",
        "JAX (vmap)": "#e377c2",
        "maxsim-cpu (Rust)": "#2ca02c",
        "Rust (padded)": "#90EE90",
        "Rust Variable API": "#006400"
    }
    
    # Plot speedups for each method
    for method_name in [m[0] for m in methods]:
        if method_name == baseline:
            continue
            
        speedup_values = []
        
        # Special handling for maxsim-cpu (Rust)
        if method_name == "maxsim-cpu (Rust)":
            # Fixed length speedups
            for doc_len in DOC_LENGTHS:
                if doc_len in results_dict and method_name in results_dict[doc_len]:
                    baseline_time = results_dict[doc_len][baseline]['avg_time']
                    method_time = results_dict[doc_len][method_name]['avg_time']
                    speedup = baseline_time / method_time
                    speedup_values.append(speedup)
                else:
                    speedup_values.append(0)
            
            # For variable length, use Rust Variable API speedup
            if 'variable' in results_dict and "Rust Variable API" in results_dict['variable']:
                baseline_total = results_dict['variable'][baseline]['total_time']
                rust_var_time = results_dict['variable']["Rust Variable API"]['avg_time']
                speedup = baseline_total / rust_var_time
                speedup_values.append(speedup)
            else:
                speedup_values.append(0)
                
            # Plot continuous line for maxsim-cpu
            if any(v > 0 for v in speedup_values):
                plt.plot(x_pos, speedup_values, marker='o', linewidth=2.5, 
                        markersize=10, label=method_name, color=colors.get(method_name, 'gray'))
                
                # Add special marker and annotation for Variable API point
                if speedup_values[-1] > 0:
                    plt.plot([len(DOC_LENGTHS)], [speedup_values[-1]], marker='s', 
                            markersize=12, color=colors.get(method_name, 'gray'))
                    
                    plt.annotate(f'{speedup_values[-1]:.1f}x', 
                                xy=(len(DOC_LENGTHS), speedup_values[-1]), 
                                xytext=(len(DOC_LENGTHS), speedup_values[-1] * 1.1),
                                ha='center', fontsize=12, fontweight='bold', color='darkgreen',
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
        else:
            # Other methods
            for doc_len in DOC_LENGTHS:
                if doc_len in results_dict and method_name in results_dict[doc_len]:
                    baseline_time = results_dict[doc_len][baseline]['avg_time']
                    method_time = results_dict[doc_len][method_name]['avg_time']
                    speedup = baseline_time / method_time
                    speedup_values.append(speedup)
                else:
                    speedup_values.append(0)
            
            # Variable length speedup for non-Rust methods
            if 'variable' in results_dict and method_name in results_dict['variable']:
                baseline_total = results_dict['variable'][baseline]['total_time']
                method_total = results_dict['variable'][method_name]['total_time']
                speedup = baseline_total / method_total
                speedup_values.append(speedup)
            else:
                speedup_values.append(0)
            
            if any(v > 0 for v in speedup_values):
                plt.plot(x_pos, speedup_values, marker='o', linewidth=2.5, 
                        markersize=10, label=method_name, color=colors.get(method_name, 'gray'))
    
    # Add labels for maxsim-cpu speedups
    y_max = plt.ylim()[1]
    for i, doc_len in enumerate(DOC_LENGTHS):
        if doc_len in results_dict and "maxsim-cpu (Rust)" in results_dict[doc_len]:
            baseline_time = results_dict[doc_len][baseline]['avg_time']
            rust_time = results_dict[doc_len]["maxsim-cpu (Rust)"]['avg_time']
            speedup = baseline_time / rust_time
            
            # Smart positioning
            offset = 0.1 * (y_max - 1)
            y_pos = speedup + offset
            
            if y_pos > y_max * 0.95:
                y_pos = y_max * 0.95
                va = 'top'
            else:
                va = 'bottom'
            
            plt.annotate(f'{speedup:.1f}x', 
                        xy=(i, speedup), 
                        xytext=(i, y_pos),
                        ha='center', va=va, fontsize=11, fontweight='bold', 
                        color='darkgreen',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", 
                                 edgecolor="darkgreen", alpha=0.8))
    
    plt.xlabel('Document Length (tokens)', fontsize=14)
    plt.ylabel(f'Speedup vs {baseline}', fontsize=14)
    if is_mac_benchmark:
        plt.title(f'MaxSim Speedup: Reranking {NUM_DOCS} Documents\n{cpu_name}', fontsize=16)
    else:
        plt.title(f'MaxSim Speedup: Reranking {NUM_DOCS} Documents\n{cores_display} Logical Cores on {cpu_name}', fontsize=16)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.xticks(x_pos, x_labels)
    
    plt.tight_layout()
    if is_mac_benchmark:
        plt.savefig(f'maxsim_speedup_mac.png', dpi=300, bbox_inches='tight')
    else:
        plt.savefig(f'maxsim_speedup_{cores_display}cores.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_markdown_report(results_dict):
    """Create detailed markdown report"""
    if is_mac_benchmark:
        filename = f"maxsim_benchmark_results_mac.md"
    else:
        filename = f"maxsim_benchmark_results_{cores_display}cores.md"
    
    with open(filename, 'w') as f:
        f.write(f"# MaxSim CPU Performance: Reranking {NUM_DOCS} Documents\n\n")
        f.write(f"**Date**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**CPU**: {cpu_name}\n\n")
        if not is_mac_benchmark:
            f.write(f"**Cores Used**: {cores_display}\n\n")
        f.write(f"**Task**: Reranking {NUM_DOCS} documents using MaxSim\n\n")
        f.write(f"**Query Dimensions**: 32 x 128\n\n")
        f.write(f"**Iterations per Test**: {ITERATIONS}\n\n")
        
        f.write("## Performance Summary\n\n")
        
        # Performance table including variable
        f.write("### Documents Processed per Second\n\n")
        headers = [f"{dl} tokens" for dl in DOC_LENGTHS] + [f"Variable ({VARIABLE_MIN_LEN}-{VARIABLE_MAX_LEN})"]
        f.write("| Method | " + " | ".join(headers) + " |\n")
        f.write("|--------|" + "-----------|" * len(headers) + "\n")
        
        # Collect all method names
        all_methods = set()
        for doc_len in DOC_LENGTHS:
            all_methods.update(results_dict[doc_len].keys())
        if 'variable' in results_dict:
            all_methods.update(results_dict['variable'].keys())
        
        for method_name in sorted(all_methods):
            if method_name == "Rust Variable API":
                continue  # Handle separately
                
            row = f"| {method_name} |"
            
            # Fixed lengths
            for doc_len in DOC_LENGTHS:
                if doc_len in results_dict and method_name in results_dict[doc_len]:
                    docs_per_sec = results_dict[doc_len][method_name]['docs_per_sec']
                    row += f" {docs_per_sec:>9,.0f} |"
                else:
                    row += " N/A |"
            
            # Variable length
            if 'variable' in results_dict and method_name in results_dict['variable']:
                docs_per_sec = results_dict['variable'][method_name]['docs_per_sec']
                row += f" {docs_per_sec:>9,.0f} |"
            else:
                row += " N/A |"
                
            f.write(row + "\n")
        
        # Add Rust Variable API row
        if 'variable' in results_dict and "Rust Variable API" in results_dict['variable']:
            row = f"| **Rust Variable API** |"
            row += " N/A |" * len(DOC_LENGTHS)
            docs_per_sec = results_dict['variable']["Rust Variable API"]['docs_per_sec']
            row += f" **{docs_per_sec:>7,.0f}** |"
            f.write(row + "\n")
        
        # Speedup table
        f.write("\n### Speedup vs PyTorch Baseline\n\n")
        f.write("| Method | " + " | ".join(headers) + " |\n")
        f.write("|--------|" + "-----------|" * len(headers) + "\n")
        
        baseline = "PyTorch (baseline)"
        for method_name in sorted(all_methods):
            row = f"| {method_name} |"
            
            # Fixed lengths
            for doc_len in DOC_LENGTHS:
                if doc_len in results_dict and method_name in results_dict[doc_len]:
                    if method_name == baseline:
                        row += " **1.00x** |"
                    else:
                        baseline_time = results_dict[doc_len][baseline]['avg_time']
                        method_time = results_dict[doc_len][method_name]['avg_time']
                        speedup = baseline_time / method_time
                        if speedup > 1:
                            row += f" **{speedup:.2f}x** |"
                        else:
                            row += f" {speedup:.2f}x |"
                else:
                    row += " N/A |"
            
            # Variable length
            if 'variable' in results_dict and method_name in results_dict['variable']:
                if method_name == baseline:
                    row += " **1.00x** |"
                elif method_name == "Rust Variable API":
                    baseline_total = results_dict['variable'][baseline]['total_time']
                    rust_time = results_dict['variable'][method_name]['avg_time']
                    speedup = baseline_total / rust_time
                    row += f" **{speedup:.1f}x** |"
                else:
                    baseline_total = results_dict['variable'][baseline]['total_time']
                    method_total = results_dict['variable'][method_name]['total_time']
                    speedup = baseline_total / method_total
                    if speedup > 1:
                        row += f" **{speedup:.2f}x** |"
                    else:
                        row += f" {speedup:.2f}x |"
            else:
                row += " N/A |"
                
            f.write(row + "\n")
        
        f.write("\n## Key Findings\n\n")
        
        # Calculate average speedup for maxsim-cpu on fixed lengths
        rust_speedups = []
        for doc_len in DOC_LENGTHS:
            if "maxsim-cpu (Rust)" in results_dict[doc_len]:
                baseline_time = results_dict[doc_len][baseline]['avg_time']
                rust_time = results_dict[doc_len]["maxsim-cpu (Rust)"]['avg_time']
                rust_speedups.append(baseline_time / rust_time)
        
        if rust_speedups:
            avg_speedup = np.mean(rust_speedups)
            min_speedup = np.min(rust_speedups)
            max_speedup = np.max(rust_speedups)
            f.write(f"- **maxsim-cpu (Rust)** achieves **{avg_speedup:.1f}x** average speedup over PyTorch baseline on fixed lengths\n")
            f.write(f"  - Speedup range: {min_speedup:.1f}x to {max_speedup:.1f}x\n")
        
        # Variable length findings
        if 'variable' in results_dict and "Rust Variable API" in results_dict['variable']:
            baseline_total = results_dict['variable'][baseline]['total_time']
            rust_var_time = results_dict['variable']["Rust Variable API"]['avg_time']
            var_speedup = baseline_total / rust_var_time
            
            avg_len = results_dict['variable_avg_len']
            max_len = VARIABLE_MAX_LEN
            padding_overhead = (max_len / avg_len - 1) * 100
            
            f.write(f"\n- **Variable Length Performance** (Real-world scenario):\n")
            f.write(f"  - Rust Variable API achieves **{var_speedup:.1f}x** speedup over PyTorch\n")
            f.write(f"  - PyTorch requires padding to {max_len} tokens (average doc length: {avg_len:.0f})\n")
            f.write(f"  - This results in ~{padding_overhead:.0f}% wasted computation for PyTorch\n")
            f.write(f"  - Rust processes actual document lengths without padding overhead\n")
        
        f.write(f"\n## System Configuration\n\n")
        f.write(f"- Python: {platform.python_version()}\n")
        f.write(f"- PyTorch: {torch.__version__}\n")
        f.write(f"- NumPy: {np.__version__}\n")
        f.write(f"- Platform: {platform.platform()}\n")
        if not is_mac_benchmark:
            f.write(f"- CPU cores used: {cores_display} (controlled via taskset)\n")

# Main execution
if __name__ == "__main__":
    print(f"\nRunning MaxSim benchmark for reranking {NUM_DOCS} documents...")
    print(f"Using {ITERATIONS} iterations per test")
    
    # Run benchmark
    results = run_benchmark()
    
    # Create plots
    create_performance_plot(results)
    create_speedup_plot(results)
    
    # Create markdown report
    create_markdown_report(results)
    
    print(f"\nBenchmark complete! Results saved to:")
    if is_mac_benchmark:
        print(f"- maxsim_benchmark_results_mac.md")
        print(f"- maxsim_performance_mac.png")
        print(f"- maxsim_speedup_mac.png")
    else:
        print(f"- maxsim_benchmark_results_{cores_display}cores.md")
        print(f"- maxsim_performance_{cores_display}cores.png")
        print(f"- maxsim_speedup_{cores_display}cores.png")