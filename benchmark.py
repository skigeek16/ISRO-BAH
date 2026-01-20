#!/usr/bin/env python3
"""
Hardware Benchmarking Script for Diffusion Model
Measures: throughput, latency, VRAM usage, multi-GPU scaling
Generates visual benchmark reports
"""

import torch
import time
import json
import os
import subprocess
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for servers

from diffusion_model import DiffusionModel
from dataset import create_dataloader


def get_gpu_memory_info():
    """Get GPU memory usage for AMD/ROCm"""
    try:
        result = subprocess.run(
            ['rocm-smi', '--showmeminfo', 'vram', '--json'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            # Parse rocm-smi output
            memory_info = []
            for card_id, card_data in data.items():
                if 'VRAM Total Memory' in str(card_data):
                    memory_info.append({
                        'gpu': card_id,
                        'used': card_data.get('VRAM Total Used Memory (B)', 0),
                        'total': card_data.get('VRAM Total Memory (B)', 0)
                    })
            return memory_info
    except Exception as e:
        pass
    
    # Fallback to PyTorch
    if torch.cuda.is_available():
        return [{
            'gpu': i,
            'used': torch.cuda.memory_allocated(i),
            'total': torch.cuda.get_device_properties(i).total_memory
        } for i in range(torch.cuda.device_count())]
    return []


def format_bytes(bytes_val):
    """Format bytes to human readable"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.2f} PB"


def benchmark_training(config):
    """Benchmark training throughput and memory"""
    print("\n" + "="*60)
    print("🔬 TRAINING BENCHMARK")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    print(f"\n📊 Hardware Configuration:")
    print(f"   Device: {device}")
    print(f"   GPUs: {num_gpus}")
    if num_gpus > 0:
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"   GPU {i}: {props.name} ({format_bytes(props.total_memory)})")
    
    # Create model
    model = DiffusionModel().to(device)
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
        print(f"   Mode: DataParallel ({num_gpus} GPUs)")
    
    # Count parameters
    param_count = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {param_count:,} ({param_count/1e6:.1f}M)")
    
    # Create dataloader
    train_loader = create_dataloader(
        config['data_dir'], 
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    # Warmup
    print(f"\n⏱️  Warming up (5 batches)...")
    warmup_batches = 5
    for i, (context, target) in enumerate(train_loader):
        if i >= warmup_batches:
            break
        context = context.to(device)
        target = target.to(device)
        t = torch.randint(0, 1000, (context.shape[0],), device=device)
        noise = torch.randn_like(target)
        
        base_model = model.module if hasattr(model, 'module') else model
        with torch.amp.autocast('cuda', enabled=config['use_amp']):
            noisy = base_model.forward_diffusion(target, t, noise)
            pred = base_model.predict_noise(noisy, context, t)
            loss = torch.nn.functional.mse_loss(pred, noise)
        loss.backward()
    
    # Clear gradients and cache
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    
    # Benchmark
    print(f"\n📈 Benchmarking ({config['benchmark_batches']} batches)...")
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    start_time = time.time()
    total_samples = 0
    batch_times = []
    
    for i, (context, target) in enumerate(tqdm(train_loader, total=config['benchmark_batches'])):
        if i >= config['benchmark_batches']:
            break
        
        batch_start = time.time()
        
        context = context.to(device)
        target = target.to(device)
        t = torch.randint(0, 1000, (context.shape[0],), device=device)
        noise = torch.randn_like(target)
        
        base_model = model.module if hasattr(model, 'module') else model
        with torch.amp.autocast('cuda', enabled=config['use_amp']):
            noisy = base_model.forward_diffusion(target, t, noise)
            pred = base_model.predict_noise(noisy, context, t)
            loss = torch.nn.functional.mse_loss(pred, noise)
        
        loss.backward()
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        total_samples += context.shape[0]
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    total_time = time.time() - start_time
    
    # Get memory info
    memory_info = get_gpu_memory_info()
    peak_memory = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    
    # Calculate metrics
    throughput = total_samples / total_time
    avg_batch_time = sum(batch_times) / len(batch_times)
    latency_per_sample = avg_batch_time / config['batch_size'] * 1000  # ms
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'hardware': {
            'num_gpus': num_gpus,
            'gpu_names': [torch.cuda.get_device_properties(i).name for i in range(num_gpus)] if num_gpus > 0 else [],
            'total_vram_gb': sum(torch.cuda.get_device_properties(i).total_memory for i in range(num_gpus)) / 1e9 if num_gpus > 0 else 0
        },
        'config': {
            'batch_size': config['batch_size'],
            'use_amp': config['use_amp'],
            'num_gpus_used': num_gpus
        },
        'training': {
            'throughput_samples_per_sec': throughput,
            'avg_batch_time_sec': avg_batch_time,
            'latency_per_sample_ms': latency_per_sample,
            'peak_memory_gb': peak_memory / 1e9,
            'total_samples': total_samples,
            'total_time_sec': total_time
        }
    }
    
    print(f"\n📊 TRAINING RESULTS:")
    print(f"   Throughput: {throughput:.2f} samples/sec")
    print(f"   Batch time: {avg_batch_time*1000:.1f} ms")
    print(f"   Latency/sample: {latency_per_sample:.1f} ms")
    print(f"   Peak VRAM: {format_bytes(peak_memory)}")
    
    return results


def benchmark_inference(config):
    """Benchmark inference latency"""
    print("\n" + "="*60)
    print("🔬 INFERENCE BENCHMARK")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model
    model = DiffusionModel().to(device)
    model.eval()
    
    # Create dummy input
    context = torch.randn(1, 20, 720, 720).to(device)
    
    # Warmup
    print(f"\n⏱️  Warming up...")
    for _ in range(3):
        with torch.no_grad():
            _ = model.sample(context, device, use_ddim=True, ddim_steps=50)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    # Benchmark DDIM with different steps
    ddim_steps_list = [50, 100, 200]
    inference_results = {}
    
    for ddim_steps in ddim_steps_list:
        print(f"\n📈 Benchmarking DDIM {ddim_steps} steps...")
        times = []
        
        for _ in tqdm(range(5)):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.time()
            
            with torch.no_grad():
                _ = model.sample(context, device, use_ddim=True, ddim_steps=ddim_steps)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            times.append(time.time() - start)
        
        avg_time = sum(times) / len(times)
        inference_results[f'ddim_{ddim_steps}_steps'] = {
            'avg_time_sec': avg_time,
            'avg_time_ms': avg_time * 1000
        }
        print(f"   DDIM {ddim_steps}: {avg_time*1000:.1f} ms")
    
    return {'inference': inference_results}


def benchmark_scaling(config):
    """Benchmark multi-GPU scaling efficiency"""
    print("\n" + "="*60)
    print("🔬 MULTI-GPU SCALING BENCHMARK")
    print("="*60)
    
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus < 2:
        print("   Skipping - need 2+ GPUs for scaling test")
        return {}
    
    device = 'cuda'
    scaling_results = {}
    
    for gpu_count in [1, num_gpus]:
        if gpu_count == 1:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            torch.cuda.set_device(0)
        else:
            if 'CUDA_VISIBLE_DEVICES' in os.environ:
                del os.environ['CUDA_VISIBLE_DEVICES']
        
        print(f"\n   Testing with {gpu_count} GPU(s)...")
        
        model = DiffusionModel().to(device)
        if gpu_count > 1:
            model = torch.nn.DataParallel(model)
        
        train_loader = create_dataloader(
            config['data_dir'],
            batch_size=config['batch_size'] * gpu_count,
            num_workers=config['num_workers']
        )
        
        # Warmup
        for i, (context, target) in enumerate(train_loader):
            if i >= 3:
                break
            context = context.to(device)
            target = target.to(device)
            t = torch.randint(0, 1000, (context.shape[0],), device=device)
            noise = torch.randn_like(target)
            base_model = model.module if hasattr(model, 'module') else model
            with torch.amp.autocast('cuda', enabled=config['use_amp']):
                noisy = base_model.forward_diffusion(target, t, noise)
                pred = base_model.predict_noise(noisy, context, t)
                loss = torch.nn.functional.mse_loss(pred, noise)
            loss.backward()
        
        # Benchmark
        torch.cuda.synchronize()
        start = time.time()
        total_samples = 0
        
        for i, (context, target) in enumerate(train_loader):
            if i >= 20:
                break
            context = context.to(device)
            target = target.to(device)
            t = torch.randint(0, 1000, (context.shape[0],), device=device)
            noise = torch.randn_like(target)
            base_model = model.module if hasattr(model, 'module') else model
            with torch.amp.autocast('cuda', enabled=config['use_amp']):
                noisy = base_model.forward_diffusion(target, t, noise)
                pred = base_model.predict_noise(noisy, context, t)
                loss = torch.nn.functional.mse_loss(pred, noise)
            loss.backward()
            total_samples += context.shape[0]
        
        torch.cuda.synchronize()
        elapsed = time.time() - start
        throughput = total_samples / elapsed
        
        scaling_results[f'{gpu_count}_gpu'] = {
            'throughput': throughput,
            'samples': total_samples,
            'time': elapsed
        }
        
        print(f"   {gpu_count} GPU: {throughput:.2f} samples/sec")
        
        del model
        torch.cuda.empty_cache()
    
    # Calculate scaling efficiency
    if '1_gpu' in scaling_results and f'{num_gpus}_gpu' in scaling_results:
        speedup = scaling_results[f'{num_gpus}_gpu']['throughput'] / scaling_results['1_gpu']['throughput']
        efficiency = speedup / num_gpus * 100
        scaling_results['scaling_efficiency'] = {
            'speedup': speedup,
            'efficiency_percent': efficiency,
            'ideal_speedup': num_gpus
        }
        print(f"\n   Speedup: {speedup:.2f}x (ideal: {num_gpus}x)")
        print(f"   Efficiency: {efficiency:.1f}%")
    
    return {'scaling': scaling_results}


def create_benchmark_visualizations(results, output_dir):
    """Create visualization graphs for benchmark results"""
    print("\n" + "="*60)
    print("📊 GENERATING VISUALIZATIONS")
    print("="*60)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Diffusion Model Hardware Benchmark Results', fontsize=16, fontweight='bold')
    
    # 1. Training Throughput Bar Chart
    ax1 = axes[0, 0]
    if 'training' in results:
        training = results['training']
        metrics = ['Throughput\n(samples/sec)', 'Batch Time\n(ms)', 'Latency/Sample\n(ms)']
        values = [
            training.get('throughput_samples_per_sec', 0),
            training.get('avg_batch_time_sec', 0) * 1000,
            training.get('latency_per_sample_ms', 0)
        ]
        colors = ['#2ecc71', '#3498db', '#9b59b6']
        bars = ax1.bar(metrics, values, color=colors, edgecolor='white', linewidth=1.5)
        ax1.set_title('Training Performance', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Value')
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 2. DDIM Inference Latency
    ax2 = axes[0, 1]
    if 'inference' in results:
        inference = results['inference']
        steps = []
        times = []
        for key, val in inference.items():
            if 'ddim' in key:
                step_num = int(key.split('_')[1])
                steps.append(step_num)
                times.append(val.get('avg_time_ms', 0))
        
        if steps:
            # Sort by steps
            sorted_data = sorted(zip(steps, times))
            steps, times = zip(*sorted_data)
            
            ax2.plot(steps, times, 'o-', color='#e74c3c', linewidth=2.5, markersize=10, markerfacecolor='white', markeredgewidth=2)
            ax2.fill_between(steps, times, alpha=0.3, color='#e74c3c')
            ax2.set_title('Inference Latency vs DDIM Steps', fontsize=12, fontweight='bold')
            ax2.set_xlabel('DDIM Steps')
            ax2.set_ylabel('Latency (ms)')
            ax2.set_xticks(steps)
            # Add value labels
            for s, t in zip(steps, times):
                ax2.annotate(f'{t:.0f}ms', (s, t), textcoords="offset points",
                            xytext=(0, 10), ha='center', fontsize=9, fontweight='bold')
    
    # 3. Memory Usage
    ax3 = axes[1, 0]
    if 'training' in results:
        peak_mem_gb = results['training'].get('peak_memory_gb', 0)
        if 'hardware' in results:
            total_mem_gb = results['hardware'].get('total_vram_gb', 256)
        else:
            total_mem_gb = 256
        
        used_pct = (peak_mem_gb / total_mem_gb) * 100 if total_mem_gb > 0 else 0
        free_pct = 100 - used_pct
        
        sizes = [used_pct, free_pct]
        labels = [f'Used\n{peak_mem_gb:.1f} GB', f'Free\n{total_mem_gb - peak_mem_gb:.1f} GB']
        colors = ['#e74c3c', '#2ecc71']
        explode = (0.05, 0)
        
        wedges, texts, autotexts = ax3.pie(sizes, labels=labels, colors=colors, explode=explode,
                                            autopct='%1.1f%%', startangle=90,
                                            textprops={'fontsize': 10})
        ax3.set_title(f'GPU Memory Usage (Total: {total_mem_gb:.0f} GB)', fontsize=12, fontweight='bold')
    
    # 4. Hardware Info & Summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = "📋 BENCHMARK SUMMARY\n" + "="*40 + "\n\n"
    
    if 'hardware' in results:
        hw = results['hardware']
        summary_text += f"🖥️  Hardware:\n"
        summary_text += f"   • GPUs: {hw.get('num_gpus', 0)}\n"
        if hw.get('gpu_names'):
            summary_text += f"   • Model: {hw['gpu_names'][0]}\n"
        summary_text += f"   • Total VRAM: {hw.get('total_vram_gb', 0):.0f} GB\n\n"
    
    if 'config' in results:
        cfg = results['config']
        summary_text += f"⚙️  Configuration:\n"
        summary_text += f"   • Batch Size: {cfg.get('batch_size', 0)}\n"
        summary_text += f"   • Mixed Precision: {'✓' if cfg.get('use_amp') else '✗'}\n\n"
    
    if 'training' in results:
        tr = results['training']
        summary_text += f"📈 Training Metrics:\n"
        summary_text += f"   • Throughput: {tr.get('throughput_samples_per_sec', 0):.2f} samples/sec\n"
        summary_text += f"   • Batch Time: {tr.get('avg_batch_time_sec', 0)*1000:.1f} ms\n"
        summary_text += f"   • Peak Memory: {tr.get('peak_memory_gb', 0):.2f} GB\n"
    
    ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save figure
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_path = os.path.join(output_dir, f'benchmark_report_{timestamp}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   ✓ Saved: {plot_path}")
    return plot_path


def main():
    config = {
        'data_dir': 'Data/',
        'batch_size': 4,
        'num_workers': 4,
        'use_amp': True,
        'benchmark_batches': 50
    }
    
    print("\n" + "="*60)
    print("🚀 DIFFUSION MODEL HARDWARE BENCHMARK")
    print("="*60)
    print(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'pytorch_version': torch.__version__,
        'config': config
    }
    
    # Run benchmarks
    training_results = benchmark_training(config)
    all_results.update(training_results)
    
    inference_results = benchmark_inference(config)
    all_results.update(inference_results)
    
    # Save results
    os.makedirs('benchmarks', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'benchmarks/benchmark_{timestamp}.json'
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Generate visualizations
    plot_path = create_benchmark_visualizations(all_results, 'benchmarks')
    
    print("\n" + "="*60)
    print("📁 RESULTS SAVED")
    print("="*60)
    print(f"   JSON: {output_file}")
    print(f"   Plot: {plot_path}")
    print("\n✅ Benchmark complete!")


if __name__ == '__main__':
    main()
