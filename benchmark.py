#!/usr/bin/env python3
"""
Hardware Benchmarking Script for Diffusion Model
Measures: throughput, latency, VRAM usage, multi-GPU scaling
"""

import torch
import time
import json
import os
import subprocess
from datetime import datetime
from tqdm import tqdm

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
    train_loader, _ = create_dataloader(
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
        
        train_loader, _ = create_dataloader(
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
    
    print("\n" + "="*60)
    print("📁 RESULTS SAVED")
    print("="*60)
    print(f"   File: {output_file}")
    print("\n✅ Benchmark complete!")


if __name__ == '__main__':
    main()
