import time
import numpy as np
try:
    import cupy as cp
except ImportError:
    print("cupy未安装，请先安装cupy。例如：")
    print("pip install cupy-cuda11x  (x替换为你的CUDA版本，如cupy-cuda118)")
    exit()

def check_gpu_info():
    """检查GPU基本信息"""
    try:
        print("\n=== GPU设备信息 ===")
        n_devices = cp.cuda.runtime.getDeviceCount()
        print(f"发现 {n_devices} 个 GPU 设备")
        
        for i in range(n_devices):
            props = cp.cuda.runtime.getDeviceProperties(i)
            print(f"\nGPU {i}: {props['name'].decode()}")
            print(f"计算能力: {props['major']}.{props['minor']}")
            print(f"总显存: {props['totalGlobalMem'] / (1024**3):.2f} GB")
            print(f"最大线程数/块: {props['maxThreadsPerBlock']}")
            print(f"最大共享内存/块: {props['sharedMemPerBlock'] / 1024:.2f} KB")
            print(f"时钟频率: {props['clockRate'] / 1000:.2f} GHz")
            
        return True
    except Exception as e:
        print(f"获取GPU信息失败: {str(e)}")
        return False

def test_gpu_performance():
    """测试GPU性能"""
    try:
        print("\n=== GPU性能测试 ===")
        
        # 测试数据大小
        sizes = [
            (1000, 1000),
            (2000, 2000),
            (4000, 4000),
            (8000, 8000)
        ]
        
        print("\n矩阵乘法性能测试:")
        print("大小\t\tCPU时间\tGPU时间\t加速比")
        print("-" * 50)
        
        for size in sizes:
            # 创建随机矩阵
            a_cpu = np.random.random(size)
            b_cpu = np.random.random(size)
            
            # CPU测试
            start_time = time.time()
            np.dot(a_cpu, b_cpu)
            cpu_time = time.time() - start_time
            
            # GPU测试
            a_gpu = cp.asarray(a_cpu)
            b_gpu = cp.asarray(b_cpu)
            
            start_time = time.time()
            cp.dot(a_gpu, b_gpu)
            cp.cuda.Stream.null.synchronize()
            gpu_time = time.time() - start_time
            
            # 计算加速比
            speedup = cpu_time / gpu_time
            
            print(f"{size[0]}x{size[1]}\t{cpu_time:.4f}s\t{gpu_time:.4f}s\t{speedup:.2f}x")
            
            # 清理GPU内存
            del a_gpu, b_gpu
            cp.get_default_memory_pool().free_all_blocks()
            
        return True
    except Exception as e:
        print(f"性能测试失败: {str(e)}")
        return False

def test_gpu_memory():
    """测试GPU内存"""
    try:
        print("\n=== GPU内存测试 ===")
        
        # 获取当前内存使用情况
        mem_info = cp.cuda.runtime.memGetInfo()
        free_mem = mem_info[0] / (1024**3)
        total_mem = mem_info[1] / (1024**3)
        used_mem = total_mem - free_mem
        
        print(f"总显存: {total_mem:.2f} GB")
        print(f"已用显存: {used_mem:.2f} GB")
        print(f"可用显存: {free_mem:.2f} GB")
        print(f"显存使用率: {(used_mem/total_mem)*100:.2f}%")
        
        # 测试内存分配
        print("\n测试内存分配...")
        test_sizes = [0.1, 0.5, 1.0, 2.0]  # GB
        
        for size in test_sizes:
            try:
                n_elements = int(size * (1024**3) / 8)  # 转换为元素数量（假设每个元素8字节）
                print(f"\n尝试分配 {size:.1f} GB 内存...")
                
                start_time = time.time()
                x = cp.zeros(n_elements, dtype=cp.float64)
                allocation_time = time.time() - start_time
                
                # 执行一些操作来验证内存可用
                x += 1
                cp.cuda.Stream.null.synchronize()
                
                print(f"✓ 成功分配和使用 {size:.1f} GB 内存，用时: {allocation_time:.4f}秒")
                
                # 释放内存
                del x
                cp.get_default_memory_pool().free_all_blocks()
                
            except Exception as e:
                print(f"✗ 分配 {size:.1f} GB 内存失败: {str(e)}")
                break
        
        return True
    except Exception as e:
        print(f"内存测试失败: {str(e)}")
        return False

def main():
    """主函数"""
    try:
        print("=== GPU可用性测试程序 ===")
        print(f"CuPy版本: {cp.__version__}")
        
        # 测试GPU信息
        if not check_gpu_info():
            return
        
        # 测试GPU内存
        if not test_gpu_memory():
            return
        
        # 测试GPU性能
        if not test_gpu_performance():
            return
        
        print("\n=== 测试完成 ===")
        print("GPU工作正常！✓")
        
    except Exception as e:
        print(f"\n测试过程中发生错误: {str(e)}")
        print("请确保：")
        print("1. NVIDIA GPU驱动已正确安装")
        print("2. CUDA工具包已正确安装")
        print("3. cupy版本与CUDA版本匹配")

if __name__ == "__main__":
    main()