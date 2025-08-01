import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os


def setup_process(rank, world_size):
    """设置分布式进程"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'

    # 初始化进程组，使用 NCCL 后端处理 GPU 通信
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_process():
    """清理进程"""
    dist.destroy_process_group()


def all_reduce_demo(rank, world_size):
    """All-Reduce 演示函数"""
    # 设置当前GPU设备
    torch.cuda.set_device(rank)
    device = f'cuda:{rank}'

    print(f"\n=== 进程 {rank} (GPU {rank}) ===")

    # 创建测试张量 - 每个进程创建不同的值
    original_tensor = torch.tensor([
        [1.0 + rank, 2.0 + rank],
        [3.0 + rank, 4.0 + rank]
    ], device=device)

    print(f"进程 {rank} 的原始张量:")
    print(original_tensor)

    # 测试不同的 All-Reduce 操作
    operations = [
        (dist.ReduceOp.SUM, "求和"),
        (dist.ReduceOp.AVG, "平均"),
        (dist.ReduceOp.MAX, "最大值"),
        (dist.ReduceOp.MIN, "最小值")
    ]

    for op, op_name in operations:
        # 复制原始张量
        tensor_copy = original_tensor.clone()

        # 执行 All-Reduce
        dist.all_reduce(tensor_copy, op=op)

        print(f"\n进程 {rank} - {op_name}操作后的结果:")
        print(tensor_copy)


def main():
    """主函数"""
    # 使用 GPU 0,1,2,3,4
    gpu_ids = [0, 1, 2, 3, 4]
    available_gpus = torch.cuda.device_count()

    print(f"系统可用GPU数量: {available_gpus}")

    # 过滤出实际可用的GPU
    valid_gpu_ids = [gpu_id for gpu_id in gpu_ids if gpu_id < available_gpus]

    if len(valid_gpu_ids) == 0:
        print("没有可用的GPU!")
        return

    world_size = len(valid_gpu_ids)
    print(f"将使用GPU: {valid_gpu_ids}")

    # 设置环境变量
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, valid_gpu_ids))

    try:
        # 启动多进程
        mp.spawn(
            all_reduce_demo,
            args=(world_size,),
            nprocs=world_size,
            join=True
        )
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        print("演示完成")


# 简化版本 - 只做基本的求和操作
def simple_all_reduce_demo(rank, world_size):
    """简化版 All-Reduce 演示"""
    setup_process(rank, world_size)

    # 设置GPU设备
    device = torch.device(f'cuda:{rank}')

    # 创建张量
    tensor = torch.tensor([[rank + 1.0, rank + 2.0],
                           [rank + 3.0, rank + 4.0]], device=device)

    print(f"进程 {rank} (GPU {rank}) - 原始张量:")
    print(tensor)
    print("-" * 30)

    # All-Reduce 操作 (默认求和)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    print(f"进程 {rank} (GPU {rank}) - All-Reduce后:")
    print(tensor)

    cleanup_process()


if __name__ == "__main__":
    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        print("CUDA 不可用!")
        exit(1)

    # 运行简化版本
    world_size = min(5, torch.cuda.device_count())  # 最多使用5个GPU
    if world_size == 0:
        print("没有可用的GPU!")
        exit(1)

    print(f"使用 {world_size} 个GPU进行演示")

    mp.spawn(
        simple_all_reduce_demo,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )