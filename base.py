import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity, schedule
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # 1. 配置参数 (与 Benchmark 保持一致)
    config = Qwen3VLVisionConfig(
        hidden_size=3072,
        intermediate_size=8192,
        num_heads=24,
        depth=6,
        patch_size=14,
        temporal_patch_size=2,
        spatial_merge_size=2,
        num_position_embeddings=16384,
    )

    # 2. 初始化原生模型
    print(f"Initializing standard Qwen3VLVisionModel on {device}...")
    model = Qwen3VLVisionModel(config).to(device).to(dtype)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # 3. 准备输入数据 (1008x1008)
    T, H, W = 2, 1008, 1008
    grid_thw = torch.tensor([[T // 2, H // 14, W // 14]], device=device)
    num_patches = (T // 2) * (H // 14) * (W // 14)
    # pixel_values 形状符合 patch_embed 的预期
    pixel_values = torch.randn(num_patches, 3, 2, 14, 14, device=device, dtype=dtype)

    print(f"Starting 10 steps of training (Resolution: {H}x{W})...")

    # 4. 训练循环
    # schedule: wait=6 (steps 1-6), warmup=0, active=2 (steps 7-8)
    my_schedule = schedule(wait=6, warmup=0, active=2, repeat=1)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=my_schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/base_train'),
        record_shapes=True,
        with_stack=True
    ) as prof:
        for step in range(10):
            optimizer.zero_grad()

            # 前向传播
            output = model(pixel_values, grid_thw)
            loss = output.last_hidden_state.sum()

            # 反向传播
            loss.backward()

            # 优化器步进
            optimizer.step()

            if device == "cuda":
                torch.cuda.synchronize()

            prof.step() # 通知 profiler 步进
            print(f"Step {step + 1}/10 | Loss: {loss.item():.4f}")

    # 同时导出 chrome trace 以便可视化
    prof.export_chrome_trace("qwen3_vl_base_train_steps_7_8.json")
    print("Training finished. Trace saved to qwen3_vl_base_train_steps_7_8.json")

if __name__ == "__main__":
    train()
