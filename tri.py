import sys
import os
# Ensure we use the local transformers code
sys.path.insert(0, "/home/layz/workspace/transformers/src")

import torch
from torch.profiler import profile, ProfilerActivity, schedule, record_function
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig, Qwen3VLVisionConfig, Qwen3VLTextConfig

def train_tri():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Use float32 for compute_3d_position_ids as it deals with indices, but model weights can be bfloat16
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # 1. Setup Config
    vision_config = Qwen3VLVisionConfig(
        hidden_size=1024, # Smaller for faster init
        intermediate_size=4096,
        num_heads=16,
        depth=2,
        patch_size=14,
        temporal_patch_size=2,
        spatial_merge_size=2,
        out_hidden_size=1024,
    )
    text_config = Qwen3VLTextConfig(
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=2,
        num_attention_heads=16,
        num_key_value_heads=16,
        vocab_size=1000,
    )
    config = Qwen3VLConfig(
        vision_config=vision_config,
        text_config=text_config,
    )
    config.image_token_id = 10
    config.video_token_id = 11

    # 2. Initialize Model
    print(f"Initializing Qwen3VLModel on {device}...")
    model = Qwen3VLModel(config).to(device).to(dtype)
    model.train()

    # 3. Prepare Inputs
    # Simulate: [Text(5)] [Image] [Text(5)]
    # Vision grid: T=2, H=1008, W=1008 -> patches: 1x72x72
    T, H, W = 2, 1008, 1008
    grid_thw = torch.tensor([[T // 2, H // 14, W // 14]], device=device)

    # Vision tokens after spatial merge (size=2): 1 * 36 * 36 = 1296
    num_vision_tokens = (T // 2) * (H // (14 * 2)) * (W // (14 * 2))

    seq_len = 5 + num_vision_tokens + 5
    input_ids = torch.zeros((1, seq_len), dtype=torch.long, device=device)
    mm_token_type_ids = torch.zeros((1, seq_len), dtype=torch.int, device=device)

    # Mark image tokens
    mm_token_type_ids[0, 5:5+num_vision_tokens] = 1
    # Place placeholder tokens (though compute_3d_position_ids might not strictly need them if mm_token_type_ids is provided)
    input_ids[0, 5:5+num_vision_tokens] = config.image_token_id

    # For the actual forward pass we'd need pixel_values, but we want to profile compute_3d_position_ids
    pixel_values = torch.randn(num_vision_tokens * 4, 3, 2, 14, 14, device=device, dtype=dtype)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    print(f"Starting 10 steps of profiling compute_3d_position_ids...")

    my_schedule = schedule(wait=6, warmup=0, active=2, repeat=1)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=my_schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/tri_train'),
        record_shapes=True,
        with_stack=True
    ) as prof:
        for step in range(10):
            optimizer.zero_grad()

            # We profile the compute_3d_position_ids specifically
            with record_function("compute_3d_position_ids_total"):
                position_ids = model.compute_3d_position_ids(
                    input_ids=input_ids,
                    inputs_embeds=None,
                    image_grid_thw=grid_thw,
                    mm_token_type_ids=mm_token_type_ids
                )

            # To make it a "training process", we do a dummy forward/backward
            # Since we only want to profile compute_3d_position_ids,
            # we can skip the heavy vision part if possible, or just do it.
            # Here we call model.forward to keep it realistic
            output = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=grid_thw,
                mm_token_type_ids=mm_token_type_ids,
                position_ids=position_ids
            )

            loss = output.last_hidden_state.sum()
            loss.backward()
            optimizer.step()

            if device == "cuda":
                torch.cuda.synchronize()

            prof.step()
            print(f"Step {step + 1}/10 | Loss: {loss.item():.4f}")

    prof.export_chrome_trace("qwen3_vl_tri_compute_3d_pos_ids.json")
    print("Profiling finished. Trace saved to qwen3_vl_tri_compute_3d_pos_ids.json")

if __name__ == "__main__":
    train_tri()
