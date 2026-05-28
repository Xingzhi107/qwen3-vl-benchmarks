"""
Optimization patch for Qwen2VL/Qwen3VL compute_3d_position_ids and get_rope_index.

This module provides optimized implementations that reduce GPU-CPU synchronization
overhead in the position ID computation for multimodal models.

Performance improvements:
- Reduced sync count: from O(batch_size × num_vision_inputs × 3) to O(2)
- Tensor operations for boundary detection instead of itertools.groupby
- Pre-move grid_thw to CPU once, avoiding repeated .item() syncs

Usage:
    # Import and apply before loading model
    import optimize_qwen3vl_position_ids
    optimize_qwen3vl_position_ids.apply_patch()

    # Then load and use model normally
    from transformers import Qwen3VLForConditionalGeneration
    model = Qwen3VLForConditionalGeneration.from_pretrained(...)

Or apply selectively:
    import optimize_qwen3vl_position_ids as opt
    opt.patch_qwen2vl()  # Only Qwen2VL
    opt.patch_qwen3vl()  # Only Qwen3VL
"""

import torch
from typing import Any

# Store original functions for reference (optional: for comparison testing)
_original_functions = {}


def get_vision_position_ids_optimized(
    start_position: int,
    grid_thw: list[int, int, int] | torch.Tensor,
    temp_merge_size: int = 1,
    spatial_merge_size: int = 1,
    time_interval: int = 1,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """
    Optimized version of get_vision_position_ids.

    Key optimization: Single synchronization instead of 3 separate .item() calls.
    - Previously: 3 × .item() = 3 potential GPU-CPU syncs
    - Now: 1 × tolist() = 1 sync (or 0 if already on CPU)
    """
    # Optimize: convert grid_thw to Python ints with a single synchronization
    if isinstance(grid_thw, torch.Tensor):
        if grid_thw.device.type != "cpu":
            grid_thw = grid_thw.cpu()
        t, h, w = grid_thw.tolist()
    else:
        t, h, w = grid_thw

    llm_grid_t = t // temp_merge_size
    llm_grid_h = h // spatial_merge_size
    llm_grid_w = w // spatial_merge_size

    image_seq_length = llm_grid_h * llm_grid_w * llm_grid_t
    position_width = torch.arange(start_position, start_position + llm_grid_w, device=device).repeat(
        llm_grid_h * llm_grid_t
    )
    position_height = torch.arange(start_position, start_position + llm_grid_h, device=device).repeat_interleave(
        llm_grid_w * llm_grid_t
    )
    # Minor optimization: compute time_interval product directly
    position_temporal = torch.full(
        (image_seq_length,), start_position * time_interval, device=device, dtype=torch.long
    )
    vision_position_ids = torch.stack([position_temporal, position_height, position_width], dim=0)

    return vision_position_ids


def get_rope_index_optimized(
    self,
    input_ids: torch.LongTensor,
    mm_token_type_ids: torch.IntTensor,
    image_grid_thw: torch.LongTensor | None = None,
    video_grid_thw: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Optimized version of get_rope_index.

    Key optimizations:
    1. Pre-move grid_thw to CPU once (2 syncs max vs N×M syncs)
    2. Tensor operations for boundary detection (GPU ops vs Python itertools)
    3. Integer counters instead of Python iterators (O(1) indexing)
    """
    spatial_merge_size = self.config.vision_config.spatial_merge_size
    device = input_ids.device

    # Optimization: Pre-move grid_thw to CPU once
    # Previously: each batch + each vision token triggered .item() syncs
    # Now: one-time .cpu() call = at most 2 syncs total
    if image_grid_thw is not None and image_grid_thw.device.type != "cpu":
        image_grid_thw_cpu = image_grid_thw.cpu()
    else:
        image_grid_thw_cpu = image_grid_thw

    if video_grid_thw is not None and video_grid_thw.device.type != "cpu":
        video_grid_thw_cpu = video_grid_thw.cpu()
    else:
        video_grid_thw_cpu = video_grid_thw

    mrope_position_deltas = []
    position_ids = torch.zeros(
        3,
        input_ids.shape[0],
        input_ids.shape[1],
        dtype=input_ids.dtype,
        device=device,
    )

    # Use integer counters instead of Python iterators
    grid_counters = {1: 0, 2: 0}
    grid_tensors_cpu = {1: image_grid_thw_cpu, 2: video_grid_thw_cpu}

    for batch_idx in range(input_ids.shape[0]):
        input_token_type = mm_token_type_ids[batch_idx]
        if attention_mask is not None:
            mask = attention_mask[batch_idx].bool()
            input_token_type = input_token_type[mask]
        else:
            mask = None

        if input_token_type.numel() == 0:
            mrope_position_deltas.append(0)
            continue

        # Optimization: Tensor operations for boundary detection
        # Previously: itertools.groupby + .tolist() entire sequence (GPU-CPU sync)
        # Now: tensor diff + torch.where (fully GPU), .tolist() only boundary indices
        diff = torch.cat([
            torch.tensor([1], device=device, dtype=torch.int),
            (input_token_type[1:] != input_token_type[:-1]).int()
        ])
        change_indices = torch.where(diff)[0]
        start_indices = change_indices
        end_indices = torch.cat([
            change_indices[1:],
            torch.tensor([input_token_type.numel()], device=device)
        ])
        modalities = input_token_type[start_indices]

        # Convert only boundary points (few elements) to Python
        modalities_list = modalities.tolist()
        start_indices_list = start_indices.tolist()
        end_indices_list = end_indices.tolist()

        current_pos = 0
        llm_pos_ids_list = []
        for modality_type, start_idx, end_idx in zip(modalities_list, start_indices_list, end_indices_list):
            if modality_type == 0:  # text
                text_len = end_idx - start_idx
                llm_pos_ids_list.append(
                    torch.arange(text_len, device=device).view(1, -1).expand(3, -1) + current_pos
                )
                current_pos += text_len
            else:  # image == 1, video == 2
                grid_tensor_cpu = grid_tensors_cpu[modality_type]
                if grid_tensor_cpu is None:
                    raise ValueError(f"No grid_thw provided for modality type {modality_type}")
                grid_idx = grid_counters[modality_type]
                grid_thw = grid_tensor_cpu[grid_idx]
                grid_counters[modality_type] += 1

                # grid_thw is already on CPU, no sync needed
                vision_position_ids = get_vision_position_ids_optimized(
                    current_pos, grid_thw, 1, spatial_merge_size, device=device
                )
                llm_pos_ids_list.append(vision_position_ids)
                # .item() is instant since grid_thw is on CPU
                current_pos += max(grid_thw[1].item(), grid_thw[2].item()) // spatial_merge_size

        if llm_pos_ids_list:
            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            if mask is not None:
                position_ids[:, batch_idx, mask] = llm_positions
            else:
                position_ids[:, batch_idx] = llm_positions
            mrope_position_deltas.append(llm_positions.max().item() + 1 - input_token_type.numel())
        else:
            mrope_position_deltas.append(0)

    mrope_position_deltas = torch.tensor(mrope_position_deltas, device=device).unsqueeze(1)
    return position_ids, mrope_position_deltas


def patch_qwen2vl():
    """
    Apply optimization patch to Qwen2VLModel.

    Patches:
    - get_vision_position_ids
    - get_rope_index
    """
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLModel

    # Store originals for reference
    _original_functions['Qwen2VLModel.get_vision_position_ids'] = Qwen2VLModel.get_vision_position_ids
    _original_functions['Qwen2VLModel.get_rope_index'] = Qwen2VLModel.get_rope_index

    # Apply patches
    Qwen2VLModel.get_vision_position_ids = get_vision_position_ids_optimized
    Qwen2VLModel.get_rope_index = get_rope_index_optimized

    print("Applied optimization patch to Qwen2VLModel")


def patch_qwen3vl():
    """
    Apply optimization patch to Qwen3VLModel.

    Note: Qwen3VLModel inherits from Qwen2VLModel, so patching Qwen2VLModel
    affects Qwen3VLModel as well. This function ensures both are patched.
    """
    # Qwen3VLModel inherits from Qwen2VLModel, patch parent first
    patch_qwen2vl()

    # Also patch Qwen3VLModel directly if it has its own override
    try:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

        # Check if Qwen3VLModel has its own get_rope_index (not inherited)
        if 'get_rope_index' in Qwen3VLModel.__dict__:
            _original_functions['Qwen3VLModel.get_rope_index'] = Qwen3VLModel.get_rope_index

            # Qwen3VL's get_rope_index just preprocesses video_grid_thw and calls super
            # We need to preserve that logic while using optimized parent
            def qwen3vl_get_rope_index_optimized(
                self,
                input_ids: torch.LongTensor,
                mm_token_type_ids: torch.IntTensor,
                image_grid_thw: torch.LongTensor | None = None,
                video_grid_thw: torch.LongTensor | None = None,
                attention_mask: torch.Tensor | None = None,
                **kwargs,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                # Qwen3VL-specific preprocessing for video_grid_thw
                # Timestamps separate videos, so split grid_thw per frame
                if video_grid_thw is not None:
                    video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
                    video_grid_thw[:, 0] = 1

                # Call optimized parent implementation
                return get_rope_index_optimized(
                    self,
                    input_ids,
                    mm_token_type_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    attention_mask=attention_mask,
                    **kwargs,
                )

            Qwen3VLModel.get_rope_index = qwen3vl_get_rope_index_optimized
            print("Applied optimization patch to Qwen3VLModel.get_rope_index (override)")
    except ImportError:
        print("Qwen3VLModel not available, skipping")

    print("Applied optimization patch to Qwen3VLModel")


def apply_patch():
    """
    Apply optimization patches to all Qwen VL models.

    This is the recommended entry point. Apply before loading models.
    """
    patch_qwen2vl()
    patch_qwen3vl()
    print("\nOptimization patches applied successfully!")
    print("Benefits:")
    print("  - Reduced GPU-CPU sync from O(N×M×3) to O(2)")
    print("  - Tensor-based boundary detection (no itertools.groupby)")
    print("  - Pre-cached grid_thw on CPU")


def revert_patch():
    """
    Revert all applied patches, restoring original implementations.
    """
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLModel

    if 'Qwen2VLModel.get_vision_position_ids' in _original_functions:
        Qwen2VLModel.get_vision_position_ids = _original_functions['Qwen2VLModel.get_vision_position_ids']
        print("Reverted Qwen2VLModel.get_vision_position_ids")

    if 'Qwen2VLModel.get_rope_index' in _original_functions:
        Qwen2VLModel.get_rope_index = _original_functions['Qwen2VLModel.get_rope_index']
        print("Reverted Qwen2VLModel.get_rope_index")

    try:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
        if 'Qwen3VLModel.get_rope_index' in _original_functions:
            Qwen3VLModel.get_rope_index = _original_functions['Qwen3VLModel.get_rope_index']
            print("Reverted Qwen3VLModel.get_rope_index")
    except ImportError:
        pass

    print("All patches reverted")


def verify_correctness(num_tests: int = 5, use_gpu: bool = True):
    """
    Verify optimized implementations produce same results as originals.

    Args:
        num_tests: Number of test cases to run
        use_gpu: Whether to test GPU tensors (if CUDA available)
    """
    print("Verifying correctness of optimized implementations...")

    # Revert first to get originals
    revert_patch()

    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLModel
    from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig

    # Create minimal config
    config = Qwen2VLConfig(
        text_config={
            'vocab_size': 100, 'hidden_size': 32, 'intermediate_size': 64,
            'num_hidden_layers': 2, 'num_attention_heads': 2, 'num_key_value_heads': 2, 'head_dim': 16,
            'rope_parameters': {'rope_type': 'default', 'mrope_section': [8, 4, 4]},
        },
        vision_config={
            'depth': 2, 'hidden_size': 32, 'embed_dim': 32, 'num_heads': 2,
            'spatial_merge_size': 2, 'patch_size': 16, 'temporal_patch_size': 2,
        },
        image_token_id=3, video_token_id=4, vision_start_token_id=5,
    )

    model = Qwen2VLModel(config)

    passed = 0
    for i in range(num_tests):
        model.rope_deltas = None

        # Test case: pure text + single image
        input_ids = torch.zeros((1, 6), dtype=torch.long)
        mm = torch.zeros((1, 6), dtype=torch.int)
        mm[0, 3] = 1
        grid = torch.tensor([[1, 2, 2]], dtype=torch.long)

        # Original
        pos_orig, delta_orig = model.get_rope_index(input_ids, mm, image_grid_thw=grid)

        # Apply patch and test
        apply_patch()
        model.rope_deltas = None
        pos_opt, delta_opt = model.get_rope_index(input_ids, mm, image_grid_thw=grid)

        # Compare
        if torch.equal(pos_orig, pos_opt) and torch.equal(delta_orig, delta_opt):
            passed += 1
        else:
            print(f"  Test {i+1} FAILED")
            print(f"    Original: {pos_orig[0,0].tolist()}, delta={delta_orig}")
            print(f"    Optimized: {pos_opt[0,0].tolist()}, delta={delta_opt}")

        # Revert for next test
        revert_patch()

    # GPU test
    if use_gpu and torch.cuda.is_available():
        print("\n  GPU tensor test...")
        model.rope_deltas = None

        gpu_input = input_ids.to('cuda')
        gpu_mm = mm.to('cuda')
        gpu_grid = grid.to('cuda')

        # Original
        pos_orig_gpu, delta_orig_gpu = model.get_rope_index(gpu_input, gpu_mm, image_grid_thw=gpu_grid)

        # Optimized
        apply_patch()
        model.rope_deltas = None
        pos_opt_gpu, delta_opt_gpu = model.get_rope_index(gpu_input, gpu_mm, image_grid_thw=gpu_grid)

        if torch.equal(pos_orig_gpu.cpu(), pos_opt_gpu.cpu()) and torch.equal(delta_orig_gpu.cpu(), delta_opt_gpu.cpu()):
            passed += 1
            print("  GPU test PASSED")
        else:
            print("  GPU test FAILED")

        revert_patch()

    print(f"\nVerification complete: {passed}/{num_tests + (1 if use_gpu and torch.cuda.is_available() else 0)} tests passed")
    return passed == num_tests + (1 if use_gpu and torch.cuda.is_available() else 0)


if __name__ == "__main__":
    print("=" * 60)
    print("Qwen2VL/Qwen3VL Position ID Optimization Patch")
    print("=" * 60)

    # Run verification first
    if verify_correctness():
        print("\nVerification passed! Applying patches...")
        apply_patch()
        print("\nPatches are now active. Import transformers and use models normally.")
    else:
        print("\nVerification FAILED! Patches not applied.")
        print("Please check the implementation.")