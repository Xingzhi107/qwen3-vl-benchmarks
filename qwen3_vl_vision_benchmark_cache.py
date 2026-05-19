import torch
import torch.nn.functional as F
import os
from torch.profiler import profile, ProfilerActivity
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel, BaseModelOutputWithDeepstackFeatures
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
import torch.nn as nn
class CustomQwen3VLVisionModelCache1(Qwen3VLVisionModel):
    """
    缓存"重排后单帧"几何参数。

    缓存 key: (h, w)
    缓存 value: (frame_indices, frame_weights)
        frame_indices: [4, h*w]  重排后的角点索引
        frame_weights: [4, h*w]  重排后的双线性权重

    训练时 t 固定，但不同 batch 的 t 可能不同，所以 repeat(t) 不能缓存。
    """

    def __init__(self,config):
        super().__init__(config)
        # self.pos_embed = pos_embed
        # self.num_grid_per_side = num_grid_per_side
        # self.spatial_merge_size = spatial_merge_size
        self._geometry_cache: dict = {}  # key: (h, w) -> (frame_indices, frame_weights)

    def _compute_pos_embed_coords_raw(self, h: int, w: int, device: torch.device):
        """计算"重排前"的原始几何参数"""
        side = self.num_grid_per_side
        merge_size = self.spatial_merge_size

        h_grid = torch.linspace(0, side - 1, h, device=device)
        w_grid = torch.linspace(0, side - 1, w, device=device)

        h_floor = h_grid.long()
        w_floor = w_grid.long()
        h_ceil = (h_floor + 1).clamp(max=side - 1)
        w_ceil = (w_floor + 1).clamp(max=side - 1)

        h_frac = h_grid - h_floor.float()
        w_frac = w_grid - w_floor.float()

        h_floor_offset = h_floor * side
        h_ceil_offset = h_ceil * side

        corner_indices = [
            (h_floor_offset[:, None] + w_floor[None, :]).flatten(),
            (h_floor_offset[:, None] + w_ceil[None, :]).flatten(),
            (h_ceil_offset[:, None] + w_floor[None, :]).flatten(),
            (h_ceil_offset[:, None] + w_ceil[None, :]).flatten(),
        ]
        corner_weights = [
            ((1 - h_frac)[:, None] * (1 - w_frac)[None, :]).flatten(),
            ((1 - h_frac)[:, None] * w_frac[None, :]).flatten(),
            (h_frac[:, None] * (1 - w_frac)[None, :]).flatten(),
            (h_frac[:, None] * w_frac[None, :]).flatten(),
        ]

        h_idx = torch.arange(h, device=device).view(h // merge_size, merge_size)
        w_idx = torch.arange(w, device=device).view(w // merge_size, merge_size)
        reorder = (h_idx[:, :, None, None] * w + w_idx[None, None, :, :]).transpose(1, 2).flatten()

        return corner_indices, corner_weights, reorder

    def _compute_single_frame_reordered(self, h: int, w: int, device: torch.device):
        """计算并返回"重排后"的单帧几何参数"""
        corner_indices, corner_weights, reorder = self._compute_pos_embed_coords_raw(h, w, device)

        # 重排：把 corner_indices[i][reorder] 预先算好
        frame_indices = torch.stack([ci[reorder] for ci in corner_indices])    # [4, h*w]
        frame_weights = torch.stack([cw[reorder] for cw in corner_weights])    # [4, h*w]

        return frame_indices, frame_weights

    def fast_pos_embed_interpolate(self, grid_thw):
        """
        Args:
            grid_thw: [num_images, 3] (t, h, w)
        Returns:
            pos_embeds: [total_tokens, hidden_dim]
        """
        device = grid_thw.device
        target_dtype = self.pos_embed.weight.dtype

        idx_parts = []
        weight_parts = []

        for t, h, w in grid_thw.tolist():
            t, h, w = int(t), int(h), int(w)
            cache_key = (h, w)

            if cache_key not in self._geometry_cache:
                # 首次：计算重排后的单帧，缓存
                frame_indices, frame_weights = self._compute_single_frame_reordered(h, w, device)
                self._geometry_cache[cache_key] = (frame_indices, frame_weights)
            else:
                frame_indices, frame_weights = self._geometry_cache[cache_key]

            # 只剩 repeat(t) —— 不可避免，因为 t 每张图不同
            idx_parts.append(frame_indices.repeat(1, t))      # [4, t*h*w]
            weight_parts.append(frame_weights.repeat(1, t))   # [4, t*h*w]

        # 拼接所有图
        bilinear_indices = torch.cat(idx_parts, dim=1)     # [4, total_tokens]
        bilinear_weights = torch.cat(weight_parts, dim=1)   # [4, total_tokens]

        # 查表（权重训练中会变，不能缓存这步）
        pos_embeds = self.pos_embed(bilinear_indices).to(target_dtype)
        pos_embeds = pos_embeds * bilinear_weights[:, :, None].to(target_dtype)
        return pos_embeds.sum(dim=0)

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        device = grid_thw.device

        # 最大高宽，仅需一次同步
        max_hw = grid_thw[:, 1:].max().item()
        freq_table = self.rotary_pos_emb(max_hw)  # (max_hw, dim // 2)

        # 一次性将数据拉到 CPU，避免循环中多次 item()
        grid_thw_list = grid_thw.tolist()

        coords_list = []
        for num_frames, h, w in grid_thw_list:
            # 直接生成行、列绝对坐标
            rows = torch.arange(h, device=device).repeat_interleave(w)   # (h*w,)
            cols = torch.arange(w, device=device).repeat(h)              # (h*w,)
            coords = torch.stack([rows, cols], dim=-1)                   # (h*w, 2)

            if num_frames > 1:
                # 用 expand 避免复制内存，然后 reshape 为 (num_frames*h*w, 2)
                coords = coords.unsqueeze(0).expand(num_frames, -1, -1).reshape(-1, 2)

            coords_list.append(coords)

        pos_ids = torch.cat(coords_list, dim=0)          # (total_tokens, 2)
        embeddings = freq_table[pos_ids].flatten(1)      # (total_tokens, dim)
        return embeddings
    
    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs):
        # 原生 forward 逻辑，调用重写后的 fast_pos_embed_interpolate
        hidden_states = self.patch_embed(hidden_states)

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            if layer_num in self.deepstack_visual_indexes:
                deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](
                    hidden_states
                )
                deepstack_feature_lists.append(deepstack_feature)

        merged_hidden_states = self.merger(hidden_states)

        return BaseModelOutputWithDeepstackFeatures(
            last_hidden_state=hidden_states,
            pooler_output=merged_hidden_states,
            deepstack_features=deepstack_feature_lists,
        )
class CustomQwen3VLVisionModelCache2(Qwen3VLVisionModel):
    """
    缓存"重排前原始"几何参数。

    缓存 key: (h, w)
    缓存 value: (corner_indices, corner_weights, reorder)
        corner_indices: list of 4 tensors [h*w]  原始角点索引
        corner_weights: list of 4 tensors [h*w]  原始双线性权重
        reorder: [h*w]  spatial merge 重排索引

    推理时 h, w, t 都可能变化，保留 reorder 可以灵活适应任意 t。
    """

    def __init__(self, config):
        super().__init__(config)

        self._geometry_cache: dict = {}  # key: (h, w) -> (corner_indices, corner_weights, reorder)
        
    def _compute_pos_embed_coords(self, h: int, w: int, device: torch.device):
        """计算"重排前"的原始几何参数"""
        side = self.num_grid_per_side
        merge_size = self.spatial_merge_size

        h_grid = torch.linspace(0, side - 1, h, device=device)
        w_grid = torch.linspace(0, side - 1, w, device=device)

        h_floor = h_grid.long()
        w_floor = w_grid.long()
        h_ceil = (h_floor + 1).clamp(max=side - 1)
        w_ceil = (w_floor + 1).clamp(max=side - 1)

        h_frac = h_grid - h_floor.float()
        w_frac = w_grid - w_floor.float()

        h_floor_offset = h_floor * side
        h_ceil_offset = h_ceil * side

        corner_indices = [
            (h_floor_offset[:, None] + w_floor[None, :]).flatten(),
            (h_floor_offset[:, None] + w_ceil[None, :]).flatten(),
            (h_ceil_offset[:, None] + w_floor[None, :]).flatten(),
            (h_ceil_offset[:, None] + w_ceil[None, :]).flatten(),
        ]
        corner_weights = [
            ((1 - h_frac)[:, None] * (1 - w_frac)[None, :]).flatten(),
            ((1 - h_frac)[:, None] * w_frac[None, :]).flatten(),
            (h_frac[:, None] * (1 - w_frac)[None, :]).flatten(),
            (h_frac[:, None] * w_frac[None, :]).flatten(),
        ]

        h_idx = torch.arange(h, device=device).view(h // merge_size, merge_size)
        w_idx = torch.arange(w, device=device).view(w // merge_size, merge_size)
        reorder = (h_idx[:, :, None, None] * w + w_idx[None, None, :, :]).transpose(1, 2).flatten()

        return corner_indices, corner_weights, reorder


    def fast_pos_embed_interpolate(self, grid_thw):
        """
        Args:
            grid_thw: [num_images, 3] (t, h, w)
        Returns:
            pos_embeds: [total_tokens, hidden_dim]
        """
        device = grid_thw.device
        target_dtype = self.pos_embed.weight.dtype

        idx_parts = [[] for _ in range(4)]
        weight_parts = [[] for _ in range(4)]

        for t, h, w in grid_thw.tolist():
            t, h, w = int(t), int(h), int(w)
            cache_key = (h, w)

            if cache_key not in self._geometry_cache:
                # 首次：计算原始几何参数，缓存
                corner_indices, corner_weights, reorder = self._compute_pos_embed_coords(h, w, device)
                self._geometry_cache[cache_key] = (corner_indices, corner_weights, reorder)
            else:
                corner_indices, corner_weights, reorder = self._geometry_cache[cache_key]

            # 每次都要：重排 + 重复（适应任意 t）
            for i in range(4):
                idx_parts[i].append(corner_indices[i][reorder].repeat(t))
                weight_parts[i].append(corner_weights[i][reorder].repeat(t))

        # 拼接
        bilinear_indices = torch.stack([torch.cat(p) for p in idx_parts])     # [4, total_tokens]
        bilinear_weights = torch.stack([torch.cat(p) for p in weight_parts])   # [4, total_tokens]

        # 查表
        pos_embeds = self.pos_embed(bilinear_indices).to(target_dtype)
        pos_embeds = pos_embeds * bilinear_weights[:, :, None].to(target_dtype)
        return pos_embeds.sum(dim=0)

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs):
        # 原生 forward 逻辑，调用重写后的 fast_pos_embed_interpolate
        hidden_states = self.patch_embed(hidden_states)

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            if layer_num in self.deepstack_visual_indexes:
                deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](
                    hidden_states
                )
                deepstack_feature_lists.append(deepstack_feature)

        merged_hidden_states = self.merger(hidden_states)

        return BaseModelOutputWithDeepstackFeatures(
            last_hidden_state=hidden_states,
            pooler_output=merged_hidden_states,
            deepstack_features=deepstack_feature_lists,
        )
def run_benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

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

    model = CustomQwen3VLVisionModelCache1(config).to(device).to(dtype)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # 测试高分辨率 (1008x1008)
    T, H, W = 2, 1008, 1008
    grid_thw = torch.tensor([[T // 2, H // 14, W // 14]], device=device)
    num_patches = (T // 2) * (H // 14) * (W // 14)
    pixel_values = torch.randn(num_patches, 3, 2, 14, 14, device=device, dtype=dtype)

    print("Warming up (Your Optimization Version)...")
    for _ in range(3):
        optimizer.zero_grad()
        output = model(pixel_values, grid_thw)
        output.last_hidden_state.sum().backward()

    print("Starting profiling for Optimized Version...")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as prof:
        for _ in range(5):
            optimizer.zero_grad()
            output = model(pixel_values, grid_thw)
            output.last_hidden_state.sum().backward()
            if device == "cuda": torch.cuda.synchronize()

    prof.export_chrome_trace("qwen3_vl_vision_optimized_trace.json")
    print("\nOptimized Trace saved: qwen3_vl_vision_optimized_trace.json")

if __name__ == "__main__":
    run_benchmark()
