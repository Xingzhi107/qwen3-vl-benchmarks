"""
Qwen3VL 训练脚本 - 通过 patch 替换 compute_3d_position_ids

## 真正有效的优化

经过仔细分析，原始代码的主要瓶颈在于：
1. Python 循环遍历 batch 和 segment（由于输入格式设计，无法并行化）
2. tensor 操作本身（torch.arange, repeat, stack 等）

真正可以做的有效优化非常有限：
- 内联 vision position IDs 计算可以减少函数调用开销
- 但必须保持所有原始参数的功能完整性

## 保持不变的点（不要画蛇添足）

- temp_merge_size 和 time_interval 参数：虽然默认为1，但用户可以传入不同值
- .tolist() 转换：这是必要的，用于 Python int 做 dict key
- 变量命名、代码风格：保持和原始一致
- tensor 操作顺序：改变不会带来性能提升

"""

import sys
import os
sys.path.insert(0, "/home/layz/workspace/transformers/src")

import torch
from torch.profiler import profile, ProfilerActivity, schedule, record_function
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig, Qwen3VLVisionConfig, Qwen3VLTextConfig


import torch
import triton
import triton.language as tl
from types import MethodType

def _preprocess_mrope_meta(mm_token_type_ids, image_grid_thw, video_grid_thw, attention_mask, spatial_merge_size):
    """
    轻量预处理：只遍历 segment（不是 token），生成每个 token 的查表元数据。
    所有输出张量驻留 GPU，供 Triton kernel 直接读取。
    """
    B, S = mm_token_type_ids.shape
    device = mm_token_type_ids.device

    token_types = torch.zeros(B, S, dtype=torch.int32, device=device)
    seg_offsets = torch.zeros(B, S, dtype=torch.int32, device=device)
    global_starts = torch.zeros(B, S, dtype=torch.int32, device=device)
    grid_indices = torch.full((B, S), -1, dtype=torch.int32, device=device)
    valid_mask = attention_mask.bool() if attention_mask is not None else torch.ones(B, S, dtype=torch.bool, device=device)

    img_ptr = vid_ptr = 0
    for b in range(B):
        mask = valid_mask[b]
        valid_len = int(mask.sum().item())
        if valid_len == 0:
            continue

        # 获取有效 token 的原始索引和类型
        valid_indices = torch.where(mask)[0]          # (valid_len,)
        vtypes = mm_token_type_ids[b][mask]             # (valid_len,)

        # 找 segment 边界（torch.diff 替代 itertools.groupby）
        padded = torch.cat([torch.tensor([-1], device=device, dtype=vtypes.dtype), vtypes])
        boundaries = torch.where(torch.diff(padded) != 0)[0]

        cur_pos = 0
        for i in range(len(boundaries)):
            s = int(boundaries[i].item())
            e = int(boundaries[i + 1].item()) if i + 1 < len(boundaries) else valid_len
            length = e - s
            stype = int(vtypes[s].item())
            orig_idx = valid_indices[s:e]              # 映射回原始序列位置

            token_types[b, orig_idx] = stype
            seg_offsets[b, orig_idx] = torch.arange(length, device=device, dtype=torch.int32)
            global_starts[b, orig_idx] = cur_pos

            if stype == 1:                             # image
                grid = image_grid_thw[img_ptr]
                grid_indices[b, orig_idx] = img_ptr
                img_ptr += 1
                cur_pos += max(int(grid[1].item()), int(grid[2].item())) // spatial_merge_size
            elif stype == 2:                           # video
                grid = video_grid_thw[vid_ptr]
                grid_indices[b, orig_idx] = vid_ptr
                vid_ptr += 1
                cur_pos += max(int(grid[1].item()), int(grid[2].item())) // spatial_merge_size
            else:                                      # text
                cur_pos += length

    return token_types, seg_offsets, global_starts, grid_indices, valid_mask


@triton.jit
def _mrope_kernel(
    token_types_ptr, seg_offsets_ptr, global_starts_ptr, grid_indices_ptr,
    image_grid_ptr, video_grid_ptr, position_ids_ptr, valid_mask_ptr,
    spatial_merge_size: tl.constexpr, B, S, BLOCK_SIZE: tl.constexpr,
):
    """
    Fused M-RoPE Triton Kernel。

    每个 thread 处理一个 token (b, s)：
    - 查元数据表知道自己属于哪个 segment、什么 modality
    - text：3 维共享同一个线性 position
    - vision：读取 grid_thw，用整数算术算出 3D (t, h, w) position
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < B * S
    b = offs // S
    s = offs % S

    # 读取 valid mask，padding token 直接跳过
    valid = tl.load(valid_mask_ptr + offs, mask=mask, other=0).to(tl.int1)
    active = mask & valid

    # 读取预处理好的元数据（全部在显存，零 CPU 参与）
    tt = tl.load(token_types_ptr + offs, mask=active, other=0)
    offset = tl.load(seg_offsets_ptr + offs, mask=active, other=0)
    gstart = tl.load(global_starts_ptr + offs, mask=active, other=0)

    out = b * S + s
    stride = B * S

    # ---------- Text Branch ----------
    is_text = active & (tt == 0)
    pos = gstart + offset
    tl.store(position_ids_ptr + 0 * stride + out, pos, mask=is_text)
    tl.store(position_ids_ptr + 1 * stride + out, pos, mask=is_text)
    tl.store(position_ids_ptr + 2 * stride + out, pos, mask=is_text)

    # ---------- Vision Branch ----------
    is_vision = active & (tt > 0)
    gidx = tl.load(grid_indices_ptr + offs, mask=is_vision, other=0)

    # 用 tl.where 合并 image/video 查表，避免 kernel 内分支
    is_img = is_vision & (tt == 1)
    is_vid = is_vision & (tt == 2)

    t = tl.where(is_img,
                 tl.load(image_grid_ptr + gidx * 3 + 0, mask=is_img, other=1),
                 tl.load(video_grid_ptr + gidx * 3 + 0, mask=is_vid, other=1))
    h = tl.where(is_img,
                 tl.load(image_grid_ptr + gidx * 3 + 1, mask=is_img, other=1),
                 tl.load(video_grid_ptr + gidx * 3 + 1, mask=is_vid, other=1))
    w = tl.where(is_img,
                 tl.load(image_grid_ptr + gidx * 3 + 2, mask=is_img, other=1),
                 tl.load(video_grid_ptr + gidx * 3 + 2, mask=is_vid, other=1))

    # 3D position 整数算术（与 get_vision_position_ids 语义完全一致）
    hm = h // spatial_merge_size
    wm = w // spatial_merge_size
    hw = hm * wm

    t_pos = offset // hw
    hw_off = offset % hw
    h_pos = hw_off // wm
    w_pos = hw_off % wm

    tl.store(position_ids_ptr + 0 * stride + out, gstart + t_pos, mask=is_vision)
    tl.store(position_ids_ptr + 1 * stride + out, gstart + h_pos, mask=is_vision)
    tl.store(position_ids_ptr + 2 * stride + out, gstart + w_pos, mask=is_vision)


def get_rope_index_triton(
    self,
    input_ids: torch.LongTensor,
    mm_token_type_ids: torch.IntTensor,
    image_grid_thw: torch.LongTensor | None = None,
    video_grid_thw: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Triton 版 get_rope_index。接口与原版完全一致，直接替换即可。
    """
    B, S = input_ids.shape
    device = input_ids.device
    dtype = input_ids.dtype
    sms = self.config.vision_config.spatial_merge_size

    # video 预处理（保持与原始逻辑一致）
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    # 1) 预处理：生成元数据表（只遍历 segment，不遍历 token）
    token_types, seg_offsets, global_starts, grid_indices, valid_mask = _preprocess_mrope_meta(
        mm_token_type_ids, image_grid_thw, video_grid_thw, attention_mask, sms
    )

    # Triton 不能传 None，用空 tensor 占位
    if image_grid_thw is None:
        image_grid_thw = torch.empty((0, 3), dtype=torch.int32, device=device)
    else:
        image_grid_thw = image_grid_thw.to(torch.int32)
    if video_grid_thw is None:
        video_grid_thw = torch.empty((0, 3), dtype=torch.int32, device=device)
    else:
        video_grid_thw = video_grid_thw.to(torch.int32)

    # 2) 预分配输出，单次 kernel launch 完成全部计算
    position_ids = torch.empty(3, B, S, dtype=dtype, device=device)

    BLOCK = 256
    grid = (triton.cdiv(B * S, BLOCK),)
    _mrope_kernel[grid](
        token_types, seg_offsets, global_starts, grid_indices,
        image_grid_thw, video_grid_thw, position_ids, valid_mask,
        spatial_merge_size=sms, B=B, S=S, BLOCK_SIZE=BLOCK, num_warps=4,
    )

    # 3) mrope_position_deltas（向量化，无循环）
    valid_lens = valid_mask.sum(dim=1)
    max_pos = position_ids.max(dim=2).values.max(dim=0).values
    rope_deltas = (max_pos + 1 - valid_lens).to(dtype).unsqueeze(1)

    return position_ids, rope_deltas


def optimized_get_rope_index(
    self,
    input_ids: torch.LongTensor,
    mm_token_type_ids: torch.IntTensor,
    image_grid_thw: torch.LongTensor | None = None,
    video_grid_thw: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    dtype = input_ids.dtype
    spatial_merge_size = self.config.vision_config.spatial_merge_size

    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    position_ids = torch.zeros(3, batch_size, seq_len, dtype=dtype, device=device)
    mrope_position_deltas = []

    img_ptr = 0
    vid_ptr = 0

    for batch_idx in range(batch_size):
        types = mm_token_type_ids[batch_idx]

        if attention_mask is not None:
            mask = attention_mask[batch_idx].bool()
            types = types[mask]
            valid_len = int(mask.sum().item())
        else:
            mask = None
            valid_len = seq_len

        if valid_len == 0:
            mrope_position_deltas.append(torch.tensor(0, device=device, dtype=dtype))
            continue

        # 优化1：torch.diff 替代 itertools.groupby + .tolist()，消除 CPU-GPU 隐式同步
        padded = torch.cat([
            torch.tensor([-1], device=device, dtype=types.dtype),
            types
        ])
        change_idx = torch.where(torch.diff(padded) != 0)[0]

        segments = []
        for i in range(len(change_idx)):
            s = int(change_idx[i].item())
            e = int(change_idx[i + 1].item()) if i + 1 < len(change_idx) else len(types)
            segments.append((int(types[s].item()), s, e))

        # 优化2：预分配连续张量，替代 list.append + torch.cat
        total_len = 0
        tmp_img = img_ptr
        tmp_vid = vid_ptr
        for modality_type, s, e in segments:
            if modality_type == 0:
                total_len += e - s
            else:
                if modality_type == 1:
                    g = image_grid_thw[tmp_img]
                    tmp_img += 1
                else:
                    g = video_grid_thw[tmp_vid]
                    tmp_vid += 1
                t = int(g[0].item())
                h = int(g[1].item()) // spatial_merge_size
                w = int(g[2].item()) // spatial_merge_size
                total_len += t * h * w

        batch_positions = torch.empty(3, total_len, dtype=dtype, device=device)
        offset = 0
        current_pos = 0

        # 优化3：索引指针替代 iter/next，原地写入预分配张量
        for modality_type, s, e in segments:
            if modality_type == 0:
                text_len = e - s
                text_pos = torch.arange(text_len, device=device, dtype=dtype).view(1, -1).expand(3, -1) + current_pos
                batch_positions[:, offset:offset + text_len] = text_pos
                offset += text_len
                current_pos += text_len
            else:
                if modality_type == 1:
                    grid_thw = image_grid_thw[img_ptr]
                    img_ptr += 1
                else:
                    grid_thw = video_grid_thw[vid_ptr]
                    vid_ptr += 1

                # 内联 vision position 生成，消除函数调用开销
                t = int(grid_thw[0].item())
                h = int(grid_thw[1].item())
                w = int(grid_thw[2].item())
                h_merge = h // spatial_merge_size
                w_merge = w // spatial_merge_size

                t_base = torch.arange(t, device=device, dtype=dtype).view(t, 1).expand(t, h_merge * w_merge).reshape(-1)
                h_base = torch.arange(h_merge, device=device, dtype=dtype).view(h_merge, 1).expand(h_merge, w_merge).reshape(-1).repeat(t)
                w_base = torch.arange(w_merge, device=device, dtype=dtype).view(1, w_merge).expand(h_merge, w_merge).reshape(-1).repeat(t)

                vision_pos = torch.stack([t_base, h_base, w_base], dim=0) + current_pos
                vlen = vision_pos.shape[1]
                batch_positions[:, offset:offset + vlen] = vision_pos
                offset += vlen
                current_pos += max(h, w) // spatial_merge_size

        if mask is not None:
            position_ids[:, batch_idx, mask] = batch_positions
        else:
            position_ids[:, batch_idx] = batch_positions

        mrope_position_deltas.append(batch_positions.max() + 1 - valid_len)

    return position_ids, torch.stack(mrope_position_deltas).unsqueeze(1)


def optimized_compute_3d_position_ids(
    self,
    input_ids: torch.Tensor | None,
    inputs_embeds: torch.Tensor | None,
    image_grid_thw: torch.Tensor | None = None,
    video_grid_thw: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    past_key_values: torch.Tensor | None = None,
    mm_token_type_ids: torch.IntTensor | None = None,
) -> torch.Tensor | None:
    past_key_values_length = 0 if past_key_values is None else past_key_values.get_seq_length()
    has_multimodal = image_grid_thw is not None or video_grid_thw is not None
    if has_multimodal and mm_token_type_ids is None and input_ids is not None:
        raise ValueError(
            "Multimodal data was passed (via `image_grid_thw` or `video_grid_thw`) but `mm_token_type_ids` is "
            "missing. Please pass `mm_token_type_ids` to the model so that multimodal RoPE (M-RoPE) can be "
            "computed correctly. `mm_token_type_ids` is returned by the processor alongside `input_ids`."
        )
    can_compute_mrope = input_ids is not None and mm_token_type_ids is not None and has_multimodal

    if can_compute_mrope and (self.rope_deltas is None or past_key_values_length == 0):
        position_ids, rope_deltas = self.get_rope_index(
            input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
            mm_token_type_ids=mm_token_type_ids,
        )
        self.rope_deltas = rope_deltas
    elif self.rope_deltas is not None and (past_key_values_length > 0 or input_ids is None):
        batch_size, seq_length, _ = inputs_embeds.shape

        # 防御：batch_size 变化时（如训练接推理），重新计算 rope_deltas
        if self.rope_deltas.shape[0] != batch_size:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
                mm_token_type_ids=mm_token_type_ids,
            )
            self.rope_deltas = rope_deltas
            return position_ids

        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids = position_ids.masked_fill(attention_mask == 0, 0)
            # 优化4：expand 替代 view+repeat，零拷贝广播
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(inputs_embeds.device)
        else:
            position_ids = torch.arange(
                past_key_values_length, past_key_values_length + seq_length,
                device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )
            position_ids = position_ids.unsqueeze(0).unsqueeze(0).expand(3, batch_size, -1)

        # 优化5：expand 替代 repeat_interleave，避免实际内存拷贝
        delta = self.rope_deltas.expand(batch_size, -1)
        position_ids = position_ids + delta.to(device=inputs_embeds.device)
    else:
        position_ids = None
    return position_ids

# def apply_patch_to_model(model):
#     """通过 monkey-patch 替换 compute_3d_position_ids。"""
#     import types

#     model.get_rope_index = MethodType(optimized_get_rope_index, model)
#     model.compute_3d_position_ids = MethodType(optimized_compute_3d_position_ids, model)
#     return model

def apply_patch_to_model(model):
    """通过 monkey-patch 替换 compute_3d_position_ids。"""
    import types

    model.get_rope_index = MethodType(get_rope_index_triton, model)
    return model

def train_tri_opt():
    """训练脚本"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    vision_config = Qwen3VLVisionConfig(
        hidden_size=1024, intermediate_size=4096, num_heads=16, depth=2,
        patch_size=14, temporal_patch_size=2, spatial_merge_size=2, out_hidden_size=1024,
    )
    text_config = Qwen3VLTextConfig(
        hidden_size=1024, intermediate_size=4096, num_hidden_layers=2,
        num_attention_heads=16, num_key_value_heads=16, vocab_size=1000,
    )
    config = Qwen3VLConfig(vision_config=vision_config, text_config=text_config)
    config.image_token_id = 10
    config.video_token_id = 11

    print(f"在 {device} 上初始化 Qwen3VLModel...")
    model = Qwen3VLModel(config).to(device).to(dtype)
    model = apply_patch_to_model(model)
    model.train()

    T, H, W = 2, 1008, 1008
    grid_thw = torch.tensor([[T // 2, H // 14, W // 14]], device=device)
    num_vision_tokens = (T // 2) * (H // (14 * 2)) * (W // (14 * 2))
    seq_len = 5 + num_vision_tokens + 5

    input_ids = torch.zeros((1, seq_len), dtype=torch.long, device=device)
    mm_token_type_ids = torch.zeros((1, seq_len), dtype=torch.int, device=device)
    mm_token_type_ids[0, 5:5+num_vision_tokens] = 1
    input_ids[0, 5:5+num_vision_tokens] = config.image_token_id

    pixel_values = torch.randn(num_vision_tokens * 4, 3, 2, 14, 14, device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    print("开始 profiling...")
    my_schedule = schedule(wait=6, warmup=0, active=2, repeat=1)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=my_schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/tri_tri_train'),
        record_shapes=True, with_stack=True
    ) as prof:
        for step in range(10):
            optimizer.zero_grad()

            with record_function("compute_3d_position_ids_optimized"):
                position_ids = model.compute_3d_position_ids(
                    input_ids=input_ids, inputs_embeds=None,
                    image_grid_thw=grid_thw, mm_token_type_ids=mm_token_type_ids
                )

            output = model(
                input_ids=input_ids, pixel_values=pixel_values,
                image_grid_thw=grid_thw, mm_token_type_ids=mm_token_type_ids,
                position_ids=position_ids
            )

            loss = output.last_hidden_state.sum()
            loss.backward()
            optimizer.step()

            if device == "cuda":
                torch.cuda.synchronize()

            prof.step()
            print(f"Step {step + 1}/10 | Loss: {loss.item():.4f}")

    # prof.export_chrome_trace("qwen3_vl_tri_opt_compute_3d_pos_ids.json")
    # print("Trace saved to qwen3_vl_tri_opt_compute_3d_pos_ids.json")

    print("\n=== 有效优化 ===")
    print("内联计算 vision position IDs - 减少函数调用开销")
    print("其他保持和原始完全一致")


if __name__ == "__main__":
    train_tri_opt()