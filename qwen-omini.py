from __future__ import annotations

import torch
import torch.nn.functional as F

from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as model_module
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeAudioEncoder, _get_feat_extract_output_lengths


class OptimizedQwen3OmniMoeAudioEncoder(Qwen3OmniMoeAudioEncoder):
    """Optimized wrapper that replaces Python list chunking with batched tensor ops."""

    def forward(self, input_features, feature_lens=None, aftercnn_lens=None, **kwargs):
        aftercnn_lens = _get_feat_extract_output_lengths(feature_lens)
        chunk_num = torch.div(feature_lens, self.n_window * 2, rounding_mode="ceil").long()
        chunk_lengths = torch.full((chunk_num.sum(),), self.n_window * 2, dtype=torch.long, device=feature_lens.device)
        tail_chunk_index = torch.cumsum(F.pad(chunk_num, (1, 0), value=-1), dim=0)[1:]
        chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
        chunk_lengths[chunk_lengths == 0] = self.n_window * 2

        chunk_starts = chunk_lengths.cumsum(0) - chunk_lengths
        max_chunk_len = chunk_lengths.max()
        total_length = input_features.shape[-1]
        positions = torch.arange(max_chunk_len, device=chunk_lengths.device)
        gather_idx = chunk_starts.unsqueeze(1) + positions.unsqueeze(0)
        gather_idx = torch.minimum(gather_idx, (total_length - 1))

        chunked = input_features.T[gather_idx]
        mask = positions.unsqueeze(0) < chunk_lengths.unsqueeze(1)
        chunked = chunked.permute(0, 2, 1)

        padded_feature = chunked.unsqueeze(1)
        padded_embed = F.gelu(self.conv2d1(padded_feature))
        padded_embed = F.gelu(self.conv2d2(padded_embed))
        padded_embed = F.gelu(self.conv2d3(padded_embed))
        b, c, f, t = padded_embed.size()
        padded_embed = self.conv_out(padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))

        positional_embedding = (
            self.positional_embedding.positional_embedding[: padded_embed.shape[1], :]
            .unsqueeze(0)
            .to(padded_embed.dtype)
        )
        padded_embed = padded_embed + positional_embedding

        feature_lens_after_cnn = _get_feat_extract_output_lengths(chunk_lengths)
        max_len_after_cnn = feature_lens_after_cnn.max()
        after_cnn_positions = torch.arange(max_len_after_cnn, device=feature_lens.device)
        padded_mask_after_cnn = after_cnn_positions.unsqueeze(0) < feature_lens_after_cnn.unsqueeze(1)
        hidden_states = padded_embed[padded_mask_after_cnn]

        cu_chunk_lens = _build_cu_chunk_lens(aftercnn_lens, self.n_window_infer, self.n_window, max_len_after_cnn)
        cu_seqlens = cu_chunk_lens.cumsum(-1, dtype=torch.int32)

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states, cu_seqlens)[0]

        hidden_states = self.ln_post(hidden_states)
        hidden_states = self.proj1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.proj2(hidden_states)
        return BaseModelOutputWithPooling(last_hidden_state=hidden_states)


def _build_cu_chunk_lens(aftercnn_lens: torch.Tensor, n_window_infer: int, n_window: int, max_len_after_cnn: int) -> torch.Tensor:
    window_aftercnn = max_len_after_cnn * (n_window_infer // (n_window * 2))
    if window_aftercnn == 0:
        return torch.tensor([0], device=aftercnn_lens.device)

    chunk_counts = torch.div(aftercnn_lens, window_aftercnn, rounding_mode="floor")
    remainder = aftercnn_lens % window_aftercnn
    chunks = []
    for count, rem in zip(chunk_counts.tolist(), remainder.tolist()):
        if count:
            chunks.extend([window_aftercnn] * count)
        if rem:
            chunks.append(rem)
    return torch.tensor([0] + chunks, device=aftercnn_lens.device)


def apply_qwen3_omni_moe_audio_encoder_patch() -> None:
    model_module.Qwen3OmniMoeAudioEncoder = OptimizedQwen3OmniMoeAudioEncoder
