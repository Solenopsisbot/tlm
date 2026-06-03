from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

ArchitectureKind = Literal["stream", "swift", "fastconv", "conv", "rnn"]
RecurrentKind = Literal["gated", "gru"]


@dataclass(frozen=True)
class TinyLanguageModelConfig:
    vocab_size: int = 256
    dim: int = 128
    layers: int = 2
    conv_kernel: int = 5
    dropout: float = 0.0
    architecture: ArchitectureKind = "stream"
    recurrent: RecurrentKind = "gru"
    state_kernel: int = 3
    max_dilation: int = 8192
    mlp_expansion: int = 2
    state_heads: int = 4
    state_rank: int = 16


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight


class CausalDepthwiseConv(nn.Module):
    def __init__(self, dim: int, kernel_size: int, dilation: int = 1) -> None:
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size must be at least 1")
        if dilation < 1:
            raise ValueError("dilation must be at least 1")
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.depthwise = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            groups=dim,
            dilation=dilation,
            bias=True,
        )
        self.pointwise = nn.Linear(dim, dim)

    @property
    def cache_len(self) -> int:
        return (self.kernel_size - 1) * self.dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kernel_size == 1:
            return self.pointwise(x)

        y = x.transpose(1, 2)
        y = F.pad(y, (self.cache_len, 0))
        y = self.depthwise(y).transpose(1, 2)
        return self.pointwise(y)

    def step(
        self,
        x: torch.Tensor,
        cache: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.kernel_size == 1:
            return self.pointwise(x), None

        batch, dim = x.shape
        if cache is None:
            cache = x.new_zeros(batch, dim, self.cache_len)

        window = torch.cat([cache, x.unsqueeze(-1)], dim=-1)
        taps = window[:, :, :: self.dilation]
        weight = self.depthwise.weight[:, 0, :].unsqueeze(0)
        y = (taps * weight).sum(dim=-1)
        if self.depthwise.bias is not None:
            y = y + self.depthwise.bias
        return self.pointwise(y), window[:, :, 1:]


class CausalDepthwiseOnlyConv(nn.Module):
    def __init__(self, dim: int, kernel_size: int, dilation: int = 1) -> None:
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size must be at least 1")
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.depthwise = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            groups=dim,
            dilation=dilation,
            bias=True,
        )

    @property
    def cache_len(self) -> int:
        return (self.kernel_size - 1) * self.dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kernel_size == 1:
            return x
        y = x.transpose(1, 2)
        y = F.pad(y, (self.cache_len, 0))
        return self.depthwise(y).transpose(1, 2)

    def step(
        self,
        x: torch.Tensor,
        cache: object | None = None,
    ) -> tuple[torch.Tensor, object | None]:
        if self.kernel_size == 1:
            return x, None
        batch, dim = x.shape
        if cache is None:
            cache_tensor = x.new_zeros(batch, dim, self.cache_len)
            position = 0
        else:
            cache_tensor, position = cache

        taps = []
        for index in range(self.kernel_size - 1):
            taps.append(cache_tensor[:, :, (position + index * self.dilation) % self.cache_len])
        taps.append(x)
        tap_tensor = torch.stack(taps, dim=-1)
        weight = self.depthwise.weight[:, 0, :].unsqueeze(0)
        y = (tap_tensor * weight).sum(dim=-1)
        if self.depthwise.bias is not None:
            y = y + self.depthwise.bias
        cache_tensor[:, :, position] = x
        position = (position + 1) % self.cache_len
        return y, (cache_tensor, position)


class GatedRecurrentLayer(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.in_norm = RMSNorm(dim)
        self.state_norm = RMSNorm(dim)
        self.input_proj = nn.Linear(dim, dim * 3)
        self.state_proj = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def step(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        gates = self.input_proj(self.in_norm(x)) + self.state_proj(self.state_norm(state))
        update, reset, candidate = gates.chunk(3, dim=-1)
        update = torch.sigmoid(update)
        reset = torch.sigmoid(reset)
        candidate = torch.tanh(candidate + reset * state)
        return state + update * (candidate - state)

    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, dim = x.shape
        if state is None:
            state = x.new_zeros(batch, dim)

        outputs = []
        current = state
        for token in x.unbind(dim=1):
            current = self.step(token, current)
            outputs.append(current)
        y = torch.stack(outputs, dim=1)
        return x + self.dropout(self.out_proj(y)), current


class GatedRecurrentStack(nn.Module):
    def __init__(self, dim: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            GatedRecurrentLayer(dim, dropout if index > 0 else 0.0)
            for index in range(layers)
        )

    def forward(
        self,
        x: torch.Tensor,
        state: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        next_state = []
        for index, layer in enumerate(self.layers):
            layer_state = None if state is None else state[index]
            x, layer_state = layer(x, layer_state)
            next_state.append(layer_state)
        return x, next_state


class CumulativeStateMixer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, in_gate, out_gate = self.in_proj(x).chunk(3, dim=-1)
        value = torch.tanh(value) * torch.sigmoid(in_gate)
        position = torch.arange(1, x.shape[1] + 1, device=x.device, dtype=x.dtype)
        position = position.view(1, -1, 1)
        state = torch.cumsum(value, dim=1) * torch.rsqrt(position)
        return self.out_proj(state * torch.sigmoid(out_gate))

    def step(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if state is None:
            running_sum = x.new_zeros(x.shape)
            count = x.new_zeros(1)
        else:
            running_sum, count = state
        value, in_gate, out_gate = self.in_proj(x).chunk(3, dim=-1)
        value = torch.tanh(value) * torch.sigmoid(in_gate)
        running_sum = running_sum + value
        count = count + 1
        normalized = running_sum * torch.rsqrt(count)
        return self.out_proj(normalized * torch.sigmoid(out_gate)), (running_sum, count)


class ConvStateBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        short_kernel: int,
        state_kernel: int,
        dilation: int,
        mlp_expansion: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.short_norm = RMSNorm(dim)
        self.short_conv = CausalDepthwiseConv(dim, short_kernel)
        self.long_norm = RMSNorm(dim)
        self.long_conv = CausalDepthwiseConv(dim, state_kernel, dilation=dilation)
        self.long_gate = nn.Linear(dim, dim)
        self.state_norm = RMSNorm(dim)
        self.state_mixer = CumulativeStateMixer(dim)
        hidden = dim * mlp_expansion
        self.mlp_norm = RMSNorm(dim)
        self.mlp_in = nn.Linear(dim, hidden * 2)
        self.mlp_out = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.short_conv(self.short_norm(x)))
        long_input = self.long_norm(x)
        x = x + self.dropout(self.long_conv(long_input) * torch.sigmoid(self.long_gate(long_input)))
        x = x + self.dropout(self.state_mixer(self.state_norm(x)))
        value, gate = self.mlp_in(self.mlp_norm(x)).chunk(2, dim=-1)
        x = x + self.dropout(self.mlp_out(F.silu(gate) * value))
        return x

    def step(
        self,
        x: torch.Tensor,
        cache: dict[str, object] | None = None,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        if cache is None:
            cache = {"short": None, "long": None, "state": None}

        short_input = self.short_norm(x)
        y, short_cache = self.short_conv.step(short_input, cache.get("short"))
        x = x + y
        long_input = self.long_norm(x)
        y, long_cache = self.long_conv.step(long_input, cache.get("long"))
        x = x + y * torch.sigmoid(self.long_gate(long_input))
        y, state_cache = self.state_mixer.step(self.state_norm(x), cache.get("state"))
        x = x + y
        value, gate = self.mlp_in(self.mlp_norm(x)).chunk(2, dim=-1)
        x = x + self.mlp_out(F.silu(gate) * value)
        return x, {"short": short_cache, "long": long_cache, "state": state_cache}


class ConvStateStack(nn.Module):
    def __init__(self, config: TinyLanguageModelConfig) -> None:
        super().__init__()
        dilations = build_dilations(config.layers, config.max_dilation)
        self.layers = nn.ModuleList(
            ConvStateBlock(
                dim=config.dim,
                short_kernel=config.conv_kernel,
                state_kernel=config.state_kernel,
                dilation=dilations[index],
                mlp_expansion=config.mlp_expansion,
                dropout=config.dropout,
            )
            for index in range(config.layers)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        for layer in self.layers:
            x = layer(x)
        return x, None

    def step(
        self,
        x: torch.Tensor,
        cache: list[dict[str, object]] | None = None,
    ) -> tuple[torch.Tensor, list[dict[str, object]]]:
        next_cache = []
        for index, layer in enumerate(self.layers):
            layer_cache = None if cache is None else cache[index]
            x, layer_cache = layer.step(x, layer_cache)
            next_cache.append(layer_cache)
        return x, next_cache


class StreamMemoryBlock(nn.Module):
    def __init__(self, config: TinyLanguageModelConfig) -> None:
        super().__init__()
        dim = config.dim
        hidden = dim * config.mlp_expansion
        self.mix = nn.Parameter(torch.zeros(dim))
        self.memory_norm = RMSNorm(dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.receptance = nn.Linear(dim, dim)
        self.memory_out = nn.Linear(dim, dim)
        self.mlp_norm = RMSNorm(dim)
        self.mlp_in = nn.Linear(dim, hidden * 2)
        self.mlp_out = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prev = F.pad(x[:, :-1], (0, 0, 1, 0))
        mixed = x + torch.sigmoid(self.mix).view(1, 1, -1) * (prev - x)
        h = self.memory_norm(mixed)
        key = F.softplus(self.key(h)) + 1e-4
        value = torch.tanh(self.value(h))
        numerator = torch.cumsum(key * value, dim=1)
        denominator = torch.cumsum(key, dim=1)
        memory = numerator / denominator.clamp_min(1e-4)
        memory = self.memory_out(memory * torch.sigmoid(self.receptance(h)))
        x = x + self.dropout(memory)

        value, gate = self.mlp_in(self.mlp_norm(x)).chunk(2, dim=-1)
        return x + self.dropout(self.mlp_out(value * F.silu(gate)))

    def step(
        self,
        x: torch.Tensor,
        cache: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if cache is None:
            prev = x.new_zeros(x.shape)
            numerator = x.new_zeros(x.shape)
            denominator = x.new_zeros(x.shape)
        else:
            prev = cache["prev"]
            numerator = cache["numerator"]
            denominator = cache["denominator"]

        current = x
        mixed = x + torch.sigmoid(self.mix).view(1, -1) * (prev - x)
        h = self.memory_norm(mixed)
        key = F.softplus(self.key(h)) + 1e-4
        value = torch.tanh(self.value(h))
        numerator = numerator + key * value
        denominator = denominator + key
        memory = numerator / denominator.clamp_min(1e-4)
        x = x + self.memory_out(memory * torch.sigmoid(self.receptance(h)))

        value, gate = self.mlp_in(self.mlp_norm(x)).chunk(2, dim=-1)
        x = x + self.mlp_out(value * F.silu(gate))
        return x, {"prev": current, "numerator": numerator, "denominator": denominator}


class StreamStack(nn.Module):
    def __init__(self, config: TinyLanguageModelConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(StreamMemoryBlock(config) for _ in range(config.layers))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        for layer in self.layers:
            x = layer(x)
        return x, None

    def step(
        self,
        x: torch.Tensor,
        cache: list[dict[str, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        next_cache = []
        for index, layer in enumerate(self.layers):
            layer_cache = None if cache is None else cache[index]
            x, layer_cache = layer.step(x, layer_cache)
            next_cache.append(layer_cache)
        return x, next_cache


class LinearAttentionStateMixer(nn.Module):
    def __init__(self, dim: int, heads: int, rank: int) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by state heads")
        self.heads = heads
        self.rank = rank
        self.value_dim = dim // heads
        self.q_proj = nn.Linear(dim, heads * rank)
        self.k_proj = nn.Linear(dim, heads * rank)
        self.v_proj = nn.Linear(dim, dim)
        self.gate_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = positive_features(self.q_proj(x)).view(batch, seq_len, self.heads, self.rank)
        k = positive_features(self.k_proj(x)).view(batch, seq_len, self.heads, self.rank)
        v = torch.tanh(self.v_proj(x)).view(batch, seq_len, self.heads, self.value_dim)

        kv = torch.einsum("bthr,bthv->bthrv", k, v)
        kv_state = torch.cumsum(kv, dim=1)
        k_state = torch.cumsum(k, dim=1)

        y = torch.einsum("bthr,bthrv->bthv", q, kv_state)
        denom = torch.einsum("bthr,bthr->bth", q, k_state).clamp_min(1e-4)
        y = y / denom.unsqueeze(-1)
        y = y.reshape(batch, seq_len, self.heads * self.value_dim)
        return self.out_proj(y * torch.sigmoid(self.gate_proj(x)))

    def step(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch = x.shape[0]
        if state is None:
            kv_state = x.new_zeros(batch, self.heads, self.rank, self.value_dim)
            k_state = x.new_zeros(batch, self.heads, self.rank)
        else:
            kv_state, k_state = state

        q = positive_features(self.q_proj(x)).view(batch, self.heads, self.rank)
        k = positive_features(self.k_proj(x)).view(batch, self.heads, self.rank)
        v = torch.tanh(self.v_proj(x)).view(batch, self.heads, self.value_dim)
        kv_state = kv_state + torch.einsum("bhr,bhv->bhrv", k, v)
        k_state = k_state + k
        y = torch.einsum("bhr,bhrv->bhv", q, kv_state)
        denom = torch.einsum("bhr,bhr->bh", q, k_state).clamp_min(1e-4)
        y = (y / denom.unsqueeze(-1)).reshape(batch, self.heads * self.value_dim)
        return self.out_proj(y * torch.sigmoid(self.gate_proj(x))), (kv_state, k_state)


class SwiftBlock(nn.Module):
    def __init__(self, config: TinyLanguageModelConfig) -> None:
        super().__init__()
        dim = config.dim
        hidden = dim * config.mlp_expansion
        self.local_norm = RMSNorm(dim)
        self.local_conv = CausalDepthwiseConv(dim, config.conv_kernel)
        self.state_norm = RMSNorm(dim)
        self.state = LinearAttentionStateMixer(dim, config.state_heads, config.state_rank)
        self.mlp_norm = RMSNorm(dim)
        self.mlp_in = nn.Linear(dim, hidden * 2)
        self.mlp_out = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.local_conv(self.local_norm(x)))
        x = x + self.dropout(self.state(self.state_norm(x)))
        value, gate = self.mlp_in(self.mlp_norm(x)).chunk(2, dim=-1)
        return x + self.dropout(self.mlp_out(value * F.silu(gate)))

    def step(
        self,
        x: torch.Tensor,
        cache: dict[str, object] | None = None,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        if cache is None:
            cache = {"local": None, "state": None}
        y, local_cache = self.local_conv.step(self.local_norm(x), cache.get("local"))
        x = x + y
        y, state_cache = self.state.step(self.state_norm(x), cache.get("state"))
        x = x + y
        value, gate = self.mlp_in(self.mlp_norm(x)).chunk(2, dim=-1)
        x = x + self.mlp_out(value * F.silu(gate))
        return x, {"local": local_cache, "state": state_cache}


class SwiftStack(nn.Module):
    def __init__(self, config: TinyLanguageModelConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(SwiftBlock(config) for _ in range(config.layers))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        for layer in self.layers:
            x = layer(x)
        return x, None

    def step(
        self,
        x: torch.Tensor,
        cache: list[dict[str, object]] | None = None,
    ) -> tuple[torch.Tensor, list[dict[str, object]]]:
        next_cache = []
        for index, layer in enumerate(self.layers):
            layer_cache = None if cache is None else cache[index]
            x, layer_cache = layer.step(x, layer_cache)
            next_cache.append(layer_cache)
        return x, next_cache


class FastConvBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.norm = RMSNorm(dim)
        self.in_proj = nn.Linear(dim, dim * 2)
        self.conv = CausalDepthwiseOnlyConv(dim, kernel_size, dilation=dilation)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.in_proj(self.norm(x)).chunk(2, dim=-1)
        value = self.conv(value)
        return x + self.dropout(self.out_proj(value * torch.sigmoid(gate)))

    def step(
        self,
        x: torch.Tensor,
        cache: object | None = None,
    ) -> tuple[torch.Tensor, object | None]:
        value, gate = self.in_proj(self.norm(x)).chunk(2, dim=-1)
        value, cache = self.conv.step(value, cache)
        return x + self.out_proj(value * torch.sigmoid(gate)), cache


class FastConvStack(nn.Module):
    def __init__(self, config: TinyLanguageModelConfig) -> None:
        super().__init__()
        dilations = build_dilations(config.layers, config.max_dilation)
        self.layers = nn.ModuleList(
            FastConvBlock(config.dim, config.state_kernel, dilations[index], config.dropout)
            for index in range(config.layers)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        for layer in self.layers:
            x = layer(x)
        return x, None

    def step(
        self,
        x: torch.Tensor,
        cache: list[object | None] | None = None,
    ) -> tuple[torch.Tensor, list[object | None]]:
        next_cache = []
        for index, layer in enumerate(self.layers):
            layer_cache = None if cache is None else cache[index]
            x, layer_cache = layer.step(x, layer_cache)
            next_cache.append(layer_cache)
        return x, next_cache


class TinyLanguageModel(nn.Module):
    """A compact non-transformer language model."""

    def __init__(self, config: TinyLanguageModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.dim)
        if config.architecture == "stream":
            self.body = StreamStack(config)
        elif config.architecture == "swift":
            self.body = SwiftStack(config)
        elif config.architecture == "fastconv":
            self.body = FastConvStack(config)
        elif config.architecture == "conv":
            self.body = ConvStateStack(config)
        elif config.architecture == "rnn":
            self.local_norm = RMSNorm(config.dim)
            self.local_mixer = CausalDepthwiseConv(config.dim, config.conv_kernel)
            if config.recurrent == "gru":
                self.recurrent = nn.GRU(
                    input_size=config.dim,
                    hidden_size=config.dim,
                    num_layers=config.layers,
                    dropout=config.dropout if config.layers > 1 else 0.0,
                    batch_first=True,
                )
            elif config.recurrent == "gated":
                self.recurrent = GatedRecurrentStack(config.dim, config.layers, config.dropout)
            else:
                raise ValueError(f"unknown recurrent kind: {config.recurrent}")
        else:
            raise ValueError(f"unknown architecture: {config.architecture}")
        self.out_norm = RMSNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.head.weight = self.embed.weight

    def forward(
        self,
        tokens: torch.Tensor,
        state: object | None = None,
    ) -> tuple[torch.Tensor, object]:
        x = self.embed(tokens)
        if self.config.architecture in {"stream", "swift", "fastconv", "conv"}:
            x, state = self.body(x)
        else:
            x = x + self.local_mixer(self.local_norm(x))
            x, state = self.recurrent(x, state)
        logits = self.head(self.out_norm(x))
        return logits, state

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.9,
        top_k: int = 40,
        context: int | None = None,
    ) -> torch.Tensor:
        if prompt.ndim != 1:
            raise ValueError("prompt must be a flat token tensor")
        if len(prompt) == 0:
            raise ValueError("prompt must contain at least one token")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        if self.config.architecture in {"stream", "swift", "fastconv", "conv"}:
            return self._generate_conv_cached(prompt, max_new_tokens, temperature, top_k)
        if isinstance(self.recurrent, nn.GRU):
            return self._generate_gru_cached(prompt, max_new_tokens, temperature, top_k)

        self.eval()
        device = next(self.parameters()).device
        tokens = prompt.to(device).clone()
        for _ in range(max_new_tokens):
            current = tokens[-context:] if context and context > 0 else tokens
            logits, _ = self(current.unsqueeze(0))
            next_token = sample_logits(logits[:, -1, :], temperature, top_k).flatten()
            tokens = torch.cat([tokens, next_token])

        return tokens.detach().cpu()

    @torch.no_grad()
    def _generate_conv_cached(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
    ) -> torch.Tensor:
        self.eval()
        device = next(self.parameters()).device
        generated = [token for token in prompt.to(device)]
        cache = None
        logits = None
        for token in generated:
            x = self.embed(token.view(1))
            x, cache = self.body.step(x, cache)
            logits = self.head(self.out_norm(x)).unsqueeze(1)

        assert logits is not None
        for _ in range(max_new_tokens):
            next_token = sample_logits(logits[:, -1, :], temperature, top_k)
            generated.append(next_token.flatten()[0])
            x = self.embed(next_token.flatten())
            x, cache = self.body.step(x, cache)
            logits = self.head(self.out_norm(x)).unsqueeze(1)
        return torch.stack(generated).detach().cpu()

    @torch.no_grad()
    def _generate_gru_cached(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
    ) -> torch.Tensor:
        self.eval()
        device = next(self.parameters()).device
        generated = [token for token in prompt.to(device)]
        state = None
        conv_cache = None
        logits = None

        for token in generated:
            logits, state, conv_cache = self._gru_step(token.view(1), state, conv_cache)

        assert logits is not None
        for _ in range(max_new_tokens):
            next_token = sample_logits(logits[:, -1, :], temperature, top_k)
            generated.append(next_token.flatten()[0])
            logits, state, conv_cache = self._gru_step(next_token.flatten(), state, conv_cache)

        return torch.stack(generated).detach().cpu()

    def _gru_step(
        self,
        token: torch.Tensor,
        state: torch.Tensor | None,
        conv_cache: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        x = self.embed(token.view(1, 1))
        mixed, conv_cache = self.local_mixer.step(self.local_norm(x).squeeze(1), conv_cache)
        x = x + mixed.unsqueeze(1)
        x, state = self.recurrent(x, state)
        logits = self.head(self.out_norm(x))
        return logits, state, conv_cache


def build_dilations(layers: int, max_dilation: int) -> list[int]:
    dilations = []
    dilation = 1
    for _ in range(layers):
        dilations.append(dilation)
        dilation *= 2
        if dilation > max_dilation:
            dilation = 1
    return dilations


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def sample_logits(logits: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
    logits = logits / temperature
    if top_k > 0:
        values, _ = torch.topk(logits, k=min(top_k, logits.shape[-1]))
        logits = logits.masked_fill(logits < values[:, [-1]], -torch.inf)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def positive_features(x: torch.Tensor) -> torch.Tensor:
    return F.elu(x) + 1.0
