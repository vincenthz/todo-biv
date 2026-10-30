//! The Whisper encoder-decoder transformer, written against Hugging Face's
//! `WhisperForConditionalGeneration` weight layout.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{
    Conv1d, Conv1dConfig, Embedding, LayerNorm, Linear, Module, VarBuilder, conv1d, embedding,
    layer_norm, linear, linear_no_bias, ops::softmax_last_dim,
};
use serde::Deserialize;

/// The architecture fields of a checkpoint's `config.json`.
#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    pub num_mel_bins: usize,
    pub d_model: usize,
    pub encoder_layers: usize,
    pub encoder_attention_heads: usize,
    pub encoder_ffn_dim: usize,
    pub decoder_layers: usize,
    pub decoder_attention_heads: usize,
    pub decoder_ffn_dim: usize,
    pub max_source_positions: usize,
    pub max_target_positions: usize,
    pub vocab_size: usize,
}

const LAYER_NORM_EPS: f64 = 1e-5;

struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    heads: usize,
}

impl Attention {
    fn load(d_model: usize, heads: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            q: linear(d_model, d_model, vb.pp("q_proj"))?,
            k: linear_no_bias(d_model, d_model, vb.pp("k_proj"))?,
            v: linear(d_model, d_model, vb.pp("v_proj"))?,
            out: linear(d_model, d_model, vb.pp("out_proj"))?,
            heads,
        })
    }

    /// `(batch, time, d_model)` → `(batch, heads, time, head_dim)`.
    fn split_heads(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, d) = x.dims3()?;
        x.reshape((b, t, self.heads, d / self.heads))?.transpose(1, 2)?.contiguous()
    }

    fn key_value(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        Ok((self.split_heads(&self.k.forward(x)?)?, self.split_heads(&self.v.forward(x)?)?))
    }

    /// Attends from `x` to precomputed keys and values. `mask` is added to the
    /// attention scores and broadcasts over batch and heads.
    fn attend(&self, x: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, t, d) = x.dims3()?;
        let scale = ((d / self.heads) as f64).powf(-0.5);
        let q = (self.split_heads(&self.q.forward(x)?)? * scale)?;
        let mut scores = q.matmul(&k.t()?)?;
        if let Some(mask) = mask {
            scores = scores.broadcast_add(mask)?;
        }
        let weights = softmax_last_dim(&scores)?;
        let y = weights.matmul(v)?.transpose(1, 2)?.reshape((b, t, d))?;
        self.out.forward(&y)
    }
}

struct Mlp {
    fc1: Linear,
    fc2: Linear,
}

impl Mlp {
    fn load(d_model: usize, ffn_dim: usize, vb: &VarBuilder) -> Result<Self> {
        Ok(Self { fc1: linear(d_model, ffn_dim, vb.pp("fc1"))?, fc2: linear(ffn_dim, d_model, vb.pp("fc2"))? })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.fc2.forward(&self.fc1.forward(x)?.gelu_erf()?)
    }
}

struct EncoderLayer {
    attn_norm: LayerNorm,
    attn: Attention,
    mlp_norm: LayerNorm,
    mlp: Mlp,
}

impl EncoderLayer {
    fn load(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            attn_norm: layer_norm(cfg.d_model, LAYER_NORM_EPS, vb.pp("self_attn_layer_norm"))?,
            attn: Attention::load(cfg.d_model, cfg.encoder_attention_heads, vb.pp("self_attn"))?,
            mlp_norm: layer_norm(cfg.d_model, LAYER_NORM_EPS, vb.pp("final_layer_norm"))?,
            mlp: Mlp::load(cfg.d_model, cfg.encoder_ffn_dim, &vb)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.attn_norm.forward(x)?;
        let (k, v) = self.attn.key_value(&h)?;
        let x = (x + self.attn.attend(&h, &k, &v, None)?)?;
        &x + self.mlp.forward(&self.mlp_norm.forward(&x)?)?
    }
}

pub struct Encoder {
    conv1: Conv1d,
    conv2: Conv1d,
    positions: Tensor,
    layers: Vec<EncoderLayer>,
    norm: LayerNorm,
}

impl Encoder {
    fn load(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let same = Conv1dConfig { padding: 1, ..Default::default() };
        let halve = Conv1dConfig { padding: 1, stride: 2, ..Default::default() };
        Ok(Self {
            conv1: conv1d(cfg.num_mel_bins, cfg.d_model, 3, same, vb.pp("conv1"))?,
            conv2: conv1d(cfg.d_model, cfg.d_model, 3, halve, vb.pp("conv2"))?,
            positions: vb.get((cfg.max_source_positions, cfg.d_model), "embed_positions.weight")?,
            layers: (0..cfg.encoder_layers)
                .map(|i| EncoderLayer::load(cfg, vb.pp(format!("layers.{i}"))))
                .collect::<Result<_>>()?,
            norm: layer_norm(cfg.d_model, LAYER_NORM_EPS, vb.pp("layer_norm"))?,
        })
    }

    /// `(batch, n_mels, 3000)` log-mel frames → `(batch, 1500, d_model)` audio features.
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        let x = self.conv1.forward(mel)?.gelu_erf()?;
        let x = self.conv2.forward(&x)?.gelu_erf()?.transpose(1, 2)?;
        let mut x = x.broadcast_add(&self.positions.narrow(0, 0, x.dim(1)?)?)?;
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }
        self.norm.forward(&x)
    }
}

struct DecoderLayer {
    self_attn_norm: LayerNorm,
    self_attn: Attention,
    cross_attn_norm: LayerNorm,
    cross_attn: Attention,
    mlp_norm: LayerNorm,
    mlp: Mlp,
}

impl DecoderLayer {
    fn load(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            self_attn_norm: layer_norm(cfg.d_model, LAYER_NORM_EPS, vb.pp("self_attn_layer_norm"))?,
            self_attn: Attention::load(cfg.d_model, cfg.decoder_attention_heads, vb.pp("self_attn"))?,
            cross_attn_norm: layer_norm(cfg.d_model, LAYER_NORM_EPS, vb.pp("encoder_attn_layer_norm"))?,
            cross_attn: Attention::load(cfg.d_model, cfg.decoder_attention_heads, vb.pp("encoder_attn"))?,
            mlp_norm: layer_norm(cfg.d_model, LAYER_NORM_EPS, vb.pp("final_layer_norm"))?,
            mlp: Mlp::load(cfg.d_model, cfg.decoder_ffn_dim, &vb)?,
        })
    }

    fn forward(&self, x: &Tensor, cache: &mut LayerCache, mask: Option<&Tensor>) -> Result<Tensor> {
        let h = self.self_attn_norm.forward(x)?;
        let (k, v) = self.self_attn.key_value(&h)?;
        let (k, v) = match cache.self_kv.take() {
            Some((past_k, past_v)) => (Tensor::cat(&[past_k, k], 2)?, Tensor::cat(&[past_v, v], 2)?),
            None => (k, v),
        };
        let x = (x + self.self_attn.attend(&h, &k, &v, mask)?)?;
        cache.self_kv = Some((k, v));

        let h = self.cross_attn_norm.forward(&x)?;
        let (k, v) = &cache.cross_kv;
        let x = (&x + self.cross_attn.attend(&h, k, v, None)?)?;
        &x + self.mlp.forward(&self.mlp_norm.forward(&x)?)?
    }
}

struct LayerCache {
    self_kv: Option<(Tensor, Tensor)>,
    cross_kv: (Tensor, Tensor),
}

/// Per-utterance decoder state: cross-attention keys and values for the audio,
/// and the self-attention cache of every token decoded so far.
pub struct DecoderState {
    layers: Vec<LayerCache>,
    len: usize,
}

pub struct Decoder {
    tokens: Embedding,
    positions: Tensor,
    layers: Vec<DecoderLayer>,
    norm: LayerNorm,
    lm_head: Linear,
    max_positions: usize,
}

impl Decoder {
    fn load(cfg: &Config, vb: VarBuilder, lm_head: Option<Tensor>) -> Result<Self> {
        let tokens = embedding(cfg.vocab_size, cfg.d_model, vb.pp("embed_tokens"))?;
        // The output projection is tied to the token embeddings, though most
        // checkpoints also store it as `proj_out`.
        let lm_head = Linear::new(lm_head.unwrap_or_else(|| tokens.embeddings().clone()), None);
        Ok(Self {
            tokens,
            positions: vb.get((cfg.max_target_positions, cfg.d_model), "embed_positions.weight")?,
            layers: (0..cfg.decoder_layers)
                .map(|i| DecoderLayer::load(cfg, vb.pp(format!("layers.{i}"))))
                .collect::<Result<_>>()?,
            norm: layer_norm(cfg.d_model, LAYER_NORM_EPS, vb.pp("layer_norm"))?,
            lm_head,
            max_positions: cfg.max_target_positions,
        })
    }

    pub fn max_positions(&self) -> usize {
        self.max_positions
    }

    /// Prepares decoding against `(1, frames, d_model)` audio features.
    pub fn start(&self, audio: &Tensor) -> Result<DecoderState> {
        let layers = self
            .layers
            .iter()
            .map(|layer| {
                Ok(LayerCache { self_kv: None, cross_kv: layer.cross_attn.key_value(audio)? })
            })
            .collect::<Result<_>>()?;
        Ok(DecoderState { layers, len: 0 })
    }

    /// Feeds `tokens` after those already in `state` and returns f32 logits,
    /// `(tokens.len(), vocab)`.
    pub fn forward(&self, tokens: &[u32], state: &mut DecoderState) -> Result<Tensor> {
        let device = self.positions.device();
        let (offset, len) = (state.len, tokens.len());
        let ids = Tensor::new(tokens, device)?.unsqueeze(0)?;
        let mut x = self
            .tokens
            .forward(&ids)?
            .broadcast_add(&self.positions.narrow(0, offset, len)?)?;
        let mask = (len > 1)
            .then(|| causal_mask(len, offset, x.dtype(), device))
            .transpose()?;
        for (layer, cache) in self.layers.iter().zip(&mut state.layers) {
            x = layer.forward(&x, cache, mask.as_ref())?;
        }
        state.len += len;
        self.lm_head.forward(&self.norm.forward(&x)?.squeeze(0)?)?.to_dtype(DType::F32)
    }
}

/// Lets query `i` (at position `offset + i`) see keys up to its own position.
fn causal_mask(len: usize, offset: usize, dtype: DType, device: &Device) -> Result<Tensor> {
    let mask: Vec<f32> = (0..len)
        .flat_map(|i| (0..offset + len).map(move |j| if j > offset + i { f32::NEG_INFINITY } else { 0.0 }))
        .collect();
    Tensor::from_vec(mask, (len, offset + len), device)?.to_dtype(dtype)
}

pub struct Whisper {
    pub encoder: Encoder,
    pub decoder: Decoder,
    pub config: Config,
}

impl Whisper {
    pub fn load(config: Config, vb: VarBuilder) -> Result<Self> {
        let lm_head = if vb.contains_tensor("proj_out.weight") {
            Some(vb.get((config.vocab_size, config.d_model), "proj_out.weight")?)
        } else {
            None
        };
        Ok(Self {
            encoder: Encoder::load(&config, vb.pp("model.encoder"))?,
            decoder: Decoder::load(&config, vb.pp("model.decoder"), lm_head)?,
            config,
        })
    }

    /// Log-mel frames, `n_mels × 3000` row-major, as an encoder input batch of one.
    pub fn mel_tensor(&self, mel: &[f32], dtype: DType, device: &Device) -> Result<Tensor> {
        let frames = mel.len() / self.config.num_mel_bins;
        Tensor::from_slice(mel, (1, self.config.num_mel_bins, frames), device)?.to_dtype(dtype)
    }
}

/// Log-softmax over the last dimension of f32 logits.
pub fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let log_sum = logits.iter().map(|l| (l - max).exp()).sum::<f32>().ln() + max;
    logits.iter().map(|l| l - log_sum).collect()
}

/// Index of the largest value along the last dimension.
pub fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(i, _)| i)
}
