//! English speech recognition with Whisper checkpoints, implemented directly
//! on candle.

pub mod hub;
pub mod mel;
pub mod model;

use std::path::Path;

use anyhow::{Context, Result, anyhow, ensure};
use candle_core::DType;
pub use candle_core::Device;
use candle_nn::VarBuilder;
use serde::Deserialize;
use tokenizers::Tokenizer;

use self::{
    mel::{MelExtractor, N_SAMPLES},
    model::{Config, Whisper, argmax, log_softmax},
};
use crate::audio;

/// Whisper's own cap on generated tokens per 30 s window (half the context).
const MAX_NEW_TOKENS: usize = 224;

/// A window whose no-speech probability is above this and whose decode was
/// unsure (mean log-probability below [`LOGPROB_THRESHOLD`]) is treated as
/// silence, as in OpenAI's reference implementation.
const NO_SPEECH_THRESHOLD: f32 = 0.6;
const LOGPROB_THRESHOLD: f32 = -1.0;

/// What Whisper tends to "hear" in silence or room noise, lower-cased and
/// without punctuation. None of them is a meaningful request on its own.
const SILENCE_HALLUCINATIONS: &[&str] = &[
    "you",
    "thank you",
    "thank you very much",
    "thanks",
    "thanks for watching",
    "thank you for watching",
    "thank you so much for watching",
    "please subscribe",
    "bye",
    "bye bye",
    "okay",
    "so",
    "hmm",
];

#[derive(Debug, Default, Deserialize)]
struct GenerationConfig {
    #[serde(default)]
    suppress_tokens: Vec<u32>,
    #[serde(default)]
    begin_suppress_tokens: Vec<u32>,
}

#[derive(Clone, Debug, Default)]
pub struct Transcript {
    /// What was said, whitespace-normalized; empty when nothing was.
    pub text: String,
    /// Probability that the audio held no speech, highest over its windows.
    pub no_speech_probability: f32,
    /// Mean log-probability of the chosen tokens.
    pub avg_logprob: f32,
}

impl Transcript {
    /// Geometric mean of the per-token probabilities.
    pub fn confidence(&self) -> f32 {
        self.avg_logprob.exp()
    }

    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    fn silent() -> Self {
        Self { text: String::new(), no_speech_probability: 1.0, avg_logprob: f32::NEG_INFINITY }
    }
}

struct SpecialTokens {
    start_of_transcript: u32,
    end_of_text: u32,
    no_speech: u32,
    /// `<|startoftranscript|><|en|><|transcribe|><|notimestamps|>`.
    prompt: [u32; 4],
}

impl SpecialTokens {
    fn resolve(tokenizer: &Tokenizer) -> Result<Self> {
        let id = |token: &str| {
            tokenizer.token_to_id(token).ok_or_else(|| anyhow!("tokenizer has no {token} token"))
        };
        let start_of_transcript = id("<|startoftranscript|>")?;
        Ok(Self {
            start_of_transcript,
            end_of_text: id("<|endoftext|>")?,
            // large-v3 renamed the token.
            no_speech: id("<|nospeech|>").or_else(|_| id("<|nocaptions|>"))?,
            prompt: [start_of_transcript, id("<|en|>")?, id("<|transcribe|>")?, id("<|notimestamps|>")?],
        })
    }
}

/// One decoded 30 s window.
struct Window {
    tokens: Vec<u32>,
    logprobs: Vec<f32>,
    no_speech: f32,
}

impl Window {
    fn avg_logprob(&self) -> f32 {
        if self.logprobs.is_empty() { f32::NEG_INFINITY } else { self.logprobs.iter().sum::<f32>() / self.logprobs.len() as f32 }
    }

    fn is_silence(&self) -> bool {
        self.tokens.is_empty() || (self.no_speech > NO_SPEECH_THRESHOLD && self.avg_logprob() < LOGPROB_THRESHOLD)
    }
}

pub struct Transcriber {
    model: Whisper,
    tokenizer: Tokenizer,
    mel: MelExtractor,
    special: SpecialTokens,
    /// Tokens never emitted, as a mask over the vocabulary.
    suppressed: Vec<bool>,
    begin_suppressed: Vec<u32>,
    device: Device,
    dtype: DType,
}

impl Transcriber {
    /// Loads a multilingual checkpoint directory holding the files listed in
    /// [`hub::FILES`], to transcribe English.
    pub fn load(dir: &Path, device: Device) -> Result<Self> {
        let read = |name: &str| -> Result<String> {
            std::fs::read_to_string(dir.join(name)).with_context(|| format!("reading {name}"))
        };
        let config: Config = serde_json::from_str(&read("config.json")?)?;
        let generation: GenerationConfig = serde_json::from_str(&read("generation_config.json")?)?;
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow!("loading tokenizer: {e}"))?;
        let special = SpecialTokens::resolve(&tokenizer)?;

        // Half precision is fine on the GPU; the CPU kernels are fastest in f32.
        let dtype = if device.is_cpu() { DType::F32 } else { DType::F16 };
        let weights = dir.join("model.safetensors");
        // SAFETY: the weights file is memory-mapped and must not change while loaded.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], dtype, &device)? };
        let model = Whisper::load(config, vb).context("loading Whisper weights")?;

        ensure!(
            special.end_of_text < special.start_of_transcript,
            "unexpected special token layout"
        );
        let mut suppressed = vec![false; model.config.vocab_size];
        for &token in &generation.suppress_tokens {
            if let Some(slot) = suppressed.get_mut(token as usize) {
                *slot = true;
            }
        }
        // Control and timestamp tokens all follow <|startoftranscript|>: only
        // text and end-of-text are generated.
        for slot in suppressed.iter_mut().skip(special.start_of_transcript as usize) {
            *slot = true;
        }

        Ok(Self {
            mel: MelExtractor::new(model.config.num_mel_bins),
            model,
            tokenizer,
            special,
            suppressed,
            begin_suppressed: generation.begin_suppress_tokens,
            device,
            dtype,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Runs the model once over silence, so the first real transcription does
    /// not also pay for paging in the weights and preparing GPU kernels.
    pub fn warm_up(&self) -> Result<()> {
        let mel = self.mel.compute(&[]);
        let mel = self.model.mel_tensor(&mel, self.dtype, &self.device)?;
        let audio = self.model.encoder.forward(&mel)?;
        let decoder = &self.model.decoder;
        let mut state = decoder.start(&audio)?;
        decoder.forward(&self.special.prompt, &mut state)?;
        // Reading the logits back waits for the GPU to finish.
        decoder.forward(&[self.special.end_of_text], &mut state)?.to_vec2::<f32>()?;
        Ok(())
    }

    /// Transcribes 16 kHz mono audio, 30 s at a time. Silence, and what
    /// Whisper typically makes up over it, gives an empty transcript.
    pub fn transcribe(&self, samples: &[f32]) -> Result<Transcript> {
        if audio::is_silent(samples) {
            return Ok(Transcript::silent());
        }
        let mut tokens = Vec::new();
        let mut logprobs = Vec::new();
        let mut no_speech_probability = 0f32;
        for samples in samples.chunks(N_SAMPLES).filter(|w| !audio::is_silent(w)) {
            let window = self.decode_window(samples)?;
            no_speech_probability = no_speech_probability.max(window.no_speech);
            if !window.is_silence() {
                tokens.extend(window.tokens);
                logprobs.extend(window.logprobs);
            }
        }
        let text = self.tokenizer.decode(&tokens, true).map_err(|e| anyhow!("decoding: {e}"))?;
        let text = clean(&text);
        if is_hallucination(&text) {
            return Ok(Transcript { no_speech_probability, ..Transcript::silent() });
        }
        Ok(Transcript {
            text,
            no_speech_probability,
            avg_logprob: if logprobs.is_empty() { f32::NEG_INFINITY } else { logprobs.iter().sum::<f32>() / logprobs.len() as f32 },
        })
    }

    /// Greedy decoding of one 30 s window.
    fn decode_window(&self, samples: &[f32]) -> Result<Window> {
        let mel = self.mel.compute(samples);
        let mel = self.model.mel_tensor(&mel, self.dtype, &self.device)?;
        let audio = self.model.encoder.forward(&mel)?;
        let decoder = &self.model.decoder;
        let mut state = decoder.start(&audio)?;

        let prompt = &self.special.prompt;
        let logits = decoder.forward(prompt, &mut state)?;
        // The no-speech token is predicted right after <|startoftranscript|>.
        let no_speech = log_softmax(&logits.get(0)?.to_vec1()?)[self.special.no_speech as usize].exp();
        let mut logits: Vec<f32> = logits.get(prompt.len() - 1)?.to_vec1()?;

        let mut tokens = Vec::new();
        let mut logprobs = Vec::new();
        let max_new = MAX_NEW_TOKENS.min(decoder.max_positions() - prompt.len());
        while tokens.len() < max_new {
            for (logit, &suppressed) in logits.iter_mut().zip(&self.suppressed) {
                if suppressed {
                    *logit = f32::NEG_INFINITY;
                }
            }
            if tokens.is_empty() {
                for &token in &self.begin_suppressed {
                    if let Some(logit) = logits.get_mut(token as usize) {
                        *logit = f32::NEG_INFINITY;
                    }
                }
            }
            let next = argmax(&logits) as u32;
            if next == self.special.end_of_text {
                break;
            }
            logprobs.push(log_softmax(&logits)[next as usize]);
            tokens.push(next);
            logits = decoder.forward(&[next], &mut state)?.get(0)?.to_vec1()?;
        }
        Ok(Window { tokens, logprobs, no_speech })
    }
}

/// Collapses runs of whitespace and trims.
fn clean(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Whether `text` is only a stock phrase Whisper produces over silence, or a
/// caption-style annotation such as `[BLANK_AUDIO]` or `(music)`.
fn is_hallucination(text: &str) -> bool {
    let annotation = |open, close| {
        text.strip_prefix(open).and_then(|t| t.strip_suffix(close)).is_some_and(|inner| !inner.contains([open, close]))
    };
    if annotation('[', ']') || annotation('(', ')') || annotation('*', '*') {
        return true;
    }
    let letters: String = text
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '\'' { c } else { ' ' })
        .collect::<String>()
        .to_lowercase();
    let words = clean(&letters);
    words.is_empty() || SILENCE_HALLUCINATIONS.contains(&words.as_str())
}

/// The Metal GPU when built with it, else the CPU.
pub fn default_device() -> Device {
    #[cfg(feature = "metal")]
    if let Ok(device) = Device::new_metal(0) {
        return device;
    }
    Device::Cpu
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cleans_whitespace() {
        assert_eq!(clean("  Buy milk\n tomorrow.  "), "Buy milk tomorrow.");
        assert_eq!(clean(" \t"), "");
    }

    #[test]
    fn spots_silence_hallucinations() {
        for text in ["", "Thank you.", "thank you!", "You", "Thanks for watching!", "[BLANK_AUDIO]", "(music)", "*silence*", "...", "Bye-bye."] {
            assert!(is_hallucination(text), "{text:?}");
        }
        for text in ["Thank you, remind me to call mum.", "Buy milk", "Call Bob (the plumber)", "[urgent] pay rent"] {
            assert!(!is_hallucination(text), "{text:?}");
        }
    }
}
