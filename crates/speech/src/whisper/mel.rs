//! Whisper's log-mel spectrogram front end.
//!
//! Mirrors `WhisperFeatureExtractor` from Hugging Face transformers: the audio
//! is zero-padded to 30 s, framed by a centered STFT (400-point periodic Hann
//! window, hop 160, reflect padding), turned into a power spectrum, projected
//! onto Slaney-normalized mel filters, and log-compressed with an 80 dB
//! dynamic range.

use std::sync::Arc;

use rustfft::{Fft, FftPlanner, num_complex::Complex};

pub const SAMPLE_RATE: u32 = 16_000;
pub const N_FFT: usize = 400;
pub const HOP_LENGTH: usize = 160;
pub const CHUNK_SECONDS: usize = 30;
/// Samples in one 30 s input window.
pub const N_SAMPLES: usize = SAMPLE_RATE as usize * CHUNK_SECONDS;
/// Spectrogram frames in one input window.
pub const N_FRAMES: usize = N_SAMPLES / HOP_LENGTH;

const N_FREQS: usize = N_FFT / 2 + 1;

pub struct MelExtractor {
    n_mels: usize,
    /// `n_mels × N_FREQS`, row-major.
    filters: Vec<f32>,
    window: Vec<f32>,
    fft: Arc<dyn Fft<f32>>,
}

impl MelExtractor {
    pub fn new(n_mels: usize) -> Self {
        let window = (0..N_FFT)
            .map(|i| {
                let phase = 2.0 * std::f64::consts::PI * i as f64 / N_FFT as f64;
                (0.5 - 0.5 * phase.cos()) as f32
            })
            .collect();
        Self {
            n_mels,
            filters: mel_filters(n_mels, N_FREQS, SAMPLE_RATE as f64),
            window,
            fft: FftPlanner::new().plan_fft_forward(N_FFT),
        }
    }

    pub fn n_mels(&self) -> usize {
        self.n_mels
    }

    /// Log-mel spectrogram of up to 30 s of 16 kHz mono audio, `n_mels × N_FRAMES`
    /// row-major. Longer input is truncated.
    pub fn compute(&self, samples: &[f32]) -> Vec<f32> {
        let mut audio = vec![0f32; N_SAMPLES];
        let len = samples.len().min(N_SAMPLES);
        audio[..len].copy_from_slice(&samples[..len]);

        let padded = reflect_pad(&audio, N_FFT / 2);
        let mut power = vec![0f32; N_FRAMES * N_FREQS];
        let mut buffer = vec![Complex::default(); N_FFT];
        let mut scratch = vec![Complex::default(); self.fft.get_inplace_scratch_len()];
        // The centered STFT yields N_FRAMES + 1 frames; Whisper drops the last.
        for (frame, row) in power.as_chunks_mut::<N_FREQS>().0.iter_mut().enumerate() {
            let start = frame * HOP_LENGTH;
            for ((slot, &sample), &w) in buffer.iter_mut().zip(&padded[start..start + N_FFT]).zip(&self.window) {
                *slot = Complex::new(sample * w, 0.0);
            }
            self.fft.process_with_scratch(&mut buffer, &mut scratch);
            for (p, c) in row.iter_mut().zip(&buffer) {
                *p = c.norm_sqr();
            }
        }

        let mut mel = vec![0f32; self.n_mels * N_FRAMES];
        let (filters, power) = (self.filters.as_chunks::<N_FREQS>().0, power.as_chunks::<N_FREQS>().0);
        for (filter, out) in filters.iter().zip(mel.as_chunks_mut::<N_FRAMES>().0) {
            for (value, row) in out.iter_mut().zip(power) {
                let energy: f32 = filter.iter().zip(row).map(|(f, p)| f * p).sum();
                *value = energy.max(1e-10).log10();
            }
        }
        let max = mel.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        for value in &mut mel {
            *value = (value.max(max - 8.0) + 4.0) / 4.0;
        }
        mel
    }
}

/// numpy's `reflect` padding: the edge sample is not repeated.
fn reflect_pad(samples: &[f32], pad: usize) -> Vec<f32> {
    let n = samples.len();
    let mut out = Vec::with_capacity(n + 2 * pad);
    out.extend((1..=pad).rev().map(|i| samples[i]));
    out.extend_from_slice(samples);
    out.extend((1..=pad).map(|i| samples[n - 1 - i]));
    out
}

fn hz_to_mel(hz: f64) -> f64 {
    // Slaney's scale: linear below 1 kHz, logarithmic above.
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1000.0;
    if hz < MIN_LOG_HZ {
        hz / F_SP
    } else {
        MIN_LOG_HZ / F_SP + (hz / MIN_LOG_HZ).ln() / (6.4f64.ln() / 27.0)
    }
}

fn mel_to_hz(mel: f64) -> f64 {
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1000.0;
    const MIN_LOG_MEL: f64 = MIN_LOG_HZ / F_SP;
    if mel < MIN_LOG_MEL {
        mel * F_SP
    } else {
        MIN_LOG_HZ * ((mel - MIN_LOG_MEL) * (6.4f64.ln() / 27.0)).exp()
    }
}

/// Triangular, area-normalized mel filters (librosa's `norm="slaney"`).
fn mel_filters(n_mels: usize, n_freqs: usize, sample_rate: f64) -> Vec<f32> {
    let nyquist = sample_rate / 2.0;
    let (mel_min, mel_max) = (hz_to_mel(0.0), hz_to_mel(nyquist));
    let edges: Vec<f64> = (0..n_mels + 2)
        .map(|i| mel_to_hz(mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64))
        .collect();
    let mut filters = vec![0f32; n_mels * n_freqs];
    for m in 0..n_mels {
        let (lower, center, upper) = (edges[m], edges[m + 1], edges[m + 2]);
        let norm = 2.0 / (upper - lower);
        for k in 0..n_freqs {
            let freq = nyquist * k as f64 / (n_freqs - 1) as f64;
            let rising = (freq - lower) / (center - lower);
            let falling = (upper - freq) / (upper - center);
            filters[m * n_freqs + k] = (rising.min(falling).max(0.0) * norm) as f32;
        }
    }
    filters
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mel_scale_round_trips() {
        for hz in [0.0, 440.0, 999.0, 1000.0, 4000.0, 8000.0] {
            assert!((mel_to_hz(hz_to_mel(hz)) - hz).abs() < 1e-6, "{hz}");
        }
    }

    #[test]
    fn silence_is_flat() {
        let mel = MelExtractor::new(80).compute(&[]);
        assert_eq!(mel.len(), 80 * N_FRAMES);
        // log10(1e-10) = -10, which clamps to max - 8 and normalizes to -1.5.
        assert!(mel.iter().all(|&v| (v + 1.5).abs() < 1e-6));
    }

    #[test]
    fn tone_peaks_in_matching_band() {
        let tone: Vec<f32> = (0..SAMPLE_RATE as usize)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / SAMPLE_RATE as f32).sin())
            .collect();
        let extractor = MelExtractor::new(80);
        let mel = extractor.compute(&tone);
        let frame = 50;
        let loudest = (0..80)
            .max_by(|&a, &b| mel[a * N_FRAMES + frame].total_cmp(&mel[b * N_FRAMES + frame]))
            .unwrap();
        let center = mel_to_hz(hz_to_mel(8000.0) * (loudest + 1) as f64 / 81.0);
        assert!((center - 1000.0).abs() < 100.0, "band {loudest} centered at {center} Hz");
    }
}
