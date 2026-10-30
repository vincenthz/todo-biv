//! Microphone capture, normalized to Whisper's 16 kHz mono.

pub mod capture;
pub mod resample;

pub use crate::whisper::mel::SAMPLE_RATE;

/// Loudest RMS over 30 ms frames below which audio counts as silence
/// (about -46 dBFS: a quiet room, well under even soft speech).
pub const SILENCE_RMS: f32 = 0.005;

/// Averages interleaved channels into one.
pub fn downmix(interleaved: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return interleaved.to_vec();
    }
    interleaved
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

/// RMS of the loudest 30 ms frame of 16 kHz audio, so a short utterance in a
/// long recording still registers.
pub fn peak_rms(samples: &[f32]) -> f32 {
    const FRAME: usize = SAMPLE_RATE as usize * 30 / 1000;
    samples
        .chunks(FRAME)
        .map(|frame| (frame.iter().map(|s| s * s).sum::<f32>() / frame.len() as f32).sqrt())
        .fold(0.0, f32::max)
}

/// Whether 16 kHz audio holds nothing loud enough to be speech.
pub fn is_silent(samples: &[f32]) -> bool {
    peak_rms(samples) < SILENCE_RMS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn downmixes_stereo() {
        assert_eq!(downmix(&[1.0, 0.0, 0.5, 0.5], 2), vec![0.5, 0.5]);
    }

    #[test]
    fn detects_silence() {
        assert!(is_silent(&[]));
        assert!(is_silent(&vec![0.001; 16_000]));
        let mut speech = vec![0.0; 16_000];
        for (i, s) in speech[8_000..9_000].iter_mut().enumerate() {
            *s = 0.1 * (i as f32 * 0.2).sin();
        }
        assert!(!is_silent(&speech));
    }
}
