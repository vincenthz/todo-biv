//! Band-limited sample rate conversion by windowed-sinc interpolation.

use std::f64::consts::PI;

/// Zero crossings of the sinc kernel on each side of the output sample.
const ZERO_CROSSINGS: usize = 16;

/// Converts `input` from `from` Hz to `to` Hz. When downsampling, the kernel
/// is widened so it also filters out everything above the new Nyquist rate.
pub fn resample(input: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || input.is_empty() {
        return input.to_vec();
    }
    let ratio = to as f64 / from as f64;
    // Cutoff relative to the input Nyquist rate, slightly below the lower of
    // the two rates to leave room for the transition band.
    let cutoff = ratio.min(1.0) * 0.95;
    let half_width = ZERO_CROSSINGS as f64 / cutoff;
    let out_len = (input.len() as f64 * ratio).round() as usize;
    (0..out_len)
        .map(|n| {
            let center = n as f64 / ratio;
            let first = (center - half_width).ceil().max(0.0) as usize;
            let last = ((center + half_width).floor() as usize).min(input.len() - 1);
            let mut sum = 0.0;
            let mut weight = 0.0;
            for (i, &sample) in input.iter().enumerate().take(last + 1).skip(first) {
                let x = i as f64 - center;
                let w = cutoff * sinc(cutoff * x) * hann(x / half_width);
                sum += w * sample as f64;
                weight += w;
            }
            // Renormalizing keeps DC gain at one, including near the edges.
            if weight.abs() > 1e-9 { (sum / weight) as f32 } else { 0.0 }
        })
        .collect()
}

fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-12 { 1.0 } else { (PI * x).sin() / (PI * x) }
}

/// Hann window over `[-1, 1]`.
fn hann(t: f64) -> f64 {
    if t.abs() >= 1.0 { 0.0 } else { 0.5 + 0.5 * (PI * t).cos() }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(freq: f64, rate: u32, seconds: f64) -> Vec<f32> {
        (0..(rate as f64 * seconds) as usize)
            .map(|i| (2.0 * PI * freq * i as f64 / rate as f64).sin() as f32)
            .collect()
    }

    fn rms(samples: &[f32]) -> f32 {
        (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
    }

    #[test]
    fn keeps_passband_tone() {
        let out = resample(&tone(440.0, 48_000, 1.0), 48_000, 16_000);
        assert_eq!(out.len(), 16_000);
        let expected = tone(440.0, 16_000, 1.0);
        let error = out[200..15_800].iter().zip(&expected[200..15_800]).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        assert!(error < 0.01, "max error {error}");
    }

    #[test]
    fn removes_aliasing_tone() {
        // 12 kHz would fold back to 4 kHz at 16 kHz without filtering.
        let out = resample(&tone(12_000.0, 48_000, 1.0), 48_000, 16_000);
        assert!(rms(&out[200..15_800]) < 0.01);
    }

    #[test]
    fn handles_fractional_ratio() {
        let out = resample(&tone(300.0, 44_100, 0.5), 44_100, 16_000);
        assert_eq!(out.len(), 8_000);
        assert!((rms(&out[200..7_800]) - std::f32::consts::FRAC_1_SQRT_2).abs() < 0.01);
    }
}
