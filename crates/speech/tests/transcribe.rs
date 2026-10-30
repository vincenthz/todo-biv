//! End-to-end checks against real checkpoints. Downloads ~2.6 GB on first
//! run: `cargo test --release -p speech -- --ignored`.

use std::path::Path;

use speech::whisper::{Transcriber, default_device, hub};

fn transcriber(checkpoint: &str) -> anyhow::Result<Transcriber> {
    let checkpoint = hub::checkpoint_or_default(checkpoint);
    Transcriber::load(&hub::fetch(checkpoint, &mut |_| {})?, default_device())
}

/// A 16 kHz 16-bit mono test recording.
fn load(name: &str) -> anyhow::Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data").join(name))?;
    anyhow::ensure!(reader.spec().sample_rate == 16_000 && reader.spec().channels == 1, "{name} is not 16 kHz mono");
    Ok(reader.samples::<i16>().map(|s| s.map(|s| s as f32 / 32768.0)).collect::<Result<_, _>>()?)
}

/// Gaussian-ish noise from a fixed linear congruential generator.
fn noise(seconds: f32, amplitude: f32) -> Vec<f32> {
    let mut state = 0x2545_f491_u32;
    (0..(seconds * 16_000.0) as usize)
        .map(|_| {
            let sum: f32 = (0..4)
                .map(|_| {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    (state >> 8) as f32 / (1 << 24) as f32 - 0.5
                })
                .sum();
            sum * amplitude * 1.7
        })
        .collect()
}

fn hears_a_request(checkpoint: &str) -> anyhow::Result<()> {
    let transcriber = transcriber(checkpoint)?;
    // Generated with `say "Remind me to buy milk tomorrow morning, it is important"`.
    let transcript = transcriber.transcribe(&load("buy_milk.wav")?)?;
    let words: String = transcript.text.to_lowercase().chars().filter(|c| c.is_alphanumeric() || *c == ' ').collect();
    assert_eq!(words, "remind me to buy milk tomorrow morning it is important", "heard {:?}", transcript.text);
    assert!(transcript.confidence() > 0.8, "{}", transcript.confidence());
    assert!(transcript.no_speech_probability < 0.05);

    for (what, samples) in [("silence", vec![0.0; 64_000]), ("hiss", noise(4.0, 0.002)), ("noise", noise(4.0, 0.02))] {
        let transcript = transcriber.transcribe(&samples)?;
        assert!(transcript.is_empty(), "{what} heard as {:?}", transcript.text);
    }
    Ok(())
}

#[test]
#[ignore = "downloads a Whisper checkpoint"]
fn large_v3_turbo_hears_a_request() -> anyhow::Result<()> {
    hears_a_request("large-v3-turbo")
}

#[test]
#[ignore = "downloads a Whisper checkpoint"]
fn small_hears_a_request() -> anyhow::Result<()> {
    hears_a_request("small")
}
