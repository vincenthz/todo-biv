//! Transcribes a WAV file, or a few seconds from the microphone, with a cached
//! (or freshly downloaded) Whisper checkpoint.
//!
//! cargo run --release -p speech --example transcribe -- [--checkpoint small] [--cpu] <file.wav | --record SECONDS>

use std::{
    io::Write,
    path::Path,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail};
use speech::{
    audio::{
        SAMPLE_RATE, downmix, peak_rms,
        capture::{Recorder, input_devices},
        resample::resample,
    },
    whisper::{Device, Transcriber, default_device, hub},
};

enum Input {
    Wav(String),
    Record(f32),
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mut input = None;
    let mut checkpoint = hub::DEFAULT_CHECKPOINT.to_string();
    let mut device = default_device();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--record" => {
                let seconds = args.next().context("--record needs a number of seconds")?;
                input = Some(Input::Record(seconds.parse().with_context(|| format!("bad duration {seconds:?}"))?));
            }
            "--checkpoint" => checkpoint = args.next().context("--checkpoint needs a name")?,
            "--cpu" => device = Device::Cpu,
            "-h" | "--help" => {
                println!("usage: transcribe [--checkpoint NAME] [--cpu] <file.wav | --record SECONDS>");
                let names: Vec<_> = hub::CHECKPOINTS.iter().map(|c| c.name).collect();
                println!("checkpoints: {}", names.join(", "));
                return Ok(());
            }
            path => input = Some(Input::Wav(path.to_string())),
        }
    }
    let Some(input) = input else { bail!("give a WAV file or --record SECONDS (see --help)") };

    let samples = match input {
        Input::Wav(path) => read_wav(Path::new(&path))?,
        Input::Record(seconds) => record(seconds)?,
    };
    let audio_seconds = samples.len() as f32 / SAMPLE_RATE as f32;
    println!("audio: {audio_seconds:.2} s, loudest frame {:.4} RMS", peak_rms(&samples));

    let checkpoint = hub::checkpoint_or_default(&checkpoint);
    let dir = match hub::cached(checkpoint) {
        Some(dir) => dir,
        None => {
            println!("downloading {} (~{} MB)", checkpoint.repo, checkpoint.download_mb);
            hub::fetch(checkpoint, &mut |p| {
                print!("\r{} {:.0}/{:.0} MB", p.file, p.downloaded as f64 / 1e6, p.total as f64 / 1e6);
                let _ = std::io::stdout().flush();
            })?
        }
    };
    let started = Instant::now();
    let transcriber = Transcriber::load(&dir, device)?;
    println!("loaded {} on {:?} in {:.2} s", checkpoint.name, transcriber.device(), started.elapsed().as_secs_f32());
    let started = Instant::now();
    transcriber.warm_up()?;
    println!("warmed up in {:.2} s", started.elapsed().as_secs_f32());

    let started = Instant::now();
    let transcript = transcriber.transcribe(&samples)?;
    let elapsed = started.elapsed().as_secs_f32();
    println!(
        "transcribed in {elapsed:.2} s ({:.1}x real time), confidence {:.3}, no-speech {:.3}",
        audio_seconds / elapsed.max(1e-6),
        transcript.confidence(),
        transcript.no_speech_probability,
    );
    if transcript.is_empty() {
        println!("(no speech)");
    } else {
        println!("{}", transcript.text);
    }
    Ok(())
}

/// Reads a WAV file as 16 kHz mono.
fn read_wav(path: &Path) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path).with_context(|| format!("opening {}", path.display()))?;
    let spec = reader.spec();
    if spec.channels == 0 {
        bail!("{} has no channels", path.display());
    }
    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1u64 << (spec.bits_per_sample - 1)) as f32;
            reader.samples::<i32>().map(|s| s.map(|s| s as f32 * scale)).collect::<Result<_, _>>()?
        }
    };
    Ok(resample(&downmix(&interleaved, spec.channels as usize), spec.sample_rate, SAMPLE_RATE))
}

/// Records from the default input, showing the level as it goes.
fn record(seconds: f32) -> Result<Vec<f32>> {
    println!("input devices: {}", input_devices()?.join(", "));
    let recorder = Recorder::start(None)?;
    println!("recording {seconds} s from {}", recorder.device_name());
    while recorder.duration() < seconds {
        let bar = "#".repeat((recorder.level() * 40.0).round() as usize);
        print!("\r{:5.1} s |{bar:<40}|", recorder.duration());
        let _ = std::io::stdout().flush();
        std::thread::sleep(Duration::from_millis(50));
    }
    println!();
    let recording = recorder.finish();
    println!("captured {:.2} s at {} Hz", recording.seconds(), recording.sample_rate);
    Ok(recording.into_16k())
}
