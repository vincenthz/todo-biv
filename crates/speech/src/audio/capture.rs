//! Recording from an input device with cpal.

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU32, Ordering},
};

use anyhow::{Context, Result, anyhow, bail};
use cpal::{
    FromSample, SampleFormat, SizedSample, Stream,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};

use super::{downmix, resample::resample};
use crate::whisper::mel::SAMPLE_RATE;

/// The longest recording kept, in seconds; what comes after is dropped, so
/// a forgotten recording does not grow without bound.
pub const MAX_SECONDS: u32 = 600;

/// Names of the available input devices.
pub fn input_devices() -> Result<Vec<String>> {
    let host = cpal::default_host();
    Ok(host
        .input_devices()?
        .filter_map(|device| device.description().ok().map(|d| d.name().to_string()))
        .collect())
}

#[derive(Default)]
struct Shared {
    samples: Mutex<Vec<f32>>,
    /// Peak amplitude of the latest callback, as `f32` bits.
    level: AtomicU32,
}

/// An open input stream accumulating mono samples at the device rate.
/// Dropping it stops the recording.
pub struct Recorder {
    _stream: Stream,
    shared: Arc<Shared>,
    sample_rate: u32,
    device_name: String,
}

impl Recorder {
    /// Starts recording from the named device, or the default input.
    pub fn start(device_name: Option<&str>) -> Result<Self> {
        let host = cpal::default_host();
        let device = match device_name {
            Some(name) => host
                .input_devices()?
                .find(|d| d.description().is_ok_and(|desc| desc.name() == name))
                .with_context(|| format!("no input device named {name:?}"))?,
            None => host.default_input_device().context("no default input device")?,
        };
        let device_name = device.description().map(|d| d.name().to_string()).unwrap_or_default();
        let supported = device.default_input_config().context("querying the input format")?;
        let config = supported.config();
        let channels = config.channels as usize;
        let shared = Arc::new(Shared::default());
        let limit = config.sample_rate as usize * MAX_SECONDS as usize;
        let stream = match supported.sample_format() {
            SampleFormat::F32 => build::<f32>(&device, config, channels, limit, &shared),
            SampleFormat::I16 => build::<i16>(&device, config, channels, limit, &shared),
            SampleFormat::I32 => build::<i32>(&device, config, channels, limit, &shared),
            SampleFormat::U16 => build::<u16>(&device, config, channels, limit, &shared),
            format => bail!("unsupported input sample format {format:?}"),
        }?;
        stream.play().context("starting the input stream")?;
        Ok(Self { _stream: stream, shared, sample_rate: config.sample_rate, device_name })
    }

    /// Name of the device being recorded.
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Peak amplitude of the latest input buffer, 0 to 1.
    pub fn level(&self) -> f32 {
        f32::from_bits(self.shared.level.load(Ordering::Relaxed)).min(1.0)
    }

    /// Seconds recorded so far.
    pub fn duration(&self) -> f32 {
        self.shared.samples.lock().map_or(0, |s| s.len()) as f32 / self.sample_rate.max(1) as f32
    }

    /// Stops recording and returns the audio at the device rate.
    pub fn finish(self) -> Recording {
        let Self { _stream, shared, sample_rate, .. } = self;
        drop(_stream);
        let samples = std::mem::take(&mut *shared.samples.lock().unwrap_or_else(|e| e.into_inner()));
        Recording { samples, sample_rate }
    }
}

/// Mono audio as captured.
pub struct Recording {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

impl Recording {
    /// Length in seconds.
    pub fn seconds(&self) -> f32 {
        self.samples.len() as f32 / self.sample_rate.max(1) as f32
    }

    /// Peak amplitude, 0 to 1. Exactly 0 means the device delivered nothing
    /// but zeros: a real microphone always picks up some noise, so the input
    /// is muted or, on macOS, the app was denied the microphone.
    pub fn peak(&self) -> f32 {
        self.samples.iter().fold(0f32, |peak, s| peak.max(s.abs())).min(1.0)
    }

    /// Resamples to Whisper's 16 kHz.
    pub fn into_16k(self) -> Vec<f32> {
        resample(&self.samples, self.sample_rate, SAMPLE_RATE)
    }
}

fn build<T>(device: &cpal::Device, config: cpal::StreamConfig, channels: usize, limit: usize, shared: &Arc<Shared>) -> Result<Stream>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    let shared = shared.clone();
    device
        .build_input_stream::<T, _, _>(
            config,
            move |data: &[T], _| {
                let interleaved: Vec<f32> = data.iter().map(|&s| f32::from_sample_(s)).collect();
                let mono = downmix(&interleaved, channels);
                let peak = mono.iter().fold(0f32, |peak, s| peak.max(s.abs()));
                shared.level.store(peak.to_bits(), Ordering::Relaxed);
                if let Ok(mut samples) = shared.samples.lock() {
                    let room = limit.saturating_sub(samples.len());
                    samples.extend_from_slice(&mono[..mono.len().min(room)]);
                }
            },
            |error| eprintln!("audio input error: {error}"),
            None,
        )
        .map_err(|e| anyhow!("opening the input stream: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peak_tells_silence_from_nothing() {
        let recording = |samples: Vec<f32>| Recording { samples, sample_rate: 16_000 };
        assert_eq!(recording(vec![0.0; 1600]).peak(), 0.0);
        assert_eq!(recording(vec![0.0, -0.25, 0.1, 2.0]).peak(), 1.0);
        assert!(recording(vec![1e-4, -3e-4]).peak() > 0.0);
    }
}
