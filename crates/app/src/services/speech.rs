//! Speech recognition as an app-wide service. A worker thread owns the
//! Whisper model, loaded only once the microphone mode is on; views send it
//! recordings and receive transcripts as events tagged with the id they
//! were given.
//!
//! Every load and unload starts a new generation: the worker gives up a
//! load (its download included) as soon as a newer one is asked for, and
//! what it reports about an older one is ignored.

use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use std::thread;
use std::time::Instant;

use anyhow::{Result, bail};
use gpui::{Context, EventEmitter, Task};
use speech2text::{
    audio::capture::Recording,
    whisper::{
        Transcriber, Transcript, default_device,
        hub::{self, Checkpoint, DownloadProgress},
    },
};

#[derive(Clone, Debug, PartialEq)]
pub enum SpeechStatus {
    /// The microphone mode is off; nothing is loaded
    Off,
    Downloading { file: String, percent: f32 },
    Loading,
    Ready { device: String },
    Failed(String),
}

pub enum SpeechEvent {
    Transcribed { id: u64, transcript: Transcript },
    Failed { id: u64, error: String },
}

pub struct SpeechService {
    /// None when the service runs without a worker (tests, screenshots)
    jobs: Option<async_channel::Sender<Job>>,
    pub model: &'static Checkpoint,
    pub status: SpeechStatus,
    /// The latest load or unload, shared with the worker
    generation: Arc<AtomicU64>,
    next_id: u64,
    _updates: Option<Task<()>>,
}

impl EventEmitter<SpeechEvent> for SpeechService {}

impl SpeechService {
    /// Starts the worker; with `enabled`, `model` is loaded (downloaded first
    /// if needed) right away
    pub fn new(model: &'static Checkpoint, enabled: bool, cx: &mut Context<Self>) -> Self {
        let (jobs, job_receiver) = async_channel::unbounded();
        let (update_sender, updates) = async_channel::unbounded();
        let mut service = Self::without_worker(model);
        let generation = service.generation.clone();
        let spawned = thread::Builder::new().name("whisper".into()).spawn(move || run(job_receiver, update_sender, generation));
        match spawned {
            Ok(_) => {
                service.jobs = Some(jobs);
                service._updates = Some(cx.spawn(async move |this, cx| {
                    while let Ok(update) = updates.recv().await {
                        if this.update(cx, |this, cx| this.apply(update, cx)).is_err() {
                            break;
                        }
                    }
                }));
                if enabled {
                    service.load(cx);
                }
            }
            Err(error) => service.status = SpeechStatus::Failed(format!("Could not start speech recognition: {error}")),
        }
        service
    }

    pub fn without_worker(model: &'static Checkpoint) -> Self {
        Self { jobs: None, model, status: SpeechStatus::Off, generation: Arc::default(), next_id: 0, _updates: None }
    }

    pub fn is_ready(&self) -> bool {
        matches!(self.status, SpeechStatus::Ready { .. })
    }

    pub fn is_busy(&self) -> bool {
        matches!(self.status, SpeechStatus::Downloading { .. } | SpeechStatus::Loading)
    }

    /// Turns recognition on (loading the model, downloading it if needed) or
    /// off (freeing its memory)
    pub fn set_enabled(&mut self, enabled: bool, cx: &mut Context<Self>) {
        if enabled {
            if !self.is_ready() && !self.is_busy() {
                self.load(cx);
            }
        } else if self.status != SpeechStatus::Off {
            self.status = SpeechStatus::Off;
            self.next_generation();
            self.send(Job::Unload);
            cx.notify();
        }
    }

    /// Downloads if needed, then loads the current model
    pub fn load(&mut self, cx: &mut Context<Self>) {
        self.status = SpeechStatus::Loading;
        let generation = self.next_generation();
        self.send(Job::Load { model: self.model, generation });
        cx.notify();
    }

    /// Supersedes the load in progress, if any
    fn next_generation(&self) -> u64 {
        self.generation.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// Recognizes with `model` from now on, loading it if recognition is on
    pub fn select(&mut self, model: &'static Checkpoint, cx: &mut Context<Self>) {
        if model == self.model {
            return;
        }
        self.model = model;
        if self.status != SpeechStatus::Off {
            self.load(cx);
        }
        cx.notify();
    }

    /// Queues a recording; the result arrives as a [`SpeechEvent`] with the
    /// returned id
    pub fn transcribe(&mut self, recording: Recording) -> u64 {
        self.next_id += 1;
        self.send(Job::Transcribe { id: self.next_id, recording });
        self.next_id
    }

    fn send(&self, job: Job) {
        if let Some(jobs) = &self.jobs {
            let _ = jobs.try_send(job);
        }
    }

    fn apply(&mut self, update: Update, cx: &mut Context<Self>) {
        match update {
            // What a load since superseded (turned off, another model) says is not shown.
            Update::Load { generation, .. } if generation != self.generation.load(Ordering::SeqCst) => {}
            Update::Load { status, .. } => self.status = status,
            Update::Transcribed { id, transcript } => cx.emit(SpeechEvent::Transcribed { id, transcript }),
            Update::TranscribeFailed { id, error } => cx.emit(SpeechEvent::Failed { id, error }),
        }
        cx.notify();
    }

    /// Sets the status directly, for screenshots
    #[cfg(test)]
    pub fn set_status(&mut self, status: SpeechStatus) {
        self.status = status;
    }
}

enum Job {
    Load { model: &'static Checkpoint, generation: u64 },
    Unload,
    Transcribe { id: u64, recording: Recording },
}

enum Update {
    /// The status a load reached (downloading, loading, ready or failed)
    Load { generation: u64, status: SpeechStatus },
    Transcribed { id: u64, transcript: Transcript },
    TranscribeFailed { id: u64, error: String },
}

fn run(jobs: async_channel::Receiver<Job>, updates: async_channel::Sender<Update>, latest: Arc<AtomicU64>) {
    let send = |update| {
        let _ = updates.send_blocking(update);
    };
    let mut transcriber = None;
    while let Ok(job) = jobs.recv_blocking() {
        match job {
            Job::Unload => transcriber = None,
            // Turned off, or another model asked for, since: the newer job follows.
            Job::Load { generation, .. } if generation != latest.load(Ordering::SeqCst) => {}
            Job::Load { model, generation } => {
                transcriber = None;
                let started = Instant::now();
                let superseded = || latest.load(Ordering::SeqCst) != generation;
                let status = |status| send(Update::Load { generation, status });
                match load(model, &status, &superseded) {
                    Ok(loaded) => {
                        let device = if loaded.device().is_cpu() { "CPU" } else { "GPU" };
                        eprintln!("whisper: {} ready on the {device} in {:.1}s", model.name, started.elapsed().as_secs_f32());
                        status(SpeechStatus::Ready { device: device.into() });
                        transcriber = Some(loaded);
                    }
                    Err(_) if superseded() => eprintln!("whisper: loading {} given up", model.name),
                    Err(error) => {
                        eprintln!("whisper: cannot load {}: {error:#}", model.name);
                        status(SpeechStatus::Failed(format!("{error:#}")));
                    }
                }
            }
            Job::Transcribe { id, recording } => {
                let Some(transcriber) = &transcriber else {
                    send(Update::TranscribeFailed { id, error: "The speech model is not loaded.".into() });
                    continue;
                };
                match transcriber.transcribe(&recording.into_16k()) {
                    Ok(transcript) => send(Update::Transcribed { id, transcript }),
                    Err(error) => send(Update::TranscribeFailed { id, error: format!("{error:#}") }),
                }
            }
        }
    }
}

/// Downloads `model` if needed and loads it, unless `superseded` first
fn load(model: &'static Checkpoint, status: &dyn Fn(SpeechStatus), superseded: &dyn Fn() -> bool) -> Result<Transcriber> {
    let dir = hub::fetch_unless(model, &mut |progress| status(downloading(progress)), superseded)?;
    status(SpeechStatus::Loading);
    let transcriber = Transcriber::load(&dir, default_device())?;
    if superseded() {
        bail!(hub::CANCELLED);
    }
    // The first run compiles GPU kernels; do it now rather than on the first
    // request (transcribing silence would not: it skips the model)
    transcriber.warm_up()?;
    Ok(transcriber)
}

fn downloading(progress: DownloadProgress) -> SpeechStatus {
    let percent = if progress.total == 0 { 0.0 } else { 100.0 * progress.downloaded as f32 / progress.total as f32 };
    SpeechStatus::Downloading { file: progress.file, percent }
}

#[cfg(test)]
mod tests {
    use gpui::{AppContext as _, TestAppContext};

    use super::*;

    #[test]
    fn a_load_superseded_before_it_starts_is_skipped() {
        let (jobs, job_receiver) = async_channel::unbounded();
        let (update_sender, updates) = async_channel::unbounded();
        let latest = Arc::new(AtomicU64::new(2));
        let worker = thread::spawn(move || run(job_receiver, update_sender, latest));
        let model = hub::checkpoint_or_default("small");
        let recording = Recording { samples: vec![0.1; 16_000], sample_rate: 16_000 };
        for job in [Job::Load { model, generation: 1 }, Job::Unload, Job::Transcribe { id: 5, recording }] {
            assert!(jobs.send_blocking(job).is_ok());
        }
        // nothing about the load: the first update is the transcription's
        assert!(matches!(updates.recv_blocking(), Ok(Update::TranscribeFailed { id: 5, .. })));
        drop(jobs);
        assert!(worker.join().is_ok());
    }

    #[test]
    fn what_a_superseded_load_reports_is_ignored() {
        let mut cx = TestAppContext::single();
        let speech = cx.new(|_| SpeechService::without_worker(hub::checkpoint_or_default("small")));
        cx.update(|cx| {
            speech.update(cx, |speech, cx| {
                speech.load(cx);
                speech.set_enabled(false, cx);
                speech.apply(Update::Load { generation: 1, status: SpeechStatus::Ready { device: "GPU".into() } }, cx);
                assert_eq!(speech.status, SpeechStatus::Off);
                speech.set_enabled(true, cx);
                assert_eq!(speech.status, SpeechStatus::Loading);
                speech.apply(Update::Load { generation: 1, status: SpeechStatus::Ready { device: "GPU".into() } }, cx);
                assert_eq!(speech.status, SpeechStatus::Loading);
                speech.apply(Update::Load { generation: 3, status: SpeechStatus::Ready { device: "GPU".into() } }, cx);
                assert!(speech.is_ready());
            });
        });
    }
}
