//! The biv language model as a service. A worker thread owns the model (the
//! mmap'd GGUF, the tokenizer, the Metal or CPU engine); callers send it
//! [`Request`]s and receive [`Update`]s tagged with the request id.
//!
//! Each request is one independent turn: the context holds only the prompt of
//! [`todo_core::prompt`]. Its system prompt is the same for every request of a
//! kind, so the model's state after it is saved once (the interpretation's
//! right after loading, the review's on the first review) and each request
//! starts from a copy of it, processing only its user message. Thinking is
//! off by default (requests must feel instant); answers are greedy-ish and
//! short.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use chrono::{DateTime, Local};
use todo_core::prompt::Interpretation;
use todo_core::{Plan, Task};

mod worker;

/// The model biv's command line uses by default
pub const DEFAULT_MODEL: &str = "hf.co/unsloth/Qwen3.8-27B-GGUF:Q4_1";

#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    /// A GGUF path or an `hf.co/<owner>/<repo>:<quant>` name in the Hugging Face cache
    pub model: String,
    /// Run on the CPU even where Metal is available
    pub cpu: bool,
    /// Let the model reason before answering (slower, sometimes better)
    pub thinking: bool,
    pub context: usize,
    /// Per answer, thinking included
    pub max_tokens: usize,
    /// Start requests from the saved state after the system prompt instead
    /// of processing it every time (off only to compare)
    pub cache_prefix: bool,
    /// Always the most likely token: the same answer to the same request,
    /// for comparisons
    pub greedy: bool,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            model: DEFAULT_MODEL.into(),
            cpu: false,
            thinking: false,
            context: 8192,
            max_tokens: 1536,
            cache_prefix: true,
            greedy: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ModelStatus {
    /// Not loaded, and no load was asked for
    Idle,
    Loading,
    /// Loaded, with the interpretation's system prompt processed: the first
    /// request is as fast as the next ones
    Ready { description: String, path: PathBuf, load_time: Duration },
    Failed(String),
}

pub enum Request {
    /// (Re)load the model with this configuration
    Load(Config),
    /// A plain-language request about the list
    Interpret { id: u64, request: String, tasks: Vec<Task>, now: DateTime<Local> },
    /// A review of the list: what next, priority suggestions
    Review { id: u64, tasks: Vec<Task>, now: DateTime<Local> },
}

#[derive(Debug, Clone)]
pub enum Update {
    Status(ModelStatus),
    /// Text generated so far for a request, to show that it is progressing
    /// (the reasoning while the model thinks, then the answer); at most ten a
    /// second, and the whole answer last, before the request's result
    Partial { id: u64, text: String },
    Interpreted { id: u64, interpretation: Interpretation, elapsed: Duration },
    Reviewed { id: u64, plan: Plan, elapsed: Duration },
    Failed { id: u64, error: String },
}

/// The ids of the requests cancelled and not reached yet (or in progress),
/// shared with the worker, which forgets them as it goes past them
#[derive(Clone, Default)]
pub(crate) struct Cancelled(Arc<Mutex<HashSet<u64>>>);

impl Cancelled {
    fn set(&self) -> std::sync::MutexGuard<'_, HashSet<u64>> {
        self.0.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn insert(&self, id: u64) {
        self.set().insert(id);
    }

    pub(crate) fn contains(&self, id: u64) -> bool {
        self.set().contains(&id)
    }

    /// Forgets request `id` and those before it: requests are answered in
    /// the order of their ids
    pub(crate) fn forget_up_to(&self, id: u64) {
        self.set().retain(|&cancelled| cancelled > id);
    }
}

/// The handle to the worker thread; dropping it stops the worker after the
/// request in progress.
pub struct Assistant {
    requests: async_channel::Sender<Request>,
    cancelled: Cancelled,
}

impl Assistant {
    /// Starts the worker thread (nothing is loaded until [`Request::Load`]),
    /// returning the handle and the stream of updates
    pub fn spawn() -> (Assistant, async_channel::Receiver<Update>) {
        let (requests, request_receiver) = async_channel::unbounded();
        let (updates, update_receiver) = async_channel::unbounded();
        let cancelled = Cancelled::default();
        let worker = worker::Worker { requests: request_receiver, updates: updates.clone(), cancelled: cancelled.clone() };
        // on failure the worker is dropped, which closes the request channel: `send` returns false
        if let Err(e) = std::thread::Builder::new().name("biv".into()).spawn(move || worker::run(worker)) {
            let _ = updates.try_send(Update::Status(ModelStatus::Failed(format!("cannot start the model thread: {e}"))));
        }
        (Assistant { requests, cancelled }, update_receiver)
    }

    /// Queues a request; returns false if the worker has stopped
    pub fn send(&self, request: Request) -> bool {
        self.requests.try_send(request).is_ok()
    }

    /// Interrupts request `id` if it is being processed, or skips it once it
    /// is reached if it is queued; either way its reply is `Failed` with
    /// "cancelled". Ids are those of [`Request`]s, sent in increasing order.
    pub fn cancel(&self, id: u64) {
        self.cancelled.insert(id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requests_before_a_load_fail() {
        let (assistant, updates) = Assistant::spawn();
        assert!(assistant.send(Request::Review { id: 7, tasks: Vec::new(), now: Local::now() }));
        match updates.recv_blocking() {
            Ok(Update::Failed { id: 7, error }) => assert_eq!(error, "model not loaded"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_cancelled_request_is_skipped_once_reached() {
        let (assistant, updates) = Assistant::spawn();
        assistant.cancel(3);
        // as the worker is not loaded, a request goes straight to a reply
        assistant.send(Request::Review { id: 3, tasks: Vec::new(), now: Local::now() });
        assistant.send(Request::Review { id: 4, tasks: Vec::new(), now: Local::now() });
        match updates.recv_blocking() {
            Ok(Update::Failed { id: 3, error }) => assert_eq!(error, "cancelled"),
            other => panic!("{other:?}"),
        }
        match updates.recv_blocking() {
            Ok(Update::Failed { id: 4, error }) => assert_eq!(error, "model not loaded"),
            other => panic!("{other:?}"),
        }
        assert!(!assistant.cancelled.contains(3));
    }

    #[test]
    fn a_missing_model_reports_its_failure() {
        let (assistant, updates) = Assistant::spawn();
        let config = Config { model: "/nonexistent/model.gguf".into(), ..Config::default() };
        assistant.send(Request::Load(config));
        assert!(matches!(updates.recv_blocking(), Ok(Update::Status(ModelStatus::Loading))));
        assert!(matches!(updates.recv_blocking(), Ok(Update::Status(ModelStatus::Failed(_)))));
        // the worker is still there, waiting for another load
        assistant.send(Request::Review { id: 1, tasks: Vec::new(), now: Local::now() });
        assert!(matches!(updates.recv_blocking(), Ok(Update::Failed { id: 1, .. })));
        // dropping the handle stops it
        drop(assistant);
        assert!(updates.recv_blocking().is_err());
    }
}
