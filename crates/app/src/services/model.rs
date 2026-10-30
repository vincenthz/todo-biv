//! The language model as an app-wide service: a handle to biv's worker
//! thread, its status, and the requests queued on it. Answers arrive as
//! events tagged with the id the request was given.

use std::time::{Duration, Instant};

use assistant::{Assistant, Config, ModelStatus, Request, Update};
use chrono::Local;
use gpui::{Context, EventEmitter, Task};
use todo_core::{Plan, prompt::Interpretation};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Interpret,
    Review,
}

pub enum ModelEvent {
    Interpreted { id: u64, interpretation: Interpretation, elapsed: Duration },
    Reviewed { id: u64, plan: Plan },
    Failed { id: u64, kind: Kind, error: String },
}

pub struct ModelService {
    /// None when the service runs without a model (tests, screenshots)
    assistant: Option<Assistant>,
    pub status: ModelStatus,
    /// The configuration last asked for
    pub config: Option<Config>,
    next_id: u64,
    /// Requests sent and not answered yet, the first being processed
    queue: Vec<(u64, Kind)>,
    /// When the queue last became empty
    idle_since: Instant,
    /// How much of the answer to the request being processed has been
    /// generated, in characters
    pub partial: Option<(u64, usize)>,
    _updates: Option<Task<()>>,
}

impl EventEmitter<ModelEvent> for ModelService {}

impl ModelService {
    /// Starts the worker thread; nothing is loaded until [`Self::load`]
    pub fn spawn(cx: &mut Context<Self>) -> Self {
        let (assistant, updates) = Assistant::spawn();
        let listener = cx.spawn(async move |this, cx| {
            while let Ok(update) = updates.recv().await {
                if this.update(cx, |this, cx| this.apply(update, cx)).is_err() {
                    break;
                }
            }
        });
        Self { assistant: Some(assistant), _updates: Some(listener), ..Self::without_model() }
    }

    /// A service with no worker, whose status stays as set
    pub fn without_model() -> Self {
        Self {
            assistant: None,
            status: ModelStatus::Idle,
            config: None,
            next_id: 0,
            queue: Vec::new(),
            idle_since: Instant::now(),
            partial: None,
            _updates: None,
        }
    }

    pub fn is_ready(&self) -> bool {
        matches!(self.status, ModelStatus::Ready { .. })
    }

    pub fn is_loading(&self) -> bool {
        matches!(self.status, ModelStatus::Loading)
    }

    /// Whether requests are queued or being processed
    pub fn is_busy(&self) -> bool {
        !self.queue.is_empty()
    }

    /// How long nothing has been asked of the model; `None` while it is busy
    pub fn idle_for(&self) -> Option<Duration> {
        (!self.is_busy()).then(|| self.idle_since.elapsed())
    }

    /// (Re)loads the model with `config`; nothing happens if it is already
    /// loaded (or loading) that way
    pub fn load(&mut self, config: Config, cx: &mut Context<Self>) {
        let same = self.config.as_ref() == Some(&config) && (self.is_ready() || self.is_loading());
        if same {
            return;
        }
        self.config = Some(config.clone());
        if let Some(assistant) = &self.assistant {
            if assistant.send(Request::Load(config)) {
                self.status = ModelStatus::Loading;
            } else {
                self.status = ModelStatus::Failed("The model's thread has stopped.".into());
            }
        }
        cx.notify();
    }

    /// Asks what `request` means for the list. Reviews in progress or queued
    /// are cancelled (and are to be asked again later): requests come first.
    pub fn interpret(&mut self, request: String, tasks: Vec<todo_core::Task>, cx: &mut Context<Self>) -> Option<u64> {
        let reviews: Vec<u64> = self.queue.iter().filter(|(_, kind)| *kind == Kind::Review).map(|(id, _)| *id).collect();
        for id in reviews {
            self.cancel(id);
        }
        let id = self.next();
        self.send(id, Kind::Interpret, Request::Interpret { id, request, tasks, now: Local::now() }, cx)
    }

    pub fn review(&mut self, tasks: Vec<todo_core::Task>, cx: &mut Context<Self>) -> Option<u64> {
        let id = self.next();
        self.send(id, Kind::Review, Request::Review { id, tasks, now: Local::now() }, cx)
    }

    /// Interrupts request `id` if it is being processed, or has the worker
    /// skip it if it is still queued; it then fails with "cancelled"
    pub fn cancel(&mut self, id: u64) {
        if let Some(assistant) = &self.assistant
            && self.queue.iter().any(|(queued, _)| *queued == id)
        {
            assistant.cancel(id);
        }
    }

    fn next(&mut self) -> u64 {
        self.next_id += 1;
        self.next_id
    }

    fn send(&mut self, id: u64, kind: Kind, request: Request, cx: &mut Context<Self>) -> Option<u64> {
        let assistant = self.assistant.as_ref()?;
        if !self.is_ready() || !assistant.send(request) {
            return None;
        }
        self.queue.push((id, kind));
        cx.notify();
        Some(id)
    }

    fn finish(&mut self, id: u64) -> Option<Kind> {
        if self.partial.is_some_and(|(partial, _)| partial == id) {
            self.partial = None;
        }
        let index = self.queue.iter().position(|(queued, _)| *queued == id)?;
        let kind = self.queue.remove(index).1;
        if self.queue.is_empty() {
            self.idle_since = Instant::now();
        }
        Some(kind)
    }

    fn apply(&mut self, update: Update, cx: &mut Context<Self>) {
        match update {
            Update::Status(status) => {
                // A load drops whatever was queued before it.
                if !matches!(status, ModelStatus::Ready { .. }) {
                    for (id, kind) in std::mem::take(&mut self.queue) {
                        cx.emit(ModelEvent::Failed { id, kind, error: "the model was reloaded".into() });
                    }
                    self.partial = None;
                    self.idle_since = Instant::now();
                }
                self.status = status;
            }
            Update::Partial { id, text } => self.partial = Some((id, text.chars().count())),
            Update::Interpreted { id, interpretation, elapsed } => {
                self.finish(id);
                cx.emit(ModelEvent::Interpreted { id, interpretation, elapsed });
            }
            Update::Reviewed { id, plan, .. } => {
                self.finish(id);
                cx.emit(ModelEvent::Reviewed { id, plan });
            }
            Update::Failed { id, error } => {
                let kind = self.finish(id).unwrap_or(Kind::Interpret);
                cx.emit(ModelEvent::Failed { id, kind, error });
            }
        }
        cx.notify();
    }

    /// A short description of the status, for the title bar
    pub fn status_label(&self) -> String {
        match &self.status {
            ModelStatus::Idle => "Model not loaded".into(),
            ModelStatus::Loading => "Loading model…".into(),
            ModelStatus::Ready { .. } => {
                let cpu = self.config.as_ref().is_some_and(|c| c.cpu);
                format!("Ready ({})", if cpu || !cfg!(target_os = "macos") { "CPU" } else { "Metal" })
            }
            ModelStatus::Failed(_) => "Model failed".into(),
        }
    }

    /// Sets the status directly, for screenshots
    #[cfg(test)]
    pub fn set_status(&mut self, status: ModelStatus, config: Option<Config>) {
        self.status = status;
        self.config = config;
    }
}
