//! The periodic review: once the list has settled after a change, or when
//! the model's plan is older than the review interval, the model is asked
//! what to do next. Requests come first: a review is only sent once the list
//! and the model have been quiet for a while (the user may be about to ask
//! something, and a review's prompt cannot be interrupted while it is read),
//! and one interrupted by a request is sent again later.

use std::time::{Duration, Instant};

use chrono::Local;
use gpui::{Context, Entity, Subscription, Task};

use crate::{
    services::model::{Kind, ModelEvent, ModelService},
    state::TodoState,
};

/// How long the list must stay unchanged, and the model unasked, before the
/// list is reviewed
pub const SETTLE: Duration = Duration::from_secs(20);
/// How often whether a review is due is checked
const TICK: Duration = Duration::from_secs(5);
/// How long to wait after a failed review before trying again
const RETRY: Duration = Duration::from_secs(120);

pub struct Reviewer {
    state: Entity<TodoState>,
    model: Entity<ModelService>,
    /// The review in progress and the revision of the list it was asked about
    pending: Option<(u64, u64)>,
    /// The revision of the list the last review was about
    reviewed: Option<u64>,
    failed_at: Option<Instant>,
    pub last_error: Option<String>,
    _ticker: Option<Task<()>>,
    _subscription: Subscription,
}

impl Reviewer {
    pub fn new(state: Entity<TodoState>, model: Entity<ModelService>, cx: &mut Context<Self>) -> Self {
        let subscription = cx.subscribe(&model, |this, _, event: &ModelEvent, cx| this.on_model(event, cx));
        // A plan the model wrote last time stands until the list changes.
        let reviewed = {
            let state = state.read(cx);
            state.store.plan.as_ref().filter(|p| p.from_model).map(|_| state.revision)
        };
        Self { state, model, pending: None, reviewed, failed_at: None, last_error: None, _ticker: None, _subscription: subscription }
    }

    /// Starts checking periodically whether a review is due
    pub fn start(&mut self, cx: &mut Context<Self>) {
        self._ticker = Some(cx.spawn(async move |this, cx| {
            loop {
                cx.background_executor().timer(TICK).await;
                if this.update(cx, |this, cx| this.tick(cx)).is_err() {
                    break;
                }
            }
        }));
    }

    pub fn is_reviewing(&self) -> bool {
        self.pending.is_some()
    }

    fn tick(&mut self, cx: &mut Context<Self>) {
        if self.is_due(cx) {
            self.review(cx);
        }
    }

    fn is_due(&self, cx: &Context<Self>) -> bool {
        let model = self.model.read(cx);
        let state = self.state.read(cx);
        if self.pending.is_some() || !model.is_ready() || model.is_busy() || !state.store.tasks.iter().any(|t| t.is_open()) {
            return false;
        }
        if self.failed_at.is_some_and(|at| at.elapsed() < RETRY) {
            return false;
        }
        let quiet = state.changed_at.elapsed() >= SETTLE && model.idle_for().is_some_and(|idle| idle >= SETTLE);
        let changed = self.reviewed != Some(state.revision);
        let interval = chrono::Duration::minutes(state.settings.review_minutes.max(1) as i64);
        let stale = match state.store.plan.as_ref().filter(|p| p.from_model) {
            Some(plan) => Local::now() - plan.generated >= interval,
            None => true,
        };
        quiet && (changed || stale)
    }

    /// Asks the model for a review now, if it is loaded
    pub fn review(&mut self, cx: &mut Context<Self>) {
        if self.pending.is_some() {
            return;
        }
        let (tasks, revision) = {
            let state = self.state.read(cx);
            (state.store.tasks.clone(), state.revision)
        };
        if let Some(id) = self.model.update(cx, |model, cx| model.review(tasks, cx)) {
            self.pending = Some((id, revision));
            cx.notify();
        }
    }

    fn on_model(&mut self, event: &ModelEvent, cx: &mut Context<Self>) {
        let Some((pending, revision)) = self.pending else { return };
        match event {
            ModelEvent::Reviewed { id, plan, .. } if *id == pending => {
                self.pending = None;
                self.reviewed = Some(revision);
                self.failed_at = None;
                self.last_error = None;
                let plan = plan.clone();
                self.state.update(cx, |state, cx| state.set_plan(plan, cx));
            }
            ModelEvent::Failed { id, kind: Kind::Review, error } if *id == pending => {
                self.pending = None;
                // An interrupted review is asked again once the list settles.
                if error != "cancelled" {
                    self.failed_at = Some(Instant::now());
                    self.last_error = Some(error.clone());
                }
            }
            _ => return,
        }
        cx.notify();
    }
}
