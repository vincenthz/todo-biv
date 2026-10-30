//! The list and everything the views share about it: the store and its
//! saving, the selected view and expanded task, the snapshot that undoes the
//! last request, the tasks a request just touched, and the settings.

use std::{
    collections::HashSet,
    path::PathBuf,
    time::{Duration, Instant},
};

use chrono::{DateTime, Local, NaiveDate};
use gpui::{Context, Task as GpuiTask};
use todo_core::{Op, Plan, Priority, PrioritySuggestion, Store, TaskId, TaskPatch};

use crate::{lists::View, settings::Settings};

/// How long tasks a request touched stay highlighted
const HIGHLIGHT: Duration = Duration::from_millis(2500);

/// The list as it was before a change, to go back to
struct Undo {
    store: Store,
    label: String,
}

/// A priority suggestion turned down, for the task as it was then: the
/// review is not told, so it would make it again
#[derive(Clone, Debug, PartialEq)]
struct Dismissed {
    id: TaskId,
    suggested: Priority,
    priority: Priority,
    due: Option<NaiveDate>,
}

pub struct TodoState {
    pub store: Store,
    pub settings: Settings,
    /// None for lists that are never saved (tests, screenshots)
    settings_path: Option<PathBuf>,
    pub view: View,
    pub expanded: Option<TaskId>,
    undo: Option<Undo>,
    dismissed: Vec<Dismissed>,
    highlighted: HashSet<TaskId>,
    _unhighlight: Option<GpuiTask<()>>,
    /// Bumped by every change to the tasks (not to the plan), for the
    /// reviewer to know whether the list changed since its last review
    pub revision: u64,
    pub changed_at: Instant,
    /// The latest failure to save the list, and to save the settings
    store_error: Option<String>,
    settings_error: Option<String>,
    /// What went wrong reading the files at startup, until dismissed
    load_problems: Vec<String>,
}

impl TodoState {
    /// The list and settings saved in the data directory. Files that cannot
    /// be read do not stop the app: what can be read is used, and the
    /// window says what happened (see [`Store::load_or_recover`] and
    /// [`Settings::load`]).
    pub fn load() -> Self {
        let (store, list_problem) = Store::load_or_recover(&Store::default_path(), Local::now());
        let path = Settings::default_path();
        let (settings, settings_problem) = Settings::load(&path);
        let mut state = Self::new(store, settings, Some(path));
        state.load_problems = list_problem.into_iter().chain(settings_problem).collect();
        for problem in &state.load_problems {
            eprintln!("todo-biv: {problem}");
        }
        state
    }

    /// A state that saves nothing to disk (when `store` is in memory)
    #[cfg(test)]
    pub fn in_memory(store: Store, settings: Settings) -> Self {
        Self::new(store, settings, None)
    }

    fn new(store: Store, settings: Settings, settings_path: Option<PathBuf>) -> Self {
        Self {
            store,
            settings,
            settings_path,
            view: View::NextUp,
            expanded: None,
            undo: None,
            dismissed: Vec::new(),
            highlighted: HashSet::new(),
            _unhighlight: None,
            revision: 0,
            changed_at: Instant::now(),
            store_error: None,
            settings_error: None,
            load_problems: Vec::new(),
        }
    }

    fn save(&mut self) {
        self.store_error = self.store.save().err().map(|e| format!("Could not save the list: {e:#}"));
    }

    /// What to warn about in the window: failures to save, then to read
    pub fn problems(&self) -> Vec<&str> {
        self.store_error.iter().chain(&self.settings_error).chain(&self.load_problems).map(String::as_str).collect()
    }

    /// Whether some of [`Self::problems`] can be dismissed
    pub fn has_load_problems(&self) -> bool {
        !self.load_problems.is_empty()
    }

    pub fn dismiss_load_problems(&mut self, cx: &mut Context<Self>) {
        self.load_problems.clear();
        cx.notify();
    }

    /// Shows a startup problem, for screenshots
    #[cfg(test)]
    pub fn show_load_problem(&mut self, problem: String, cx: &mut Context<Self>) {
        self.load_problems.push(problem);
        cx.notify();
    }

    fn changed(&mut self, cx: &mut Context<Self>) {
        self.revision += 1;
        self.changed_at = Instant::now();
        self.save();
        self.collapse_hidden();
        cx.notify();
    }

    /// Closes the editor of a task the view no longer shows (deleted,
    /// completed out of "Next up", dropped from the plan): it would be
    /// invisible, and Esc would go to closing it
    fn collapse_hidden(&mut self) {
        if self.expanded.is_some_and(|id| !self.shows(id)) {
            self.expanded = None;
        }
    }

    /// Whether the selected view lists task `id`
    fn shows(&self, id: TaskId) -> bool {
        let now = Local::now();
        let plan = self.plan(now);
        crate::lists::tasks_in(&self.view, &self.store.tasks, &plan, now).iter().any(|t| t.id == id)
    }

    /// Applies what a request asked, keeping the list as it was to undo it,
    /// and highlights the tasks it touched. Returns their ids: none when the
    /// ops name tasks that are gone, and then there is nothing to undo.
    pub fn apply_request(&mut self, ops: &[Op], request: &str, cx: &mut Context<Self>) -> Vec<TaskId> {
        if ops.is_empty() {
            return Vec::new();
        }
        let before = self.store.clone();
        let touched = self.store.apply(ops, request, Local::now());
        if touched.is_empty() {
            return touched;
        }
        self.undo = Some(Undo { store: before, label: format!("“{}”", request.trim()) });
        self.changed(cx);
        self.highlight(&touched, cx);
        touched
    }

    /// Applies a change made by hand (a checkbox, the editor); `undo` keeps
    /// the list as it was for destructive ones. Any other change drops what
    /// there was to undo: going back to a snapshot from before it would
    /// silently revert it too. Returns whether anything changed.
    pub fn apply_edit(&mut self, op: Op, undo: Option<String>, cx: &mut Context<Self>) -> bool {
        let before = self.store.clone();
        let touched = self.store.apply(std::slice::from_ref(&op), "", Local::now());
        if touched.is_empty() || self.store.tasks == before.tasks {
            return false;
        }
        self.undo = undo.map(|label| Undo { store: before, label });
        self.changed(cx);
        true
    }

    pub fn patch(&mut self, id: TaskId, patch: TaskPatch, cx: &mut Context<Self>) {
        self.apply_edit(Op::Update { id, patch }, None, cx);
    }

    pub fn set_done(&mut self, id: TaskId, done: bool, cx: &mut Context<Self>) {
        let op = if done { Op::Complete { id } } else { Op::Reopen { id } };
        self.apply_edit(op, None, cx);
    }

    pub fn delete(&mut self, id: TaskId, cx: &mut Context<Self>) {
        let title = self.store.get(id).map(|t| t.title.clone()).unwrap_or_default();
        if self.expanded == Some(id) {
            self.expanded = None;
        }
        self.apply_edit(Op::Delete { id }, Some(format!("deleting “{title}”")), cx);
    }

    /// What undoing would revert, if there is anything
    pub fn undo_label(&self) -> Option<&str> {
        self.undo.as_ref().map(|u| u.label.as_str())
    }

    /// Puts the list back as it was before the last request (or deletion).
    /// The plan stays, being newer than the snapshot's, and so do the ids
    /// given since: they are not given again to other tasks.
    pub fn undo(&mut self, cx: &mut Context<Self>) -> bool {
        let Some(undo) = self.undo.take() else { return false };
        self.store.restore(undo.store);
        self.highlighted.clear();
        self.changed(cx);
        true
    }

    pub fn is_highlighted(&self, id: TaskId) -> bool {
        self.highlighted.contains(&id)
    }

    fn highlight(&mut self, ids: &[TaskId], cx: &mut Context<Self>) {
        self.highlighted = ids.iter().copied().collect();
        self._unhighlight = Some(cx.spawn(async move |this, cx| {
            cx.background_executor().timer(HIGHLIGHT).await;
            let _ = this.update(cx, |this, cx| {
                this.highlighted.clear();
                cx.notify();
            });
        }));
    }

    /// Stores a review of the model's, without the suggestions dismissed
    /// for the tasks as they still are
    pub fn set_plan(&mut self, mut plan: Plan, cx: &mut Context<Self>) {
        let tasks = &self.store.tasks;
        self.dismissed.retain(|d| tasks.iter().any(|t| t.id == d.id && t.priority == d.priority && t.due == d.due));
        let dismissed = &self.dismissed;
        plan.suggestions.retain(|s| !dismissed.iter().any(|d| d.id == s.id && d.suggested == s.priority));
        self.store.plan = Some(plan);
        self.save();
        self.collapse_hidden();
        cx.notify();
    }

    /// Takes the review's advice for task `id`
    pub fn accept_suggestion(&mut self, id: TaskId, cx: &mut Context<Self>) {
        let Some(suggestion) = self.take_suggestion(id) else { return };
        let patch = TaskPatch { priority: Some(suggestion.priority), ..Default::default() };
        if !self.apply_edit(Op::Update { id, patch }, None, cx) {
            self.save();
            cx.notify();
        }
    }

    /// Turns down the review's advice for task `id`; it is not shown again
    /// until the task's priority or due date changes
    pub fn dismiss_suggestion(&mut self, id: TaskId, cx: &mut Context<Self>) {
        let Some(suggestion) = self.take_suggestion(id) else { return };
        if let Some(task) = self.store.get(id) {
            let dismissed = Dismissed { id, suggested: suggestion.priority, priority: task.priority, due: task.due };
            self.dismissed.push(dismissed);
        }
        self.save();
        cx.notify();
    }

    fn take_suggestion(&mut self, id: TaskId) -> Option<PrioritySuggestion> {
        let plan = self.store.plan.as_mut()?;
        let index = plan.suggestions.iter().position(|s| s.id == id)?;
        Some(plan.suggestions.remove(index))
    }

    /// The plan to show (see [`crate::lists::effective_plan`])
    pub fn plan(&self, now: DateTime<Local>) -> Plan {
        crate::lists::effective_plan(self.store.plan.as_ref(), &self.store.tasks, now)
    }

    pub fn select_view(&mut self, view: View, cx: &mut Context<Self>) {
        self.view = view;
        self.collapse_hidden();
        cx.notify();
    }

    pub fn toggle_expanded(&mut self, id: TaskId, cx: &mut Context<Self>) {
        self.expanded = if self.expanded == Some(id) { None } else { Some(id) };
        cx.notify();
    }

    /// Shows task `id` expanded, switching to a view that holds it if the
    /// current one does not
    pub fn reveal(&mut self, id: TaskId, cx: &mut Context<Self>) {
        let Some(task) = self.store.get(id) else { return };
        if !self.shows(id) {
            self.view = if task.is_open() { View::AllOpen } else { View::Done };
        }
        self.expanded = Some(id);
        cx.notify();
    }

    pub fn update_settings(&mut self, edit: impl FnOnce(&mut Settings), cx: &mut Context<Self>) {
        edit(&mut self.settings);
        if let Some(path) = &self.settings_path {
            self.settings_error = self.settings.save(path).err().map(|e| format!("Could not save the settings: {e:#}"));
        }
        cx.notify();
    }

    /// Where the list is saved, for the settings page
    pub fn data_dir(&self) -> Option<PathBuf> {
        self.settings_path.as_ref().and_then(|p| p.parent()).map(|p| p.to_path_buf())
    }
}

#[cfg(test)]
mod tests {
    use gpui::{AppContext as _, Entity, TestAppContext};
    use todo_core::{NewTask, Priority};

    use super::*;

    fn state(cx: &mut TestAppContext) -> Entity<TodoState> {
        cx.new(|_| TodoState::in_memory(crate::screenshots::sample_store(), Settings::default()))
    }

    #[test]
    fn a_request_is_undone_but_the_plan_stays() {
        let mut cx = TestAppContext::single();
        let state = state(&mut cx);
        let ops = [Op::Add(NewTask { title: "Call the bank".into(), ..Default::default() }), Op::Complete { id: TaskId(1) }];
        cx.update(|cx| {
            let before = state.read(cx).store.tasks.clone();
            let touched = state.update(cx, |state, cx| state.apply_request(&ops, "call the bank; slides done", cx));
            assert_eq!(touched.len(), 2);
            assert!(touched.iter().all(|id| state.read(cx).is_highlighted(*id)));
            assert_eq!(state.read(cx).undo_label(), Some("“call the bank; slides done”"));
            let revision = state.read(cx).revision;

            state.update(cx, |state, cx| state.set_plan(Plan { summary: "newer".into(), ..state.plan(Local::now()) }, cx));
            assert!(state.update(cx, |state, cx| state.undo(cx)));
            let state = state.read(cx);
            assert_eq!(state.store.tasks, before);
            assert_eq!(state.store.plan.as_ref().map(|p| p.summary.as_str()), Some("newer"));
            assert!(state.revision > revision);
            assert_eq!(state.undo_label(), None);
        });
    }

    #[test]
    fn suggestions_are_accepted_or_dismissed() {
        let mut cx = TestAppContext::single();
        let state = state(&mut cx);
        cx.update(|cx| {
            let suggested = state.read(cx).store.plan.as_ref().map(|p| p.suggestions[0].id).expect("a suggestion");
            state.update(cx, |state, cx| state.accept_suggestion(suggested, cx));
            let read = state.read(cx);
            assert_eq!(read.store.get(suggested).map(|t| t.priority), Some(Priority::High));
            assert!(read.store.plan.as_ref().is_some_and(|p| p.suggestions.is_empty()));
            // A manual change is not a request: there is nothing to undo.
            assert_eq!(read.undo_label(), None);
        });
    }

    #[test]
    fn deleting_can_be_undone() {
        let mut cx = TestAppContext::single();
        let state = state(&mut cx);
        cx.update(|cx| {
            let count = state.read(cx).store.tasks.len();
            state.update(cx, |state, cx| {
                state.toggle_expanded(TaskId(2), cx);
                state.delete(TaskId(2), cx);
            });
            assert_eq!(state.read(cx).store.tasks.len(), count - 1);
            assert_eq!(state.read(cx).expanded, None);
            assert_eq!(state.read(cx).undo_label(), Some("deleting “File the tax return”"));
            state.update(cx, |state, cx| state.undo(cx));
            assert_eq!(state.read(cx).store.tasks.len(), count);
        });
    }

    fn add(title: &str) -> Op {
        Op::Add(NewTask { title: title.into(), ..Default::default() })
    }

    #[test]
    fn a_hand_edit_after_a_request_is_not_undone_with_it() {
        let mut cx = TestAppContext::single();
        let state = state(&mut cx);
        cx.update(|cx| {
            state.update(cx, |state, cx| state.apply_request(&[add("Call the bank")], "call the bank", cx));
            // a change that changes nothing keeps the undo
            let title = state.read(cx).store.get(TaskId(1)).map(|t| t.title.clone());
            state.update(cx, |state, cx| state.patch(TaskId(1), TaskPatch { title, ..Default::default() }, cx));
            assert!(state.read(cx).undo_label().is_some());

            state.update(cx, |state, cx| state.set_done(TaskId(3), true, cx));
            assert_eq!(state.read(cx).undo_label(), None);
            assert!(!state.update(cx, |state, cx| state.undo(cx)));
            let state = state.read(cx);
            assert!(state.store.get(TaskId(3)).is_some_and(|t| !t.is_open()));
            assert!(state.store.tasks.iter().any(|t| t.title == "Call the bank"));
        });
    }

    #[test]
    fn undoing_a_request_does_not_give_its_ids_again() {
        let mut cx = TestAppContext::single();
        let state = state(&mut cx);
        cx.update(|cx| {
            let first = state.update(cx, |state, cx| state.apply_request(&[add("Call the bank")], "call the bank", cx));
            assert_eq!(first, [TaskId(12)]);
            state.update(cx, |state, cx| state.undo(cx));
            assert!(state.read(cx).store.get(TaskId(12)).is_none());
            let second = state.update(cx, |state, cx| state.apply_request(&[add("Water the plants")], "water the plants", cx));
            assert_eq!(second, [TaskId(13)]);
        });
    }

    #[test]
    fn a_request_about_tasks_that_are_gone_changes_nothing() {
        let mut cx = TestAppContext::single();
        let state = state(&mut cx);
        cx.update(|cx| {
            let revision = state.read(cx).revision;
            let touched = state.update(cx, |state, cx| state.apply_request(&[Op::Complete { id: TaskId(99) }], "the gone one is done", cx));
            assert!(touched.is_empty());
            assert_eq!(state.read(cx).revision, revision);
            assert_eq!(state.read(cx).undo_label(), None);
        });
    }

    #[test]
    fn a_dismissed_suggestion_stays_dismissed_until_the_task_changes() {
        let mut cx = TestAppContext::single();
        let state = state(&mut cx);
        cx.update(|cx| {
            let plan = state.read(cx).store.plan.clone().expect("a plan");
            let id = plan.suggestions[0].id;
            state.update(cx, |state, cx| state.dismiss_suggestion(id, cx));
            state.update(cx, |state, cx| state.set_plan(plan.clone(), cx));
            assert!(state.read(cx).plan(Local::now()).suggestions.is_empty());

            state.update(cx, |state, cx| state.patch(id, TaskPatch { priority: Some(Priority::Low), ..Default::default() }, cx));
            state.update(cx, |state, cx| state.set_plan(plan.clone(), cx));
            assert_eq!(state.read(cx).plan(Local::now()).suggestions.len(), 1);
        });
    }

    #[test]
    fn the_editor_of_a_task_that_leaves_the_view_closes() {
        let mut cx = TestAppContext::single();
        let state = state(&mut cx);
        cx.update(|cx| {
            let expand = |id, cx: &mut gpui::App| state.update(cx, |state, cx| state.toggle_expanded(TaskId(id), cx));
            expand(2, cx);
            state.update(cx, |state, cx| state.apply_request(&[Op::Complete { id: TaskId(2) }], "taxes done", cx));
            assert_eq!(state.read(cx).expanded, None);
            expand(3, cx);
            state.update(cx, |state, cx| state.apply_request(&[Op::Delete { id: TaskId(3) }], "forget mum", cx));
            assert_eq!(state.read(cx).expanded, None);
            expand(4, cx);
            state.update(cx, |state, cx| state.set_done(TaskId(4), true, cx));
            assert_eq!(state.read(cx).expanded, None);
            // a change that keeps it in view keeps it open
            expand(5, cx);
            state.update(cx, |state, cx| state.patch(TaskId(5), TaskPatch { notes: Some("2nd floor".into()), ..Default::default() }, cx));
            assert_eq!(state.read(cx).expanded, Some(TaskId(5)));
            state.update(cx, |state, cx| state.select_view(View::Done, cx));
            assert_eq!(state.read(cx).expanded, None);
        });
    }

    #[test]
    fn saving_the_settings_does_not_hide_a_failure_to_save_the_list() {
        let dir = std::env::temp_dir().join(format!("todo-biv-state-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("tasks.json");
        let mut store = Store::load(&path).expect("an empty list");
        store.apply(&[add("One")], "", Local::now());
        // a directory where the list goes: it cannot be written
        std::fs::create_dir_all(&path).expect("a directory");
        let mut cx = TestAppContext::single();
        let state = cx.new(|_| TodoState::new(store, Settings::default(), Some(dir.join("settings.json"))));
        cx.update(|cx| {
            state.update(cx, |state, cx| state.set_done(TaskId(1), true, cx));
            assert!(state.read(cx).problems().iter().any(|p| p.starts_with("Could not save the list")));
            state.update(cx, |state, cx| state.update_settings(|s| s.cpu = true, cx));
            assert!(state.read(cx).problems().iter().any(|p| p.starts_with("Could not save the list")));
        });
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn revealing_a_task_switches_to_a_view_that_holds_it() {
        let mut cx = TestAppContext::single();
        let state = state(&mut cx);
        cx.update(|cx| {
            state.update(cx, |state, cx| {
                state.select_view(View::Category("Health".into()), cx);
                state.reveal(TaskId(1), cx);
            });
            assert_eq!(state.read(cx).view, View::AllOpen);
            assert_eq!(state.read(cx).expanded, Some(TaskId(1)));
            state.update(cx, |state, cx| state.reveal(TaskId(10), cx));
            assert_eq!(state.read(cx).view, View::Done);
        });
    }
}
