//! The list on disk: one JSON file in the app's data directory, written
//! atomically (temporary file, then rename) after every change.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::Context;
use chrono::{DateTime, Local};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;

use crate::heuristics;
use crate::plan::Plan;
use crate::task::{NewTask, Op, Priority, Status, Task, TaskId, TaskPatch, normalize_category};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Store {
    #[serde(default)]
    pub tasks: Vec<Task>,
    /// The id the next task gets; ids are never reused, even after a delete
    #[serde(default)]
    next_id: u64,
    /// The last review, kept so the app opens with a plan. One that cannot
    /// be read (from another version) is dropped rather than failing the list.
    #[serde(default, deserialize_with = "lenient_plan")]
    pub plan: Option<Plan>,
    #[serde(skip)]
    path: Option<PathBuf>,
}

/// Where the app keeps its files: `$TODO_BIV_DATA` if set, else
/// `<data dir>/todo-biv` (on macOS, ~/Library/Application Support/todo-biv)
pub fn data_dir() -> PathBuf {
    match std::env::var_os("TODO_BIV_DATA") {
        Some(dir) if !dir.is_empty() => PathBuf::from(dir),
        _ => dirs::data_dir()
            .unwrap_or_else(std::env::temp_dir)
            .join("todo-biv"),
    }
}

impl Store {
    /// `tasks.json` in [`data_dir`]
    pub fn default_path() -> PathBuf {
        data_dir().join("tasks.json")
    }

    /// Loads the list at `path`, or an empty one if the file does not exist
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let mut store = match fs::read(path) {
            Ok(bytes) => serde_json::from_slice::<Store>(&bytes)
                .with_context(|| format!("reading {}", path.display()))?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Store::default(),
            Err(e) => return Err(e).with_context(|| format!("opening {}", path.display())),
        };
        store.next_id = store.next_id.max(store.max_id() + 1);
        store.path = Some(path.to_path_buf());
        Ok(store)
    }

    /// Loads the list at `path` like [`Store::load`], but always gives a list
    /// to work with, and a message for the user when it is not the one on
    /// disk. A file that is not entirely readable is copied to
    /// `tasks.json.unreadable-<time>` first, then the tasks that can be read
    /// are kept (none if it is not JSON) and saved over it at the next change.
    /// A file that cannot be opened (permissions) gives a list that is never
    /// saved, so it is not overwritten.
    pub fn load_or_recover(path: &Path, now: DateTime<Local>) -> (Self, Option<String>) {
        let error = match Self::load(path) {
            Ok(store) => return (store, None),
            Err(error) => error,
        };
        let Ok(bytes) = fs::read(path) else {
            let message = format!(
                "The list could not be opened ({error:#}). Changes made now will not be saved."
            );
            return (Self::in_memory(), Some(message));
        };
        let aside = path.with_extension(format!("json.unreadable-{}", now.format("%Y%m%d-%H%M%S")));
        if let Err(copy_error) = fs::copy(path, &aside) {
            let message = format!(
                "The list could not be read ({error:#}), nor copied aside ({copy_error}). \
                 Changes made now will not be saved."
            );
            return (Self::in_memory(), Some(message));
        }
        let mut store = Self::salvage(&bytes);
        store.path = Some(path.to_path_buf());
        let kept = match store.tasks.len() {
            0 => "a new list was started".to_string(),
            1 => "the 1 task that could be read was kept".to_string(),
            n => format!("the {n} tasks that could be read were kept"),
        };
        let message = format!(
            "The list could not be read entirely ({error:#}). It was copied to {}, and {kept}.",
            aside.display()
        );
        (store, Some(message))
    }

    /// The tasks of a list file that can be read, each on its own; ids are
    /// not given again even for the tasks left out
    fn salvage(bytes: &[u8]) -> Self {
        #[derive(Deserialize)]
        struct Raw {
            #[serde(default)]
            tasks: Vec<Value>,
            #[serde(default)]
            next_id: Value,
            #[serde(default)]
            plan: Value,
        }
        let Ok(raw) = serde_json::from_slice::<Raw>(bytes) else {
            return Self::default();
        };
        let seen_id = raw
            .tasks
            .iter()
            .filter_map(|t| t.get("id").and_then(Value::as_u64))
            .max()
            .unwrap_or(0);
        let tasks: Vec<Task> = raw
            .tasks
            .into_iter()
            .filter_map(|t| serde_json::from_value(t).ok())
            .collect();
        let next_id = raw.next_id.as_u64().unwrap_or(0).max(seen_id + 1);
        let plan = serde_json::from_value(raw.plan).ok();
        Self {
            tasks,
            next_id,
            plan,
            path: None,
        }
    }

    /// A list that is never saved (tests, previews)
    pub fn in_memory() -> Self {
        Self::default()
    }

    /// Where the list is saved; `None` for an in-memory one
    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    /// Writes the list to its file: a temporary file next to it, then a rename,
    /// so a crash never leaves half a list. Does nothing for an in-memory list.
    pub fn save(&self) -> anyhow::Result<()> {
        let Some(path) = &self.path else {
            return Ok(());
        };
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        }
        let json = serde_json::to_vec_pretty(self)?;
        let tmp = path.with_extension("json.tmp");
        let mut file =
            fs::File::create(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
        file.write_all(&json)
            .and_then(|()| file.sync_all())
            .with_context(|| format!("writing {}", tmp.display()))?;
        fs::rename(&tmp, path)
            .with_context(|| format!("renaming {} to {}", tmp.display(), path.display()))
    }

    /// Puts back the tasks of `snapshot`, an earlier clone of this list. The
    /// plan and the ids given since stay: a task added after the snapshot
    /// keeps its id to itself, even once undone.
    pub fn restore(&mut self, snapshot: Store) {
        let next_id = self.next_id.max(self.max_id() + 1).max(snapshot.next_id);
        self.tasks = snapshot.tasks;
        self.next_id = next_id;
    }

    pub fn get(&self, id: TaskId) -> Option<&Task> {
        self.tasks.iter().find(|t| t.id == id)
    }

    /// The default categories followed by the others in use
    pub fn categories(&self) -> Vec<String> {
        crate::task::categories(&self.tasks)
    }

    /// Applies `ops` in order, returning the ids of the tasks they touched.
    /// Ops naming unknown ids are skipped. New tasks get the next id, `now`
    /// as creation date, and a category and priority when the op has none.
    pub fn apply(&mut self, ops: &[Op], source: &str, now: DateTime<Local>) -> Vec<TaskId> {
        let mut touched = Vec::new();
        for op in ops {
            let id = match op {
                Op::Add(new) => self.add(new, source, now),
                Op::Update { id, patch } => self.get(*id).is_some().then(|| {
                    self.update(*id, patch, now);
                    *id
                }),
                Op::Complete { id } => self.task_mut(*id).map(|task| {
                    if task.status != Status::Done {
                        task.status = Status::Done;
                        task.completed = Some(now);
                    }
                    task.id
                }),
                Op::Reopen { id } => self.task_mut(*id).map(|task| {
                    task.status = Status::Todo;
                    task.completed = None;
                    task.id
                }),
                Op::Delete { id } => {
                    let before = self.tasks.len();
                    self.tasks.retain(|t| t.id != *id);
                    (self.tasks.len() < before).then_some(*id)
                }
            };
            if let Some(id) = id
                && !touched.contains(&id)
            {
                touched.push(id);
            }
        }
        touched
    }

    fn task_mut(&mut self, id: TaskId) -> Option<&mut Task> {
        self.tasks.iter_mut().find(|t| t.id == id)
    }

    fn max_id(&self) -> u64 {
        self.tasks.iter().map(|t| t.id.0).max().unwrap_or(0)
    }

    fn add(&mut self, new: &NewTask, source: &str, now: DateTime<Local>) -> Option<TaskId> {
        let title = new.title.trim();
        if title.is_empty() {
            return None;
        }
        let text = format!("{title} {}", new.notes);
        let category = new
            .category
            .as_deref()
            .and_then(|c| normalize_category(c, &self.tasks))
            .or_else(|| heuristics::category(&text).map(str::to_string))
            .unwrap_or_else(|| "Personal".into());
        let priority = new
            .priority
            .or_else(|| heuristics::priority(&text))
            .unwrap_or_else(|| match new.due {
                Some(due) if due <= now.date_naive() => Priority::High,
                _ => Priority::Medium,
            });
        let id = TaskId(self.next_id.max(self.max_id() + 1));
        self.next_id = id.0 + 1;
        self.tasks.push(Task {
            id,
            title: title.to_string(),
            notes: new.notes.trim().to_string(),
            category,
            priority,
            status: Status::Todo,
            due: new.due,
            tags: clean_tags(&new.tags),
            estimate_minutes: new.estimate_minutes,
            created: now,
            completed: None,
            source: source.trim().to_string(),
        });
        Some(id)
    }

    fn update(&mut self, id: TaskId, patch: &TaskPatch, now: DateTime<Local>) {
        let category = patch
            .category
            .as_deref()
            .and_then(|c| normalize_category(c, &self.tasks));
        let Some(task) = self.task_mut(id) else {
            return;
        };
        if let Some(title) = patch
            .title
            .as_deref()
            .map(str::trim)
            .filter(|t| !t.is_empty())
        {
            task.title = title.to_string();
        }
        if let Some(notes) = &patch.notes {
            task.notes = notes.trim().to_string();
        }
        if let Some(category) = category {
            task.category = category;
        }
        if let Some(priority) = patch.priority {
            task.priority = priority;
        }
        if let Some(status) = patch.status {
            match status {
                Status::Done if task.status != Status::Done => task.completed = Some(now),
                Status::Done => {}
                _ => task.completed = None,
            }
            task.status = status;
        }
        if let Some(due) = patch.due {
            task.due = due;
        }
        if let Some(tags) = &patch.tags {
            task.tags = clean_tags(tags);
        }
        if let Some(minutes) = patch.estimate_minutes {
            task.estimate_minutes = Some(minutes);
        }
    }
}

/// The plan, or none if it cannot be read
fn lenient_plan<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<Plan>, D::Error> {
    let value = Value::deserialize(deserializer)?;
    Ok(serde_json::from_value(value).ok())
}

/// Tags trimmed, without a leading '#', blanks and repeats
fn clean_tags(tags: &[String]) -> Vec<String> {
    let mut clean: Vec<String> = Vec::new();
    for tag in tags {
        let tag = tag.trim().trim_start_matches('#').trim();
        if !tag.is_empty() && !clean.iter().any(|t| t.eq_ignore_ascii_case(tag)) {
            clean.push(tag.to_string());
        }
    }
    clean
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{NaiveDate, TimeZone};

    fn now() -> DateTime<Local> {
        Local
            .from_local_datetime(
                &NaiveDate::from_ymd_opt(2026, 10, 30)
                    .unwrap()
                    .and_hms_opt(9, 0, 0)
                    .unwrap(),
            )
            .unwrap()
    }

    fn add(title: &str) -> Op {
        Op::Add(NewTask {
            title: title.into(),
            ..NewTask::default()
        })
    }

    /// A fresh directory under the system temporary directory
    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("todo-core-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn add_fills_id_category_priority() {
        let mut store = Store::in_memory();
        let ops = [
            add("Buy milk"),
            Op::Add(NewTask {
                title: "  ship it ".into(),
                category: Some(" work".into()),
                priority: Some(Priority::Low),
                ..NewTask::default()
            }),
            Op::Add(NewTask {
                title: "Pay rent".into(),
                due: Some(now().date_naive()),
                ..NewTask::default()
            }),
            Op::Add(NewTask {
                title: "Fix the thing asap".into(),
                ..NewTask::default()
            }),
            Op::Add(NewTask {
                title: "Ponder".into(),
                category: Some("side project".into()),
                ..NewTask::default()
            }),
            Op::Add(NewTask {
                title: "xyzzy".into(),
                tags: vec!["#a".into(), "A".into(), " ".into()],
                ..NewTask::default()
            }),
            add("   "),
        ];
        let ids = store.apply(&ops, " buy milk ", now());
        assert_eq!(ids, (1..=6).map(TaskId).collect::<Vec<_>>());
        let t = |id| store.get(TaskId(id)).unwrap();
        assert_eq!(
            (t(1).category.as_str(), t(1).priority, t(1).source.as_str()),
            ("Errands", Priority::Medium, "buy milk")
        );
        assert_eq!(
            (t(2).title.as_str(), t(2).category.as_str(), t(2).priority),
            ("ship it", "Work", Priority::Low)
        );
        assert_eq!(
            (t(3).category.as_str(), t(3).priority),
            ("Finance", Priority::High)
        );
        assert_eq!(t(4).priority, Priority::Urgent);
        assert_eq!(t(5).category, "Side project");
        assert_eq!(
            (t(6).category.as_str(), t(6).tags.clone()),
            ("Personal", vec!["a".to_string()])
        );
        assert_eq!(t(1).created, now());

        // a known category in another case keeps the known spelling
        store.apply(
            &[Op::Add(NewTask {
                title: "More".into(),
                category: Some("SIDE PROJECT".into()),
                ..NewTask::default()
            })],
            "",
            now(),
        );
        assert_eq!(store.get(TaskId(7)).unwrap().category, "Side project");
    }

    #[test]
    fn complete_reopen_update_delete() {
        let mut store = Store::in_memory();
        store.apply(&[add("One"), add("Two")], "", now());
        let later = now() + chrono::Duration::hours(1);

        assert_eq!(
            store.apply(
                &[
                    Op::Complete { id: TaskId(1) },
                    Op::Complete { id: TaskId(42) }
                ],
                "",
                later
            ),
            [TaskId(1)]
        );
        let one = store.get(TaskId(1)).unwrap();
        assert_eq!((one.status, one.completed), (Status::Done, Some(later)));

        store.apply(&[Op::Reopen { id: TaskId(1) }], "", later);
        let one = store.get(TaskId(1)).unwrap();
        assert_eq!((one.status, one.completed), (Status::Todo, None));

        let due = NaiveDate::from_ymd_opt(2026, 11, 6);
        let patch = TaskPatch {
            title: Some("Two!".into()),
            due: Some(due),
            status: Some(Status::Done),
            category: Some("health".into()),
            ..TaskPatch::default()
        };
        assert_eq!(
            store.apply(
                &[Op::Update {
                    id: TaskId(2),
                    patch
                }],
                "",
                later
            ),
            [TaskId(2)]
        );
        let two = store.get(TaskId(2)).unwrap();
        assert_eq!(
            (
                two.title.as_str(),
                two.due,
                two.status,
                two.completed,
                two.category.as_str()
            ),
            ("Two!", due, Status::Done, Some(later), "Health")
        );

        let clear = TaskPatch {
            due: Some(None),
            status: Some(Status::InProgress),
            ..TaskPatch::default()
        };
        store.apply(
            &[Op::Update {
                id: TaskId(2),
                patch: clear,
            }],
            "",
            later,
        );
        let two = store.get(TaskId(2)).unwrap();
        assert_eq!(
            (two.due, two.status, two.completed),
            (None, Status::InProgress, None)
        );

        assert_eq!(
            store.apply(
                &[Op::Delete { id: TaskId(2) }, Op::Delete { id: TaskId(2) }],
                "",
                later
            ),
            [TaskId(2)]
        );
        assert!(store.get(TaskId(2)).is_none());
        // the deleted id is not given again
        assert_eq!(store.apply(&[add("Three")], "", later), [TaskId(3)]);
    }

    #[test]
    fn round_trip_keeps_tasks_plan_and_ids() {
        let dir = temp_dir("round-trip");
        let path = dir.join("nested").join("tasks.json");
        let mut store = Store::load(&path).unwrap();
        assert!(store.tasks.is_empty());
        assert_eq!(store.path(), Some(path.as_path()));

        store.apply(&[add("One"), add("Two"), add("Three")], "req", now());
        store.apply(&[Op::Delete { id: TaskId(3) }], "", now());
        store.plan = Some(crate::plan::fallback_plan(&store.tasks, now()));
        store.save().unwrap();
        assert!(!path.with_extension("json.tmp").exists());

        let mut loaded = Store::load(&path).unwrap();
        assert_eq!(loaded.tasks, store.tasks);
        assert_eq!(loaded.plan, store.plan);
        assert_eq!(loaded.apply(&[add("Four")], "", now()), [TaskId(4)]);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_repairs_a_stale_next_id_and_rejects_garbage() {
        let dir = temp_dir("stale");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tasks.json");
        let task =
            r#"{"id":5,"title":"x","category":"Work","created":"2026-10-30T09:00:00+00:00"}"#;
        fs::write(&path, format!(r#"{{"tasks":[{task}]}}"#)).unwrap();
        let mut store = Store::load(&path).unwrap();
        assert_eq!(store.apply(&[add("y")], "", now()), [TaskId(6)]);

        fs::write(&path, "not json").unwrap();
        assert!(Store::load(&path).is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_bad_plan_is_dropped_not_the_list() {
        let dir = temp_dir("bad-plan");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tasks.json");
        let task =
            r#"{"id":1,"title":"x","category":"Work","created":"2026-10-30T09:00:00+00:00"}"#;
        fs::write(
            &path,
            format!(r#"{{"tasks":[{task}],"plan":{{"summary":"?"}}}}"#),
        )
        .unwrap();
        let store = Store::load(&path).unwrap();
        assert_eq!(store.tasks.len(), 1);
        assert!(store.plan.is_none());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_unreadable_list_is_copied_aside_and_what_can_be_read_kept() {
        let dir = temp_dir("recover");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tasks.json");
        let task = |id: u64, priority: &str| {
            format!(
                r#"{{"id":{id},"title":"t{id}","category":"Work","priority":"{priority}","created":"2026-10-30T09:00:00+00:00"}}"#
            )
        };
        let text = format!(
            r#"{{"tasks":[{},{}],"next_id":2}}"#,
            task(1, "high"),
            task(7, "critical")
        );
        fs::write(&path, &text).unwrap();
        assert!(Store::load(&path).is_err());

        let (mut store, message) = Store::load_or_recover(&path, now());
        assert!(message.is_some_and(|m| m.contains("the 1 task")));
        assert_eq!(store.tasks.len(), 1);
        let aside = dir.join("tasks.json.unreadable-20261030-090000");
        assert_eq!(fs::read_to_string(&aside).unwrap(), text);
        // the id of the task left out is not given again
        assert_eq!(store.apply(&[add("new")], "", now()), [TaskId(8)]);
        store.save().unwrap();
        assert_eq!(Store::load(&path).unwrap().tasks.len(), 2);

        fs::write(&path, r#"{"tasks": [ {"id": 1"#).unwrap();
        let (store, message) = Store::load_or_recover(&path, now());
        assert!(store.tasks.is_empty());
        assert_eq!(store.path(), Some(path.as_path()));
        assert!(message.is_some_and(|m| m.contains("a new list was started")));

        let missing = dir.join("none.json");
        let (store, message) = Store::load_or_recover(&missing, now());
        assert!(store.tasks.is_empty() && message.is_none());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn restoring_a_snapshot_keeps_ids_given_since() {
        let mut store = Store::in_memory();
        store.apply(&[add("One")], "", now());
        let snapshot = store.clone();
        assert_eq!(store.apply(&[add("Two")], "", now()), [TaskId(2)]);
        store.plan = Some(crate::plan::fallback_plan(&store.tasks, now()));
        store.restore(snapshot.clone());
        assert_eq!(store.tasks, snapshot.tasks);
        assert!(store.plan.is_some());
        assert_eq!(store.apply(&[add("Three")], "", now()), [TaskId(3)]);
    }

    #[test]
    fn in_memory_save_is_a_no_op() {
        Store::in_memory().save().unwrap();
    }
}
