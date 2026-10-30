//! A task and what describes it: category, priority, status, dates.

use chrono::{DateTime, Datelike, Local, NaiveDate};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TaskId(pub u64);

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
#[serde(rename_all = "lowercase")]
pub enum Priority {
    Low,
    #[default]
    Medium,
    High,
    Urgent,
}

impl Priority {
    pub const ALL: [Priority; 4] = [
        Priority::Urgent,
        Priority::High,
        Priority::Medium,
        Priority::Low,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Priority::Low => "Low",
            Priority::Medium => "Medium",
            Priority::High => "High",
            Priority::Urgent => "Urgent",
        }
    }

    /// Lenient parse of a model's or a person's word for a priority
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s.trim().to_ascii_lowercase().as_str() {
            "low" | "p3" | "someday" | "minor" => Priority::Low,
            "medium" | "normal" | "p2" | "mid" => Priority::Medium,
            "high" | "important" | "p1" => Priority::High,
            "urgent" | "critical" | "asap" | "p0" => Priority::Urgent,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    #[default]
    Todo,
    InProgress,
    Done,
}

impl Status {
    pub fn label(self) -> &'static str {
        match self {
            Status::Todo => "To do",
            Status::InProgress => "In progress",
            Status::Done => "Done",
        }
    }
}

/// The categories offered by default; the model may also invent others,
/// which are kept as written (capitalized).
pub const DEFAULT_CATEGORIES: [&str; 8] = [
    "Work", "Personal", "Home", "Health", "Finance", "Errands", "Learning", "Social",
];

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Task {
    pub id: TaskId,
    pub title: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub notes: String,
    pub category: String,
    #[serde(default)]
    pub priority: Priority,
    #[serde(default)]
    pub status: Status,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub due: Option<NaiveDate>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
    /// Rough effort, in minutes
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub estimate_minutes: Option<u32>,
    pub created: DateTime<Local>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completed: Option<DateTime<Local>>,
    /// The request the task came from, as typed or heard
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub source: String,
}

impl Task {
    pub fn is_open(&self) -> bool {
        self.status != Status::Done
    }
}

/// What a request asks to create; the store gives it an id and dates.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct NewTask {
    pub title: String,
    #[serde(default)]
    pub notes: String,
    #[serde(default)]
    pub category: Option<String>,
    #[serde(default)]
    pub priority: Option<Priority>,
    #[serde(default)]
    pub due: Option<NaiveDate>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub estimate_minutes: Option<u32>,
}

/// Fields a request changes on an existing task; `None` leaves one as is.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct TaskPatch {
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub notes: Option<String>,
    #[serde(default)]
    pub category: Option<String>,
    #[serde(default)]
    pub priority: Option<Priority>,
    #[serde(default)]
    pub status: Option<Status>,
    /// `Some(None)` clears the due date
    #[serde(default)]
    pub due: Option<Option<NaiveDate>>,
    #[serde(default)]
    pub tags: Option<Vec<String>>,
    #[serde(default)]
    pub estimate_minutes: Option<u32>,
}

/// One change a request makes to the list
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Op {
    Add(NewTask),
    Update { id: TaskId, patch: TaskPatch },
    Complete { id: TaskId },
    Reopen { id: TaskId },
    Delete { id: TaskId },
}

/// The default categories followed by the others used in `tasks`, in order
/// of first use
pub fn categories(tasks: &[Task]) -> Vec<String> {
    let mut all: Vec<String> = DEFAULT_CATEGORIES.iter().map(|c| c.to_string()).collect();
    for task in tasks {
        if !task.category.is_empty() && !all.iter().any(|c| c.eq_ignore_ascii_case(&task.category))
        {
            all.push(task.category.clone());
        }
    }
    all
}

/// `raw` trimmed and capitalized, or spelled like a category of [`categories`]
/// when it is one in another case. `None` when blank.
pub fn normalize_category(raw: &str, tasks: &[Task]) -> Option<String> {
    let raw = raw.trim();
    if raw.is_empty() {
        return None;
    }
    if let Some(known) = categories(tasks)
        .into_iter()
        .find(|c| c.eq_ignore_ascii_case(raw))
    {
        return Some(known);
    }
    let mut chars = raw.chars();
    let first = chars.next()?;
    Some(first.to_uppercase().chain(chars).collect())
}

/// `date` as a person says it relative to `today`: "today", "tomorrow",
/// "yesterday", the weekday within a week ahead ("Fri"), else "Fri 6 Nov"
pub fn relative_day(date: NaiveDate, today: NaiveDate) -> String {
    match (date - today).num_days() {
        0 => "today".into(),
        1 => "tomorrow".into(),
        -1 => "yesterday".into(),
        2..=6 => date.format("%a").to_string(),
        _ if date.year() == today.year() => date.format("%a %-d %b").to_string(),
        _ => date.format("%a %-d %b %Y").to_string(),
    }
}
