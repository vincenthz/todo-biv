//! The prompts given to the language model and the parsing of its replies.
//!
//! The model answers in JSON. Replies are parsed leniently: the JSON may be
//! wrapped in a code fence or surrounded by prose, enum words may vary in case,
//! dates are ISO `YYYY-MM-DD` or relative words, and ids that are not in the
//! list are dropped rather than failing the whole reply.

use std::fmt::Write;

use anyhow::bail;
use chrono::{DateTime, Days, Local, NaiveDate};
use serde_json::{Map, Value};

use crate::heuristics;
use crate::plan::{self, MAX_NEXT, Plan, PlanItem, PrioritySuggestion};
use crate::task::{
    NewTask, Op, Priority, Status, Task, TaskId, TaskPatch, categories, relative_day,
};

/// A system prompt and a user message for one model turn
#[derive(Debug, Clone, PartialEq)]
pub struct Prompt {
    pub system: String,
    pub user: String,
}

/// What the interpretation of a request came to
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Interpretation {
    pub ops: Vec<Op>,
    /// A one-line reply to show the person ("Added 2 tasks", or a question
    /// back when the request was unclear)
    pub reply: String,
}

/// The system prompt of [`interpret_prompt`]. It is the same for every
/// request, so a model can process it once and keep that state: what changes
/// from one request to the next (the date, the categories, the list) is in
/// the user message.
pub const INTERPRET_SYSTEM: &str = r#"You keep a personal to-do list. Turn the person's request into changes to the list.
The message gives today's date, the categories, the list and then the request.

Answer with only JSON: {"reply": "one short sentence on what you did", "ops": [...]}
Ops:
{"op": "add", "title": "...", "notes": "...", "category": "...", "priority": "medium", "due": "YYYY-MM-DD", "tags": ["..."], "estimate_minutes": 30}
{"op": "update", "id": 12, ...only the changed fields; "status" is todo, in_progress or done; "due": null removes the date}
{"op": "complete", "id": 12} {"op": "reopen", "id": 12} {"op": "delete", "id": 12}

Rules:
- One op per task or change: "buy milk and call the bank" is two adds.
- ids are the #numbers of the list; never invent one.
- Resolve relative dates to YYYY-MM-DD; leave "due" out when no date is given.
- category: one of the categories listed in the message, or a new one when none fits.
- priority: low, medium, high or urgent; follow words like "urgent", "important" or "no rush", else judge by stakes and deadline.
- title: a short imperative ("Call mum"); details such as a time go in notes. Leave out empty fields.
- "I did X" completes X; "move X to friday" updates its due date.
- If the request is unclear or names a task not in the list, give no ops and ask a short question in reply."#;

/// Turning a plain-language request ("remind me to call mum tomorrow, it's
/// important", "I finished the report", "move the dentist to friday") into
/// operations on the list. `tasks` are given with their ids so the model can
/// refer to them; `now` lets it resolve relative dates. The system prompt is
/// [`INTERPRET_SYSTEM`].
pub fn interpret_prompt(request: &str, tasks: &[Task], now: DateTime<Local>) -> Prompt {
    let user = format!(
        "{}\nCategories: {}\n{}\nRequest: {}",
        today_line(now),
        categories(tasks).join(", "),
        task_list(tasks, now),
        request.trim()
    );
    Prompt {
        system: INTERPRET_SYSTEM.into(),
        user,
    }
}

pub fn parse_interpretation(
    reply: &str,
    tasks: &[Task],
    now: DateTime<Local>,
) -> anyhow::Result<Interpretation> {
    let today = now.date_naive();
    let Some(value) = find_json(reply, false) else {
        bail!("no JSON in the model's reply")
    };
    let (text, raw_ops) = match &value {
        Value::Array(items) => (String::new(), items.clone()),
        Value::Object(obj) => {
            let text = text_field(
                obj,
                &[
                    "reply", "message", "response", "answer", "summary", "question",
                ],
            )
            .unwrap_or_default();
            let raw_ops = match field(
                obj,
                &[
                    "ops",
                    "operations",
                    "actions",
                    "changes",
                    "tasks",
                    "commands",
                    "op_list",
                ],
            ) {
                Some(Value::Array(items)) => items.clone(),
                Some(op @ Value::Object(_)) => vec![op.clone()],
                _ if field(obj, &["op", "action", "title"]).is_some() => vec![value.clone()],
                _ => Vec::new(),
            };
            (text, raw_ops)
        }
        _ => bail!("the model's reply is not a JSON object"),
    };

    let mut ops: Vec<Op> = Vec::new();
    for raw in &raw_ops {
        let Some(op) = raw.as_object().and_then(|obj| parse_op(obj, tasks, today)) else {
            continue;
        };
        let duplicate = ops.iter().any(|o| match (o, &op) {
            (Op::Add(a), Op::Add(b)) => a.title.eq_ignore_ascii_case(&b.title),
            (a, b) => a == b,
        });
        if !duplicate {
            ops.push(op);
        }
    }
    let reply = if text.is_empty() {
        describe_ops(&ops, tasks, today)
    } else {
        text
    };
    Ok(Interpretation { ops, reply })
}

/// The system prompt of [`review_prompt`], the same for every review (see
/// [`INTERPRET_SYSTEM`]). "up to 7" is [`MAX_NEXT`].
pub const REVIEW_SYSTEM: &str = r#"You review a personal to-do list and decide what should be done next.
The message gives today's date and the list.

Answer with ONLY a JSON object, no other text:
{"summary": "one or two sentences on where the list stands", "next": [{"id": 12, "reason": "..."}], "suggestions": [{"id": 7, "priority": "high", "reason": "..."}]}

Rules:
- next: up to 7 open tasks in the order to do them, the first one being what to do now. Weigh priority, due dates (overdue first, then due soon), tasks already in progress, effort (a quick task can go before a long one) and how long a task has been waiting.
- reason: one short sentence on why the task comes at that point.
- suggestions: only for open tasks whose priority looks wrong, e.g. overdue or due soon but low, or urgent with no date and untouched for weeks. priority is low, medium, high or urgent. Leave the list empty when the priorities look right.
- Refer to tasks by the number after # in the list. Never invent ids."#;

/// Reviewing the list: what to do next and in which order, which priorities
/// look wrong (overdue, blocking others, stale), and a short summary. The
/// system prompt is [`REVIEW_SYSTEM`].
pub fn review_prompt(tasks: &[Task], now: DateTime<Local>) -> Prompt {
    let user = format!(
        "{}\n{}\nReview the list.",
        today_line(now),
        task_list(tasks, now)
    );
    Prompt {
        system: REVIEW_SYSTEM.into(),
        user,
    }
}

pub fn parse_review(reply: &str, tasks: &[Task], now: DateTime<Local>) -> anyhow::Result<Plan> {
    let Some(value) = find_json(reply, true) else {
        bail!("no JSON in the model's reply")
    };
    let empty = Map::new();
    let (obj, raw_next) = match &value {
        Value::Array(items) => (&empty, items.clone()),
        Value::Object(obj) => {
            let next = match field(
                obj,
                &[
                    "next",
                    "order",
                    "plan",
                    "queue",
                    "next_tasks",
                    "ranking",
                    "tasks",
                ],
            ) {
                Some(Value::Array(items)) => items.clone(),
                _ => Vec::new(),
            };
            (obj, next)
        }
        _ => bail!("the model's reply is not a JSON object"),
    };
    let open = |id: TaskId| tasks.iter().find(|t| t.id == id && t.is_open());

    let mut next: Vec<PlanItem> = Vec::new();
    for raw in &raw_next {
        let (id, reason) = match raw {
            Value::Object(item) => (
                field(item, ID_KEYS).and_then(id_of),
                text_field(item, &["reason", "why", "note", "because", "explanation"])
                    .unwrap_or_default(),
            ),
            other => (id_of(other), String::new()),
        };
        let Some(task) = id.map(TaskId).and_then(open) else {
            continue;
        };
        if next.len() < MAX_NEXT && !next.iter().any(|i| i.id == task.id) {
            let reason = if reason.is_empty() {
                plan::reason(task, now)
            } else {
                reason
            };
            next.push(PlanItem {
                id: task.id,
                reason,
            });
        }
    }
    if next.is_empty() {
        next = plan::fallback_plan(tasks, now).next;
    }

    let mut suggestions: Vec<PrioritySuggestion> = Vec::new();
    if let Some(Value::Array(items)) = field(
        obj,
        &[
            "suggestions",
            "priority_changes",
            "priorities",
            "changes",
            "reprioritize",
        ],
    ) {
        for item in items.iter().filter_map(Value::as_object) {
            let Some(task) = field(item, ID_KEYS)
                .and_then(id_of)
                .map(TaskId)
                .and_then(open)
            else {
                continue;
            };
            let Some(priority) = field(item, SUGGESTED_PRIORITY_KEYS).and_then(priority_of) else {
                continue;
            };
            if priority != task.priority && !suggestions.iter().any(|s| s.id == task.id) {
                let reason = text_field(item, &["reason", "why", "because", "explanation"])
                    .unwrap_or_default();
                suggestions.push(PrioritySuggestion {
                    id: task.id,
                    priority,
                    reason,
                });
            }
        }
    }

    let summary = text_field(obj, &["summary", "overview", "reply", "message"])
        .unwrap_or_else(|| plan::summary(tasks, now));
    Ok(Plan {
        generated: now,
        summary,
        next,
        suggestions,
        from_model: true,
    })
}

/// A sentence saying what `ops` do: `Added "Call mum" (due tomorrow, high)`,
/// `Marked "Pay rent" as done`, "Added 2 tasks and completed 1"
pub fn describe_ops(ops: &[Op], tasks: &[Task], today: NaiveDate) -> String {
    let title = |id: &TaskId| {
        tasks
            .iter()
            .find(|t| t.id == *id)
            .map_or_else(|| format!("#{}", id.0), |t| t.title.clone())
    };
    match ops {
        [] => "Nothing to change.".into(),
        [Op::Add(new)] => {
            let mut details = Vec::new();
            if let Some(due) = new.due {
                details.push(format!("due {}", relative_day(due, today)));
            }
            if let Some(priority) = new.priority {
                details.push(priority.label().to_lowercase());
            }
            if details.is_empty() {
                format!("Added \"{}\".", new.title)
            } else {
                format!("Added \"{}\" ({}).", new.title, details.join(", "))
            }
        }
        [Op::Update { id, patch }] => match patch {
            TaskPatch {
                due: Some(Some(due)),
                title: None,
                notes: None,
                category: None,
                priority: None,
                status: None,
                ..
            } => {
                format!("Moved \"{}\" to {}.", title(id), relative_day(*due, today))
            }
            _ => format!("Updated \"{}\".", title(id)),
        },
        [Op::Complete { id }] => format!("Marked \"{}\" as done.", title(id)),
        [Op::Reopen { id }] => format!("Reopened \"{}\".", title(id)),
        [Op::Delete { id }] => format!("Deleted \"{}\".", title(id)),
        _ => {
            let count = |f: fn(&Op) -> bool| ops.iter().filter(|o| f(o)).count();
            let counts = [
                (count(|o| matches!(o, Op::Add(_))), "added"),
                (count(|o| matches!(o, Op::Update { .. })), "updated"),
                (count(|o| matches!(o, Op::Complete { .. })), "completed"),
                (count(|o| matches!(o, Op::Reopen { .. })), "reopened"),
                (count(|o| matches!(o, Op::Delete { .. })), "deleted"),
            ];
            let parts: Vec<String> = counts
                .iter()
                .filter(|(n, _)| *n > 0)
                .enumerate()
                .map(|(i, (n, verb))| {
                    let noun = if i == 0 {
                        if *n == 1 { " task" } else { " tasks" }
                    } else {
                        ""
                    };
                    format!("{verb} {n}{noun}")
                })
                .collect();
            let sentence = match parts.as_slice() {
                [one] => one.clone(),
                [rest @ .., last] => format!("{} and {last}", rest.join(", ")),
                [] => String::new(),
            };
            let mut chars = sentence.chars();
            let first = chars
                .next()
                .map(|c| c.to_uppercase().collect::<String>())
                .unwrap_or_default();
            format!("{first}{}.", chars.as_str())
        }
    }
}

const ID_KEYS: &[&str] = &["id", "task_id", "taskid", "task", "ref", "number"];
const PRIORITY_KEYS: &[&str] = &[
    "priority",
    "prio",
    "importance",
    "urgency",
    "new_priority",
    "suggested_priority",
];
/// In a suggestion, a new priority wins over one that may be the current one
const SUGGESTED_PRIORITY_KEYS: &[&str] = &[
    "new_priority",
    "suggested_priority",
    "priority",
    "prio",
    "importance",
    "urgency",
];

/// "Today is Friday 2026-10-30, 09:00. The coming days: Sat 2026-10-31, ..."
fn today_line(now: DateTime<Local>) -> String {
    let today = now.date_naive();
    let days: Vec<String> = (1..=7)
        .filter_map(|n| today.checked_add_days(Days::new(n)))
        .map(|d| d.format("%a %Y-%m-%d").to_string())
        .collect();
    format!(
        "Today is {}. The coming days: {}.",
        now.format("%A %Y-%m-%d, %H:%M"),
        days.join(", ")
    )
}

/// The open tasks and those done in the last week, one compact line each
fn task_list(tasks: &[Task], now: DateTime<Local>) -> String {
    let today = now.date_naive();
    let mut out = String::from("Open tasks:\n");
    let open: Vec<&Task> = tasks.iter().filter(|t| t.is_open()).collect();
    if open.is_empty() {
        out.push_str("(none)\n");
    }
    for task in open {
        let mut line = format!(
            "#{} {} | {} | {} | {}",
            task.id.0,
            task.title,
            task.category,
            task.priority.label().to_lowercase(),
            status_word(task.status)
        );
        if let Some(due) = task.due {
            let days = (due - today).num_days();
            let when = match days {
                ..0 => format!("overdue {}d", -days),
                0 => "today".into(),
                1 => "tomorrow".into(),
                _ => due.format("%a").to_string(),
            };
            let _ = write!(line, " | due {due} ({when})");
        }
        if let Some(minutes) = task.estimate_minutes {
            let _ = write!(line, " | ~{minutes} min");
        }
        if !task.tags.is_empty() {
            let _ = write!(line, " | tags: {}", task.tags.join(", "));
        }
        let age = (now - task.created).num_days();
        let _ = write!(
            line,
            " | added {}",
            if age <= 0 {
                "today".into()
            } else {
                format!("{age}d ago")
            }
        );
        if !task.notes.is_empty() {
            let notes: String = task.notes.chars().take(80).collect();
            let _ = write!(line, " | notes: {}", notes.replace('\n', " "));
        }
        out.push_str(&line);
        out.push('\n');
    }

    let week_ago = now - chrono::Duration::days(7);
    let mut done: Vec<&Task> = tasks
        .iter()
        .filter(|t| !t.is_open() && t.completed.is_some_and(|c| c >= week_ago))
        .collect();
    done.sort_by_key(|t| std::cmp::Reverse(t.completed));
    if !done.is_empty() {
        out.push_str("Done in the last week:\n");
        for task in done.iter().take(10) {
            let when = task
                .completed
                .map(|c| c.date_naive().to_string())
                .unwrap_or_default();
            let _ = writeln!(
                out,
                "#{} {} | {} | done {when}",
                task.id.0, task.title, task.category
            );
        }
    }
    out
}

fn status_word(status: Status) -> &'static str {
    match status {
        Status::Todo => "todo",
        Status::InProgress => "in_progress",
        Status::Done => "done",
    }
}

/// One op of a reply, or `None` when it is malformed or names an unknown task
fn parse_op(obj: &Map<String, Value>, tasks: &[Task], today: NaiveDate) -> Option<Op> {
    let kind = text_field(
        obj,
        &["op", "action", "type", "kind", "operation", "command"],
    )
    .map(|k| key(&k));
    let id = field(obj, ID_KEYS).and_then(id_of).map(TaskId);
    let task = id.and_then(|id| tasks.iter().find(|t| t.id == id));
    let title = text_field(obj, &["title", "name", "task", "text", "summary"]);

    let kind = match kind.as_deref() {
        Some("add" | "create" | "new" | "insert" | "add_task" | "create_task" | "new_task") => {
            "add"
        }
        Some(
            "update" | "edit" | "modify" | "change" | "move" | "reschedule" | "set" | "update_task"
            | "edit_task" | "postpone" | "rename" | "prioritize",
        ) => "update",
        Some(
            "complete" | "done" | "finish" | "finished" | "completed" | "check" | "mark_done"
            | "complete_task" | "close",
        ) => "complete",
        Some(
            "reopen" | "undo" | "uncomplete" | "undone" | "restore" | "reopen_task" | "unfinish",
        ) => "reopen",
        Some(
            "delete" | "remove" | "cancel" | "drop" | "delete_task" | "remove_task" | "discard",
        ) => "delete",
        None if id.is_some() => "update",
        None if title.is_some() => "add",
        _ => return None,
    };

    let id = task.map(|t| t.id);
    match kind {
        "add" => {
            let title = title?.trim_end_matches('.').to_string();
            Some(Op::Add(NewTask {
                title,
                notes: text_field(obj, NOTES_KEYS).unwrap_or_default(),
                category: text_field(obj, CATEGORY_KEYS),
                priority: field(obj, PRIORITY_KEYS).and_then(priority_of),
                due: field(obj, DUE_KEYS)
                    .and_then(|v| date_of(v, today))
                    .flatten(),
                tags: field(obj, TAG_KEYS).and_then(tags_of).unwrap_or_default(),
                estimate_minutes: field(obj, ESTIMATE_KEYS).and_then(minutes_of),
            }))
        }
        "update" => {
            let patch = TaskPatch {
                title: text_field(obj, &["title", "name", "new_title"])
                    .map(|t| t.trim_end_matches('.').to_string()),
                notes: field(obj, NOTES_KEYS).map(|v| text(v).unwrap_or_default()),
                category: text_field(obj, CATEGORY_KEYS),
                priority: field(obj, PRIORITY_KEYS).and_then(priority_of),
                status: field(obj, &["status", "state"])
                    .and_then(Value::as_str)
                    .and_then(status_of),
                due: field(obj, DUE_KEYS).and_then(|v| date_of(v, today)),
                tags: field(obj, TAG_KEYS).and_then(tags_of),
                estimate_minutes: field(obj, ESTIMATE_KEYS).and_then(minutes_of),
            };
            (patch != TaskPatch::default()).then_some(Op::Update { id: id?, patch })
        }
        "complete" => task
            .filter(|t| t.is_open())
            .map(|t| Op::Complete { id: t.id }),
        "reopen" => task
            .filter(|t| !t.is_open())
            .map(|t| Op::Reopen { id: t.id }),
        _ => id.map(|id| Op::Delete { id }),
    }
}

const NOTES_KEYS: &[&str] = &[
    "notes",
    "note",
    "details",
    "description",
    "detail",
    "comment",
];
const CATEGORY_KEYS: &[&str] = &["category", "cat", "list", "area"];
const DUE_KEYS: &[&str] = &[
    "due", "due_date", "duedate", "date", "deadline", "when", "due_on", "by",
];
const TAG_KEYS: &[&str] = &["tags", "labels", "tag", "label"];
const ESTIMATE_KEYS: &[&str] = &[
    "estimate_minutes",
    "estimate",
    "minutes",
    "duration",
    "duration_minutes",
    "effort",
    "estimated_minutes",
];

/// A key or enum word made comparable: lowercase, '_' for spaces and dashes
fn key(word: &str) -> String {
    word.trim().to_lowercase().replace([' ', '-'], "_")
}

/// The value of the first of `keys` present in `obj`, keys compared with [`key`]
fn field<'a>(obj: &'a Map<String, Value>, keys: &[&str]) -> Option<&'a Value> {
    keys.iter()
        .find_map(|k| obj.iter().find(|(name, _)| key(name) == *k).map(|(_, v)| v))
}

/// A non-blank string (or a number written as one)
fn text(value: &Value) -> Option<String> {
    match value {
        Value::String(s) if !s.trim().is_empty() => Some(s.trim().to_string()),
        Value::Number(n) => Some(n.to_string()),
        _ => None,
    }
}

fn text_field(obj: &Map<String, Value>, keys: &[&str]) -> Option<String> {
    field(obj, keys).and_then(text)
}

/// A task id: `12`, `12.0`, `"12"`, `"#12"`, `"task 12"`
fn id_of(value: &Value) -> Option<u64> {
    match value {
        Value::Number(n) => n.as_u64().or_else(|| {
            n.as_f64()
                .filter(|f| f.fract() == 0.0 && *f >= 0.0)
                .map(|f| f as u64)
        }),
        Value::String(s) => {
            let s = s.trim();
            let (prefix, digits) = s.split_at(s.find(|c: char| c.is_ascii_digit())?);
            // "#12" or "task 12", not a title that happens to hold a number
            let prefix: String = prefix
                .chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>()
                .to_lowercase();
            let id_like = matches!(prefix.as_str(), "" | "task" | "id" | "taskid" | "no");
            if id_like { digits.parse().ok() } else { None }
        }
        _ => None,
    }
}

fn priority_of(value: &Value) -> Option<Priority> {
    value
        .as_str()
        .and_then(|s| Priority::parse(&s.replace(['-', '_'], " ").replace(" priority", "")))
}

fn status_of(word: &str) -> Option<Status> {
    Some(match key(word).as_str() {
        "todo" | "to_do" | "open" | "pending" | "not_started" | "new" | "reopened" => Status::Todo,
        "in_progress" | "inprogress" | "doing" | "started" | "active" | "wip" | "ongoing"
        | "progress" => Status::InProgress,
        "done" | "complete" | "completed" | "finished" | "closed" => Status::Done,
        _ => return None,
    })
}

/// A due date: `Some(None)` when the reply clears it (null, "", "none"),
/// `None` when it cannot be read
fn date_of(value: &Value, today: NaiveDate) -> Option<Option<NaiveDate>> {
    match value {
        Value::Null => Some(None),
        Value::String(s) => {
            let s = s.trim();
            if matches!(
                key(s).as_str(),
                "" | "none" | "null" | "no_date" | "n/a" | "never" | "no"
            ) {
                return Some(None);
            }
            if let Some(date) = s
                .get(..10)
                .and_then(|d| NaiveDate::parse_from_str(d, "%Y-%m-%d").ok())
            {
                return Some(Some(date));
            }
            heuristics::due_date(s, today).map(Some)
        }
        _ => None,
    }
}

fn tags_of(value: &Value) -> Option<Vec<String>> {
    match value {
        Value::Array(items) => Some(items.iter().filter_map(text).collect()),
        Value::String(s) => Some(
            s.split(',')
                .map(|t| t.trim().to_string())
                .filter(|t| !t.is_empty())
                .collect(),
        ),
        Value::Null => Some(Vec::new()),
        _ => None,
    }
}

/// Minutes from `30`, `"30"`, `"30 min"`, `"1.5h"`, `"2 hours"`, `"1h 30m"`,
/// `"2 days"` (of 8 hours of work) or `"1 week"` (5 such days); `None` for a
/// unit it does not know
fn minutes_of(value: &Value) -> Option<u32> {
    let minutes = match value {
        Value::Number(n) => n.as_f64()?,
        Value::String(s) => {
            let s = s.trim().to_lowercase();
            let mut rest = s
                .trim_start_matches('~')
                .trim_start_matches("about ")
                .trim();
            let mut total = 0.0;
            while !rest.is_empty() {
                let number_len = rest
                    .find(|c: char| !c.is_ascii_digit() && c != '.')
                    .unwrap_or(rest.len());
                let n: f64 = rest[..number_len].parse().ok()?;
                rest = rest[number_len..].trim_start();
                let unit_len = rest
                    .find(|c: char| !c.is_alphabetic())
                    .unwrap_or(rest.len());
                let scale = match &rest[..unit_len] {
                    "" | "m" | "min" | "mins" | "minute" | "minutes" => 1.0,
                    "h" | "hr" | "hrs" | "hour" | "hours" => 60.0,
                    "d" | "day" | "days" => 8.0 * 60.0,
                    "w" | "wk" | "week" | "weeks" => 5.0 * 8.0 * 60.0,
                    _ => return None,
                };
                total += n * scale;
                rest = rest[unit_len..]
                    .trim_start_matches([' ', ','])
                    .trim_start_matches("and ");
            }
            total
        }
        _ => return None,
    };
    (minutes > 0.0 && minutes < 100_000.0).then_some(minutes.round() as u32)
}

/// The JSON object (or array) in a reply: after any `<think>` block, past
/// code fences and prose. A trailing comma or a reply cut off by the token
/// limit is repaired where possible. An array of numbers counts only with
/// `bare_ids` (a review's order); otherwise it is prose ("[1] is first").
fn find_json(reply: &str, bare_ids: bool) -> Option<Value> {
    if let Some(end) = reply.rfind("</think>")
        && let Some(value) = find_json_in(&reply[end + "</think>".len()..], bare_ids)
    {
        return Some(value);
    }
    find_json_in(
        &reply.replace("<think>", "").replace("</think>", ""),
        bare_ids,
    )
}

fn find_json_in(text: &str, bare_ids: bool) -> Option<Value> {
    for (start, c) in text.char_indices() {
        if c != '{' && c != '[' {
            continue;
        }
        let candidate = &text[start..];
        let value = match balanced_len(candidate) {
            Some(len) => parse_lenient(&candidate[..len]),
            None => close_truncated(candidate),
        };
        if let Some(value @ (Value::Object(_) | Value::Array(_))) = value {
            // an array of words (or of numbers, but in a review) in prose is no answer
            if value.as_array().is_some_and(|a| {
                a.is_empty()
                    || a.iter()
                        .all(|v| !v.is_object() && !(bare_ids && v.is_number()))
            }) {
                continue;
            }
            return Some(value);
        }
    }
    None
}

/// The length of the bracketed value `text` starts with, or `None` when it
/// does not close
fn balanced_len(text: &str) -> Option<usize> {
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (i, c) in text.char_indices() {
        if in_string {
            match c {
                _ if escaped => escaped = false,
                '\\' => escaped = true,
                '"' => in_string = false,
                _ => {}
            }
            continue;
        }
        match c {
            '"' => in_string = true,
            '{' | '[' => depth += 1,
            '}' | ']' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(i + 1);
                }
            }
            _ => {}
        }
    }
    None
}

fn parse_lenient(json: &str) -> Option<Value> {
    if let Ok(value) = serde_json::from_str(json) {
        return Some(value);
    }
    let repaired = drop_trailing_commas(json);
    if let Ok(value) = serde_json::from_str(&repaired) {
        return Some(value);
    }
    serde_json::from_str(&repaired.replace(['\u{201c}', '\u{201d}'], "\"")).ok()
}

/// `json` without commas right before a closing bracket
fn drop_trailing_commas(json: &str) -> String {
    let mut out = String::with_capacity(json.len());
    let mut in_string = false;
    let mut escaped = false;
    for c in json.chars() {
        if in_string {
            match c {
                _ if escaped => escaped = false,
                '\\' => escaped = true,
                '"' => in_string = false,
                _ => {}
            }
        } else if c == '"' {
            in_string = true;
        } else if c == '}' || c == ']' {
            let kept = out.trim_end().len();
            if out[..kept].ends_with(',') {
                out.truncate(kept - 1);
            }
        }
        out.push(c);
    }
    out
}

/// A value cut off before its end, closed after its last complete member
fn close_truncated(text: &str) -> Option<Value> {
    let mut stack: Vec<char> = Vec::new();
    // where each member ended (at a comma) and what was open there
    let mut cuts: Vec<(usize, Vec<char>)> = Vec::new();
    let mut in_string = false;
    let mut escaped = false;
    for (i, c) in text.char_indices() {
        if in_string {
            match c {
                _ if escaped => escaped = false,
                '\\' => escaped = true,
                '"' => in_string = false,
                _ => {}
            }
            continue;
        }
        match c {
            '"' => in_string = true,
            '{' => stack.push('}'),
            '[' => stack.push(']'),
            '}' | ']' => {
                stack.pop();
                if stack.is_empty() {
                    return None;
                }
                cuts.push((i + 1, stack.clone()));
            }
            ',' => cuts.push((i, stack.clone())),
            _ => {}
        }
    }
    cuts.iter().rev().take(32).find_map(|(at, open)| {
        let mut json = text[..*at].to_string();
        json.extend(open.iter().rev());
        parse_lenient(&json)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    /// Friday 30 October 2026, 09:00
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

    fn d(m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(2026, m, day).unwrap()
    }

    fn task(id: u64, title: &str, status: Status) -> Task {
        Task {
            id: TaskId(id),
            title: title.into(),
            notes: String::new(),
            category: "Work".into(),
            priority: Priority::Medium,
            status,
            due: None,
            tags: Vec::new(),
            estimate_minutes: None,
            created: now() - chrono::Duration::days(2),
            completed: (status == Status::Done).then(now),
            source: String::new(),
        }
    }

    fn tasks() -> Vec<Task> {
        let mut report = task(1, "Write the report", Status::Todo);
        report.due = Some(d(10, 28));
        vec![
            report,
            task(2, "Dentist appointment", Status::Todo),
            task(3, "Buy milk", Status::Done),
        ]
    }

    #[test]
    fn prompts_carry_date_list_and_request() {
        let mut list = tasks();
        list.push(Task {
            category: "Garden".into(),
            ..task(4, "Prune roses", Status::InProgress)
        });
        let p = interpret_prompt("  move the dentist to friday ", &list, now());
        assert!(
            p.user.starts_with("Today is Friday 2026-10-30, 09:00. The coming days: Sat 2026-10-31, "),
            "{}",
            p.user
        );
        assert!(p.user.contains("Fri 2026-11-06.\nCategories: Work, Personal"), "{}", p.user);
        assert!(p.user.contains(", Garden\nOpen tasks:\n"), "{}", p.user);
        assert!(p.user.contains("#1 Write the report | Work | medium | todo | due 2026-10-28 (overdue 2d) | added 2d ago"), "{}", p.user);
        assert!(
            p.user
                .contains("#4 Prune roses | Garden | medium | in_progress")
        );
        assert!(
            p.user
                .contains("Done in the last week:\n#3 Buy milk | Work | done 2026-10-30")
        );
        assert!(p.user.ends_with("Request: move the dentist to friday"));
        assert!(p.system.contains("one of the categories listed in the message"));

        let r = review_prompt(&[], now());
        assert!(r.system.contains(&format!("up to {MAX_NEXT} open tasks")));
        assert!(r.user.starts_with("Today is Friday 2026-10-30, 09:00."), "{}", r.user);
        assert!(r.user.contains("Open tasks:\n(none)"));
        assert!(r.user.ends_with("Review the list."));
    }

    /// The system prompts are constant, so that the model's state after
    /// them can be saved once and reused for every request
    #[test]
    fn system_prompts_do_not_depend_on_the_list_or_the_time() {
        let mut list = tasks();
        list.push(Task {
            category: "Garden".into(),
            ..task(4, "Prune roses", Status::InProgress)
        });
        let later = now() + chrono::Duration::days(40) + chrono::Duration::hours(5);
        let a = interpret_prompt("buy milk", &[], now());
        let b = interpret_prompt("move the dentist to friday", &list, later);
        assert_eq!(a.system, INTERPRET_SYSTEM);
        assert_eq!(b.system, INTERPRET_SYSTEM);
        assert_ne!(a.user, b.user);
        let a = review_prompt(&[], now());
        let b = review_prompt(&list, later);
        assert_eq!(a.system, REVIEW_SYSTEM);
        assert_eq!(b.system, REVIEW_SYSTEM);
        for system in [INTERPRET_SYSTEM, REVIEW_SYSTEM] {
            assert!(!system.contains("2026") && !system.contains("Today is"), "{system}");
            assert!(!system.contains("Garden") && !system.contains("Work"), "{system}");
            for placeholder in ["{today}", "{categories}", "{MAX_NEXT}", "{{"] {
                assert!(!system.contains(placeholder), "{placeholder} in {system}");
            }
        }
    }

    #[test]
    fn parses_a_clean_reply() {
        let reply = r#"{"reply": "Done.", "ops": [
            {"op": "add", "title": "Call mum", "category": "Personal", "priority": "high", "due": "2026-10-31", "tags": ["family"], "estimate_minutes": 15},
            {"op": "complete", "id": 1},
            {"op": "update", "id": 2, "due": "2026-11-06"}
        ]}"#;
        let i = parse_interpretation(reply, &tasks(), now()).unwrap();
        assert_eq!(i.reply, "Done.");
        assert_eq!(
            i.ops,
            vec![
                Op::Add(NewTask {
                    title: "Call mum".into(),
                    notes: String::new(),
                    category: Some("Personal".into()),
                    priority: Some(Priority::High),
                    due: Some(d(10, 31)),
                    tags: vec!["family".into()],
                    estimate_minutes: Some(15),
                }),
                Op::Complete { id: TaskId(1) },
                Op::Update {
                    id: TaskId(2),
                    patch: TaskPatch {
                        due: Some(Some(d(11, 6))),
                        ..TaskPatch::default()
                    }
                },
            ]
        );
    }

    #[test]
    fn parses_messy_replies() {
        let reply = "<think>The user wants {milk}. Maybe {\"ops\": []}</think>\n\nSure! Here you go:\n```json\n\
            {\"Reply\": \"\", \"Operations\": [\n\
              {\"Action\": \"Create\", \"Title\": \"Pay rent\", \"Priority\": \"URGENT\", \"Due\": \"tomorrow\", \"estimate\": \"1.5h\"},\n\
              {\"op\": \"add\", \"title\": \"pay rent\"},\n\
              {\"op\": \"done\", \"id\": \"#2\"},\n\
              {\"op\": \"delete\", \"id\": 99},\n\
              {\"op\": \"complete\", \"id\": 3},\n\
              {\"op\": \"update\", \"id\": \"1\", \"due\": null, \"status\": \"In Progress\"},\n\
              {\"op\": \"launch\", \"id\": 1},\n\
            ],}\n```\nLet me know!";
        let i = parse_interpretation(reply, &tasks(), now()).unwrap();
        assert_eq!(
            i.ops,
            vec![
                Op::Add(NewTask {
                    title: "Pay rent".into(),
                    priority: Some(Priority::Urgent),
                    due: Some(d(10, 31)),
                    estimate_minutes: Some(90),
                    ..NewTask::default()
                }),
                Op::Complete { id: TaskId(2) },
                Op::Update {
                    id: TaskId(1),
                    patch: TaskPatch {
                        due: Some(None),
                        status: Some(Status::InProgress),
                        ..TaskPatch::default()
                    }
                },
            ]
        );
        assert_eq!(i.reply, "Added 1 task, updated 1 and completed 1.");
    }

    #[test]
    fn parses_bare_arrays_single_ops_and_truncated_replies() {
        let i = parse_interpretation(r#"[{"title": "Buy eggs"}]"#, &[], now()).unwrap();
        assert!(matches!(i.ops.as_slice(), [Op::Add(n)] if n.title == "Buy eggs"));
        assert_eq!(i.reply, "Added \"Buy eggs\".");

        let i = parse_interpretation(r#"{"op": "reopen", "id": 3}"#, &tasks(), now()).unwrap();
        assert_eq!(i.ops, vec![Op::Reopen { id: TaskId(3) }]);

        let cut = r#"{"reply": "Added two", "ops": [{"op": "add", "title": "A"}, {"op": "add", "title": "Bu"#;
        let i = parse_interpretation(cut, &[], now()).unwrap();
        assert!(
            matches!(i.ops.as_slice(), [Op::Add(n)] if n.title == "A"),
            "{:?}",
            i.ops
        );

        let i = parse_interpretation(
            r#"{"reply": "Which report do you mean?", "ops": []}"#,
            &tasks(),
            now(),
        )
        .unwrap();
        assert!(i.ops.is_empty());
        assert_eq!(i.reply, "Which report do you mean?");

        assert!(parse_interpretation("I don't know what you mean.", &tasks(), now()).is_err());
        assert!(parse_interpretation("<think>hmm</think>", &tasks(), now()).is_err());
    }

    #[test]
    fn a_numbered_aside_in_prose_does_not_hide_the_answer() {
        let reply = r#"Here: [1] is first. {"reply":"x","ops":[{"op":"complete","id":1}]}"#;
        let i = parse_interpretation(reply, &tasks(), now()).unwrap();
        assert_eq!(i.ops, vec![Op::Complete { id: TaskId(1) }]);
        assert_eq!(i.reply, "x");
        // a review may still answer with bare ids
        let plan = parse_review("[3, 1]", &tasks(), now()).unwrap();
        assert_eq!(plan.next[0].id, TaskId(1));
    }

    #[test]
    fn estimates_take_their_unit() {
        let minutes = |s: &str| minutes_of(&Value::from(s));
        assert_eq!(minutes("45"), Some(45));
        assert_eq!(minutes("30 min"), Some(30));
        assert_eq!(minutes("1.5h"), Some(90));
        assert_eq!(minutes("2 hours"), Some(120));
        assert_eq!(minutes("1h 30m"), Some(90));
        assert_eq!(minutes("1 hour and 15 minutes"), Some(75));
        assert_eq!(minutes("about 20 mins"), Some(20));
        assert_eq!(minutes("2 days"), Some(960));
        assert_eq!(minutes("1 week"), Some(2400));
        assert_eq!(minutes("3 fortnights"), None);
        assert_eq!(minutes("soon"), None);
        assert_eq!(minutes_of(&Value::from(25)), Some(25));
    }

    #[test]
    fn ids_are_read_strictly_from_strings() {
        assert_eq!(id_of(&Value::from("#12")), Some(12));
        assert_eq!(id_of(&Value::from("task 7")), Some(7));
        assert_eq!(id_of(&Value::from(4.0)), Some(4));
        assert_eq!(id_of(&Value::from("Buy 2 apples")), None);
    }

    #[test]
    fn parses_a_review() {
        let mut list = tasks();
        list.push(task(4, "Prune roses", Status::Todo));
        let reply = r#"Here's my review:
        {"summary": "One task is overdue.",
         "next": [{"id": 1, "reason": "It is overdue."}, {"id": 3, "reason": "done already"}, {"id": 42}, 4, {"id": 1}],
         "suggestions": [{"id": 1, "priority": "High", "reason": "Overdue"}, {"id": 2, "priority": "medium"}, {"id": 3, "priority": "low"}, {"id": 4, "priority": "whatever"}]}"#;
        let plan = parse_review(reply, &list, now()).unwrap();
        assert!(plan.from_model);
        assert_eq!(plan.generated, now());
        assert_eq!(plan.summary, "One task is overdue.");
        let next: Vec<(u64, &str)> = plan
            .next
            .iter()
            .map(|i| (i.id.0, i.reason.as_str()))
            .collect();
        assert_eq!(next, [(1, "It is overdue."), (4, "Medium priority")]);
        assert_eq!(
            plan.suggestions,
            vec![PrioritySuggestion {
                id: TaskId(1),
                priority: Priority::High,
                reason: "Overdue".into()
            }]
        );
    }

    #[test]
    fn a_suggestion_with_the_current_and_new_priority_takes_the_new() {
        let reply = r#"{"next":[2],"suggestions":[{"id":2,"priority":"medium","suggested_priority":"high"}]}"#;
        let plan = parse_review(reply, &tasks(), now()).unwrap();
        let suggested: Vec<(TaskId, Priority)> = plan
            .suggestions
            .iter()
            .map(|s| (s.id, s.priority))
            .collect();
        assert_eq!(suggested, [(TaskId(2), Priority::High)]);
    }

    #[test]
    fn an_empty_review_falls_back_to_the_ranking() {
        let plan = parse_review(r#"```{"next": [{"id": 3}]}```"#, &tasks(), now()).unwrap();
        let ids: Vec<u64> = plan.next.iter().map(|i| i.id.0).collect();
        assert_eq!(ids, [1, 2]);
        assert_eq!(plan.summary, "2 open tasks, 1 overdue");
        assert!(plan.from_model);
        assert!(parse_review("no idea", &tasks(), now()).is_err());
    }

    #[test]
    fn describes_ops() {
        let today = now().date_naive();
        let add = Op::Add(NewTask {
            title: "Call mum".into(),
            due: Some(d(10, 31)),
            priority: Some(Priority::High),
            ..NewTask::default()
        });
        assert_eq!(
            describe_ops(&[add], &[], today),
            "Added \"Call mum\" (due tomorrow, high)."
        );
        let mv = Op::Update {
            id: TaskId(2),
            patch: TaskPatch {
                due: Some(Some(d(11, 6))),
                ..TaskPatch::default()
            },
        };
        assert_eq!(
            describe_ops(&[mv], &tasks(), today),
            "Moved \"Dentist appointment\" to Fri 6 Nov."
        );
        assert_eq!(
            describe_ops(
                &[Op::Delete { id: TaskId(1) }, Op::Delete { id: TaskId(2) }],
                &tasks(),
                today
            ),
            "Deleted 2 tasks."
        );
        assert_eq!(describe_ops(&[], &[], today), "Nothing to change.");
    }
}
