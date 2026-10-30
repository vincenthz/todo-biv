//! What should be done next: the plan the model writes when it reviews the
//! list, and the deterministic ranking used before (or without) it.

use chrono::{DateTime, Local};
use serde::{Deserialize, Serialize};

use crate::task::{Priority, Status, Task, TaskId, relative_day};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlanItem {
    pub id: TaskId,
    /// Why it comes at this point, in a sentence
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrioritySuggestion {
    pub id: TaskId,
    pub priority: Priority,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Plan {
    pub generated: DateTime<Local>,
    /// A short overview of where the list stands
    pub summary: String,
    /// Open tasks in the order to take them, the first one being "next"
    pub next: Vec<PlanItem>,
    /// Priorities the review thinks are wrong; shown to be accepted or dismissed
    #[serde(default)]
    pub suggestions: Vec<PrioritySuggestion>,
    /// Whether the model wrote it (false: the deterministic ranking)
    #[serde(default)]
    pub from_model: bool,
}

/// How many tasks a plan lists
pub const MAX_NEXT: usize = 7;

/// How pressing an open task is at `now`: priority, then due date (overdue and
/// due-soon tasks rise), then age as a tie-breaker. Higher comes first.
pub fn score(task: &Task, now: DateTime<Local>) -> f64 {
    let priority = match task.priority {
        Priority::Low => 100.0,
        Priority::Medium => 200.0,
        Priority::High => 300.0,
        Priority::Urgent => 400.0,
    };
    // an overdue task climbs a priority level, one due today nearly does
    let due = match task.due.map(|d| (d - now.date_naive()).num_days()) {
        None => 0.0,
        Some(days) if days < 0 => 110.0 + (5 * (-days).min(8)) as f64,
        Some(0) => 75.0,
        Some(1) => 50.0,
        Some(days @ 2..=3) => 40.0 - days as f64,
        Some(days @ 4..=7) => 25.0 - days as f64,
        Some(days) => (10.0 - days as f64 / 7.0).max(1.0),
    };
    let started = if task.status == Status::InProgress {
        15.0
    } else {
        0.0
    };
    let age = (now - task.created).num_hours().max(0) as f64 / 24.0;
    priority + due + started + age.min(30.0) * 0.2
}

/// Open tasks ranked by [`score`], with a reason for each
pub fn fallback_plan(tasks: &[Task], now: DateTime<Local>) -> Plan {
    let mut open: Vec<(&Task, f64)> = tasks
        .iter()
        .filter(|t| t.is_open())
        .map(|t| (t, score(t, now)))
        .collect();
    open.sort_by(|(a, sa), (b, sb)| sb.total_cmp(sa).then(a.id.cmp(&b.id)));
    let next = open
        .iter()
        .take(MAX_NEXT)
        .map(|(t, _)| PlanItem {
            id: t.id,
            reason: reason(t, now),
        })
        .collect();
    Plan {
        generated: now,
        summary: summary(tasks, now),
        next,
        suggestions: Vec::new(),
        from_model: false,
    }
}

/// Why `task` ranks where it does: "Urgent, due tomorrow", "Overdue by 2 days"
pub fn reason(task: &Task, now: DateTime<Local>) -> String {
    let today = now.date_naive();
    let mut parts: Vec<String> = Vec::new();
    match task.priority {
        Priority::Urgent => parts.push("urgent".into()),
        Priority::High => parts.push("high priority".into()),
        _ => {}
    }
    if let Some(due) = task.due {
        let days = (due - today).num_days();
        parts.push(if days < 0 {
            format!("overdue by {}", plural(-days, "day"))
        } else {
            format!("due {}", relative_day(due, today))
        });
    }
    if task.status == Status::InProgress {
        parts.push("already started".into());
    }
    if parts.is_empty() {
        let age = (now - task.created).num_days();
        parts.push(if age >= 7 {
            format!("waiting for {}", plural(age, "day"))
        } else {
            format!("{} priority", task.priority.label().to_lowercase())
        });
    }
    let reason = parts.join(", ");
    let mut chars = reason.chars();
    chars
        .next()
        .map(|c| c.to_uppercase().chain(chars).collect())
        .unwrap_or_default()
}

/// "5 open tasks, 1 overdue, 2 due today"
pub fn summary(tasks: &[Task], now: DateTime<Local>) -> String {
    let today = now.date_naive();
    let open: Vec<&Task> = tasks.iter().filter(|t| t.is_open()).collect();
    if open.is_empty() {
        return "Nothing open, all clear.".into();
    }
    let overdue = open
        .iter()
        .filter(|t| t.due.is_some_and(|d| d < today))
        .count();
    let due_today = open.iter().filter(|t| t.due == Some(today)).count();
    let urgent = open
        .iter()
        .filter(|t| t.priority == Priority::Urgent)
        .count();
    let tasks = if open.len() == 1 { "task" } else { "tasks" };
    let mut parts = vec![format!("{} open {tasks}", open.len())];
    if overdue > 0 {
        parts.push(format!("{overdue} overdue"));
    }
    if due_today > 0 {
        parts.push(format!("{due_today} due today"));
    }
    if urgent > 0 {
        parts.push(format!("{urgent} urgent"));
    }
    parts.join(", ")
}

/// "1 day", "3 days"
fn plural(n: i64, noun: &str) -> String {
    if n == 1 {
        format!("1 {noun}")
    } else {
        format!("{n} {noun}s")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, NaiveDate, TimeZone};

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

    fn task(id: u64, priority: Priority, due_in: Option<i64>, age_days: i64) -> Task {
        Task {
            id: TaskId(id),
            title: format!("task {id}"),
            notes: String::new(),
            category: "Work".into(),
            priority,
            status: Status::Todo,
            due: due_in.map(|d| now().date_naive() + Duration::days(d)),
            tags: Vec::new(),
            estimate_minutes: None,
            created: now() - Duration::days(age_days),
            completed: None,
            source: String::new(),
        }
    }

    #[test]
    fn priority_dominates_then_due_then_age() {
        let n = now();
        let s = |t: &Task| score(t, n);
        assert!(s(&task(1, Priority::Urgent, None, 0)) > s(&task(2, Priority::High, Some(0), 30)));
        assert!(
            s(&task(1, Priority::High, Some(10), 0)) > s(&task(2, Priority::Medium, Some(2), 0))
        );
        assert!(s(&task(1, Priority::High, Some(0), 0)) > s(&task(2, Priority::High, Some(1), 0)));
        assert!(s(&task(1, Priority::High, Some(1), 0)) > s(&task(2, Priority::High, Some(5), 0)));
        assert!(s(&task(1, Priority::High, Some(5), 0)) > s(&task(2, Priority::High, None, 0)));
        assert!(s(&task(1, Priority::Medium, None, 10)) > s(&task(2, Priority::Medium, None, 1)));
        // overdue climbs a level
        assert!(s(&task(1, Priority::Medium, Some(-1), 0)) > s(&task(2, Priority::High, None, 0)));
        assert!(
            s(&task(1, Priority::Medium, Some(-3), 0)) > s(&task(2, Priority::Medium, Some(-1), 0))
        );
        let mut started = task(1, Priority::Medium, None, 0);
        started.status = Status::InProgress;
        assert!(s(&started) > s(&task(2, Priority::Medium, None, 0)));
    }

    #[test]
    fn fallback_orders_open_tasks_with_reasons() {
        let mut done = task(9, Priority::Urgent, None, 0);
        done.status = Status::Done;
        let tasks = vec![
            task(1, Priority::Low, None, 0),
            task(2, Priority::Medium, Some(-2), 0),
            task(3, Priority::Urgent, Some(1), 0),
            task(4, Priority::Medium, None, 10),
            done,
        ];
        let plan = fallback_plan(&tasks, now());
        let ids: Vec<u64> = plan.next.iter().map(|i| i.id.0).collect();
        assert_eq!(ids, [3, 2, 4, 1]);
        assert_eq!(plan.next[0].reason, "Urgent, due tomorrow");
        assert_eq!(plan.next[1].reason, "Overdue by 2 days");
        assert_eq!(plan.next[2].reason, "Waiting for 10 days");
        assert_eq!(plan.next[3].reason, "Low priority");
        assert_eq!(plan.summary, "4 open tasks, 1 overdue, 1 urgent");
        assert!(!plan.from_model);
        assert_eq!(
            fallback_plan(&[], now()).summary,
            "Nothing open, all clear."
        );
    }

    #[test]
    fn fallback_lists_at_most_max_next() {
        let tasks: Vec<Task> = (1..=12)
            .map(|i| task(i, Priority::Medium, None, 0))
            .collect();
        assert_eq!(fallback_plan(&tasks, now()).next.len(), MAX_NEXT);
    }
}
