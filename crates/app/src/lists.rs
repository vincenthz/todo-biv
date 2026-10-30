//! What the views show, without any UI: which tasks a smart view or a
//! category holds and in which order, how a due date reads ("tomorrow",
//! "3 days overdue"), and the plan to display.

use std::collections::{BTreeSet, HashSet};

use chrono::{DateTime, Datelike as _, Local, NaiveDate};
use todo_core::{DEFAULT_CATEGORIES, Plan, PlanItem, Task, TaskId, plan};

/// A selection in the sidebar
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum View {
    /// Open tasks in the order of the plan
    NextUp,
    /// Open tasks due today or overdue
    Today,
    /// Open tasks due after today
    Upcoming,
    AllOpen,
    Done,
    /// Open tasks of one category
    Category(String),
}

impl View {
    pub const SMART: [View; 5] = [View::NextUp, View::Today, View::Upcoming, View::AllOpen, View::Done];

    pub fn label(&self) -> &str {
        match self {
            View::NextUp => "Next up",
            View::Today => "Today",
            View::Upcoming => "Upcoming",
            View::AllOpen => "All open",
            View::Done => "Done",
            View::Category(name) => name,
        }
    }

    /// Whether `task` belongs to the view (ignoring the plan's order)
    pub fn contains(&self, task: &Task, today: NaiveDate) -> bool {
        match self {
            View::Done => !task.is_open(),
            _ if !task.is_open() => false,
            View::NextUp | View::AllOpen => true,
            View::Today => task.due.is_some_and(|due| due <= today),
            View::Upcoming => task.due.is_some_and(|due| due > today),
            View::Category(name) => task.category.eq_ignore_ascii_case(name),
        }
    }
}

/// The tasks of `view`, in display order: the plan's order for "Next up",
/// due date for "Upcoming", most recently completed first for "Done", and
/// [`plan::score`] otherwise.
pub fn tasks_in<'a>(view: &View, tasks: &'a [Task], plan: &Plan, now: DateTime<Local>) -> Vec<&'a Task> {
    let today = now.date_naive();
    let mut shown: Vec<&Task> = tasks.iter().filter(|t| view.contains(t, today)).collect();
    match view {
        View::NextUp => {
            let position = |id: TaskId| plan.next.iter().position(|item| item.id == id).unwrap_or(usize::MAX);
            shown.sort_by_key(|t| position(t.id));
            shown.retain(|t| position(t.id) != usize::MAX);
        }
        View::Upcoming => shown.sort_by(|a, b| a.due.cmp(&b.due).then_with(|| by_score(a, b, now))),
        View::Done => shown.sort_by(|a, b| b.completed.cmp(&a.completed).then_with(|| b.id.cmp(&a.id))),
        _ => shown.sort_by(|a, b| by_score(a, b, now)),
    }
    shown
}

fn by_score(a: &Task, b: &Task, now: DateTime<Local>) -> std::cmp::Ordering {
    plan::score(b, now).total_cmp(&plan::score(a, now)).then_with(|| a.id.cmp(&b.id))
}

/// How many tasks `view` holds
pub fn count(view: &View, tasks: &[Task], plan: &Plan, now: DateTime<Local>) -> usize {
    match view {
        View::NextUp => tasks_in(view, tasks, plan, now).len(),
        _ => tasks.iter().filter(|t| view.contains(t, now.date_naive())).count(),
    }
}

/// Categories with open tasks, in the default order and then alphabetically,
/// with how many open tasks each has
pub fn categories(tasks: &[Task]) -> Vec<(String, usize)> {
    let mut counts: Vec<(String, usize)> = Vec::new();
    for task in tasks.iter().filter(|t| t.is_open()) {
        match counts.iter_mut().find(|(name, _)| name.eq_ignore_ascii_case(&task.category)) {
            Some((_, n)) => *n += 1,
            None => counts.push((task.category.clone(), 1)),
        }
    }
    counts.sort_by_key(|(name, _)| category_rank(name));
    counts
}

fn category_rank(name: &str) -> (usize, String) {
    let default = DEFAULT_CATEGORIES.iter().position(|c| c.eq_ignore_ascii_case(name)).unwrap_or(usize::MAX);
    (default, name.to_lowercase())
}

/// Every category to offer when editing a task: the defaults, then those in use
pub fn category_choices(tasks: &[Task]) -> Vec<String> {
    let mut seen = BTreeSet::new();
    let mut choices: Vec<String> = DEFAULT_CATEGORIES.iter().map(|c| c.to_string()).collect();
    for c in &choices {
        seen.insert(c.to_lowercase());
    }
    let mut extra: Vec<String> =
        tasks.iter().map(|t| t.category.clone()).filter(|c| !c.is_empty() && seen.insert(c.to_lowercase())).collect();
    extra.sort_by_key(|c| c.to_lowercase());
    choices.extend(extra);
    choices
}

/// How pressing a due date is, for its color
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Due {
    Overdue,
    Today,
    Soon,
    Later,
}

/// A due date as it reads next to a task: "today", "tomorrow", "Fri",
/// "3 days overdue", "Nov 12"
pub fn due_label(due: NaiveDate, today: NaiveDate) -> (String, Due) {
    let days = (due - today).num_days();
    match days {
        ..=-2 => (format!("{} days overdue", -days), Due::Overdue),
        -1 => ("yesterday".into(), Due::Overdue),
        0 => ("today".into(), Due::Today),
        1 => ("tomorrow".into(), Due::Soon),
        2..=6 => (due.format("%a").to_string(), Due::Soon),
        _ if due.year() == today.year() => (due.format("%b %-d").to_string(), Due::Later),
        _ => (due.format("%b %-d, %Y").to_string(), Due::Later),
    }
}

/// How long ago `then` was, briefly: "just now", "5 min ago", "2 h ago",
/// "3 days ago"
pub fn ago(then: DateTime<Local>, now: DateTime<Local>) -> String {
    let minutes = (now - then).num_minutes();
    match minutes {
        ..=0 => "just now".into(),
        1..=59 => format!("{minutes} min ago"),
        60..=1439 => format!("{} h ago", minutes / 60),
        _ => match minutes / 1440 {
            1 => "yesterday".into(),
            days => format!("{days} days ago"),
        },
    }
}

/// The plan to show: the model's last review with the tasks completed or
/// deleted since left out and those added since appended in the rules'
/// order, or the rules' ranking when the model has not reviewed the list.
pub fn effective_plan(stored: Option<&Plan>, tasks: &[Task], now: DateTime<Local>) -> Plan {
    let fallback = plan::fallback_plan(tasks, now);
    let Some(stored) = stored.filter(|p| p.from_model) else { return fallback };
    let open: HashSet<TaskId> = tasks.iter().filter(|t| t.is_open()).map(|t| t.id).collect();
    let mut seen = HashSet::new();
    let mut next: Vec<PlanItem> =
        stored.next.iter().filter(|item| open.contains(&item.id) && seen.insert(item.id)).cloned().collect();
    if next.is_empty() {
        return fallback;
    }
    next.extend(fallback.next.into_iter().filter(|item| seen.insert(item.id)));
    let suggestions = stored
        .suggestions
        .iter()
        .filter(|s| tasks.iter().any(|t| t.id == s.id && t.is_open() && t.priority != s.priority))
        .cloned()
        .collect();
    Plan { generated: stored.generated, summary: stored.summary.clone(), next, suggestions, from_model: true }
}

#[cfg(test)]
mod tests {
    use chrono::{Duration, TimeZone as _};
    use todo_core::{Priority, PrioritySuggestion, Status};

    use super::*;

    fn now() -> DateTime<Local> {
        Local.with_ymd_and_hms(2026, 10, 30, 9, 0, 0).single().expect("a valid local time")
    }

    fn task(id: u64, title: &str, category: &str, priority: Priority, due: Option<i64>) -> Task {
        Task {
            id: TaskId(id),
            title: title.into(),
            notes: String::new(),
            category: category.into(),
            priority,
            status: Status::Todo,
            due: due.map(|days| now().date_naive() + Duration::days(days)),
            tags: Vec::new(),
            estimate_minutes: None,
            created: now() - Duration::days(id as i64),
            completed: None,
            source: String::new(),
        }
    }

    fn done(mut task: Task, hours_ago: i64) -> Task {
        task.status = Status::Done;
        task.completed = Some(now() - Duration::hours(hours_ago));
        task
    }

    fn sample() -> Vec<Task> {
        vec![
            task(1, "File taxes", "Finance", Priority::High, Some(-3)),
            task(2, "Call mum", "Personal", Priority::Medium, Some(0)),
            task(3, "Dentist", "Health", Priority::Medium, Some(1)),
            task(4, "Slides", "Work", Priority::Urgent, Some(4)),
            task(5, "Read book", "Learning", Priority::Low, None),
            done(task(6, "Buy milk", "Errands", Priority::Low, None), 5),
            done(task(7, "Report", "Work", Priority::High, None), 1),
            task(8, "Fix shelf", "home", Priority::Low, Some(30)),
        ]
    }

    fn plan_of(ids: &[u64]) -> Plan {
        Plan {
            generated: now() - Duration::minutes(5),
            summary: "Busy week".into(),
            next: ids.iter().map(|&id| PlanItem { id: TaskId(id), reason: format!("because {id}") }).collect(),
            suggestions: Vec::new(),
            from_model: true,
        }
    }

    fn ids(tasks: &[&Task]) -> Vec<u64> {
        tasks.iter().map(|t| t.id.0).collect()
    }

    #[test]
    fn views_filter_tasks() {
        let tasks = sample();
        let today = now().date_naive();
        let of = |view: View| tasks.iter().filter(|t| view.contains(t, today)).map(|t| t.id.0).collect::<Vec<_>>();
        assert_eq!(of(View::Today), [1, 2]);
        assert_eq!(of(View::Upcoming), [3, 4, 8]);
        assert_eq!(of(View::AllOpen), [1, 2, 3, 4, 5, 8]);
        assert_eq!(of(View::Done), [6, 7]);
        assert_eq!(of(View::Category("Work".into())), [4]);
        assert_eq!(of(View::Category("Home".into())), [8], "categories match whatever their case");
    }

    #[test]
    fn next_up_follows_the_plan() {
        let tasks = sample();
        let plan = plan_of(&[4, 1, 7, 2]);
        assert_eq!(ids(&tasks_in(&View::NextUp, &tasks, &plan, now())), [4, 1, 2], "done tasks are left out");
        assert_eq!(count(&View::NextUp, &tasks, &plan, now()), 3);
    }

    #[test]
    fn upcoming_is_by_date_and_done_most_recent_first() {
        let tasks = sample();
        let plan = plan_of(&[]);
        assert_eq!(ids(&tasks_in(&View::Upcoming, &tasks, &plan, now())), [3, 4, 8]);
        assert_eq!(ids(&tasks_in(&View::Done, &tasks, &plan, now())), [7, 6]);
    }

    #[test]
    fn categories_count_open_tasks() {
        let tasks = sample();
        assert_eq!(
            categories(&tasks),
            [
                ("Work".to_string(), 1),
                ("Personal".to_string(), 1),
                ("home".to_string(), 1),
                ("Health".to_string(), 1),
                ("Finance".to_string(), 1),
                ("Learning".to_string(), 1),
            ]
        );
        let mut extra = sample();
        extra.push(task(9, "Water plants", "Garden", Priority::Low, None));
        let choices = category_choices(&extra);
        assert_eq!(choices.len(), DEFAULT_CATEGORIES.len() + 1);
        assert_eq!(choices.last().map(String::as_str), Some("Garden"));
    }

    #[test]
    fn due_dates_read_relatively() {
        let today = now().date_naive(); // a Friday
        let label = |days: i64| due_label(today + Duration::days(days), today);
        assert_eq!(label(0), ("today".into(), Due::Today));
        assert_eq!(label(1), ("tomorrow".into(), Due::Soon));
        assert_eq!(label(-1), ("yesterday".into(), Due::Overdue));
        assert_eq!(label(-3), ("3 days overdue".into(), Due::Overdue));
        assert_eq!(label(3), ("Mon".into(), Due::Soon));
        assert_eq!(label(12), ("Nov 11".into(), Due::Later));
        assert_eq!(label(80), ("Jan 18, 2027".into(), Due::Later));
    }

    #[test]
    fn ages_read_briefly() {
        assert_eq!(ago(now(), now()), "just now");
        assert_eq!(ago(now() - Duration::minutes(5), now()), "5 min ago");
        assert_eq!(ago(now() - Duration::hours(3), now()), "3 h ago");
        assert_eq!(ago(now() - Duration::days(1), now()), "yesterday");
        assert_eq!(ago(now() - Duration::days(4), now()), "4 days ago");
    }

    #[test]
    fn a_stale_model_plan_is_filtered_and_completed() {
        let tasks = sample();
        let mut plan = plan_of(&[7, 4, 99, 1, 4]);
        plan.suggestions = vec![
            PrioritySuggestion { id: TaskId(2), priority: Priority::High, reason: "due today".into() },
            PrioritySuggestion { id: TaskId(4), priority: Priority::Urgent, reason: "already urgent".into() },
            PrioritySuggestion { id: TaskId(6), priority: Priority::High, reason: "done".into() },
        ];
        let shown = effective_plan(Some(&plan), &tasks, now());
        assert!(shown.from_model);
        assert_eq!(&shown.next.iter().map(|i| i.id.0).collect::<Vec<_>>()[..2], [4, 1]);
        let mut all: Vec<u64> = shown.next.iter().map(|i| i.id.0).collect();
        all.sort();
        assert_eq!(all, [1, 2, 3, 4, 5, 8], "every open task once, none done or unknown");
        assert_eq!(shown.suggestions.iter().map(|s| s.id.0).collect::<Vec<_>>(), [2]);
        assert_eq!(shown.next[0].reason, "because 4");
    }

    #[test]
    fn without_a_model_plan_the_rules_rank() {
        let tasks = sample();
        let shown = effective_plan(None, &tasks, now());
        assert!(!shown.from_model);
        let mut rules = plan_of(&[]);
        rules.from_model = false;
        assert!(!effective_plan(Some(&rules), &tasks, now()).from_model);
        assert!(!effective_plan(Some(&plan_of(&[6, 7])), &tasks, now()).from_model, "nothing left of the model's plan");
    }
}
