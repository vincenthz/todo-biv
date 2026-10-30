//! A rule-based reading of requests, used while the model is loading or when
//! it is unavailable, and to fill in what the model left out: a category from
//! keywords, a priority from words like "urgent" or "whenever", and due dates
//! from "today", "tomorrow", "friday", "next week", "in 3 days", "on the 12th".

use std::ops::Range;

use chrono::{DateTime, Datelike, Days, Local, Months, NaiveDate, Weekday};

use crate::prompt::{Interpretation, describe_ops};
use crate::task::{NewTask, Op, Priority, Task, TaskPatch};

/// The best guess at what `request` asks, without the model. Recognizes
/// adding (the default), "done/finished/completed <task>" and "delete/remove
/// <task>" matched against `tasks` by title words.
pub fn interpret(request: &str, tasks: &[Task], now: DateTime<Local>) -> Interpretation {
    let today = now.date_naive();
    let request = request.trim();
    let lower = normalize(request);
    if lower.is_empty() {
        return Interpretation {
            ops: Vec::new(),
            reply: "There was nothing to add.".into(),
        };
    }

    if let Some(rest) = completion_target(&lower) {
        let open: Vec<Task> = tasks.iter().filter(|t| t.is_open()).cloned().collect();
        match best_match(rest, &open, false) {
            Some(task) => return done(today, vec![Op::Complete { id: task.id }], tasks),
            // "complete the tax form" is as likely a task to add
            None if !lower.starts_with("complete ") => return not_found(rest),
            None => {}
        }
    }

    if let Some((rest, whole_title)) = strip_any_prefix(&lower, DELETE_PREFIXES)
        .map(|rest| (rest, false))
        .or_else(|| strip_any_prefix(&lower, WEAK_DELETE_PREFIXES).map(|rest| (rest, true)))
        && let Some(task) = best_match(rest, tasks, true)
        && (!whole_title || names_whole_title(rest, task))
    {
        return done(today, vec![Op::Delete { id: task.id }], tasks);
    }

    if let Some(rest) = strip_any_prefix(&lower, MOVE_PREFIXES) {
        let words = words(rest);
        let norms: Vec<&str> = words.iter().map(|w| w.norm.as_str()).collect();
        if let Some((span, due)) = find_dates(&norms, today).into_iter().next() {
            let what = norms[..span.start].join(" ");
            if let Some(task) = best_match(&what, tasks, false) {
                let patch = TaskPatch {
                    due: Some(Some(due)),
                    ..TaskPatch::default()
                };
                return done(today, vec![Op::Update { id: task.id, patch }], tasks);
            }
        }
    }

    let (title, notes) = title_and_notes(request, today);
    if title.is_empty() {
        return Interpretation {
            ops: Vec::new(),
            reply: "I couldn't tell what to add.".into(),
        };
    }
    let new = NewTask {
        title,
        notes,
        category: category(request).map(str::to_string),
        priority: priority(request),
        due: due_date(request, today),
        ..NewTask::default()
    };
    done(today, vec![Op::Add(new)], tasks)
}

pub fn category(text: &str) -> Option<&'static str> {
    let words: Vec<String> = words(&text.replace('-', " "))
        .into_iter()
        .map(|w| stem(&w.norm))
        .collect();
    let words: Vec<&str> = words.iter().map(String::as_str).collect();
    let mut best: Option<(&'static str, usize)> = None;
    for (category, keywords) in CATEGORY_KEYWORDS {
        let hits = keywords
            .iter()
            .filter(|k| {
                has_phrase(
                    &words,
                    &k.split(' ').map(stem).collect::<Vec<_>>().join(" "),
                )
            })
            .count();
        if hits > 0 && best.is_none_or(|(_, n)| hits > n) {
            best = Some((category, hits));
        }
    }
    best.map(|(c, _)| c)
}

pub fn priority(text: &str) -> Option<Priority> {
    let words: Vec<String> = words(&text.replace('-', " "))
        .into_iter()
        .map(|w| w.norm)
        .collect();
    let words: Vec<&str> = words.iter().map(String::as_str).collect();
    let any = |phrases: &[&str]| phrases.iter().any(|p| has_phrase(&words, p));
    // negations first: "not urgent" is no reason to hurry
    if any(&[
        "not urgent",
        "no rush",
        "not important",
        "no hurry",
        "not a priority",
        "low priority",
    ]) {
        return Some(Priority::Low);
    }
    if any(URGENT_WORDS) {
        Some(Priority::Urgent)
    } else if any(HIGH_WORDS) {
        Some(Priority::High)
    } else if any(LOW_WORDS) {
        Some(Priority::Low)
    } else {
        None
    }
}

/// A due date named in `text`, relative to `today`
pub fn due_date(text: &str, today: NaiveDate) -> Option<NaiveDate> {
    let words = words(text);
    let norms: Vec<&str> = words.iter().map(|w| w.norm.as_str()).collect();
    find_dates(&norms, today)
        .into_iter()
        .next()
        .map(|(_, date)| date)
}

const URGENT_WORDS: &[&str] = &[
    "urgent",
    "urgently",
    "asap",
    "a.s.a.p",
    "critical",
    "immediately",
    "right away",
    "right now",
    "emergency",
    "as soon as possible",
    "top priority",
];
const HIGH_WORDS: &[&str] = &[
    "important",
    "high priority",
    "must",
    "crucial",
    "priority",
    "really need",
    "essential",
];
const LOW_WORDS: &[&str] = &[
    "someday",
    "some day",
    "sometime",
    "whenever",
    "maybe",
    "eventually",
    "if i have time",
    "when i have time",
    "if possible",
    "one day",
];

/// Keywords per default category (singular: words are matched after a plain
/// "s" is dropped)
const CATEGORY_KEYWORDS: [(&str, &[&str]); 8] = [
    (
        "Work",
        &[
            "work",
            "meeting",
            "report",
            "slide",
            "presentation",
            "deck",
            "client",
            "customer",
            "boss",
            "manager",
            "colleague",
            "project",
            "deadline",
            "office",
            "standup",
            "sprint",
            "deploy",
            "release",
            "code",
            "bug",
            "pr",
            "proposal",
            "spec",
            "review",
            "interview",
            "team",
            "quarterly",
            "roadmap",
        ],
    ),
    (
        "Personal",
        &[
            "mum", "mom", "dad", "mother", "father", "family", "sister", "brother", "grandma",
            "grandpa", "journal", "haircut", "passport", "visa", "vacation", "holiday", "trip",
            "travel", "pack", "letter", "kid", "wife", "husband",
        ],
    ),
    (
        "Home",
        &[
            "clean",
            "laundry",
            "dish",
            "vacuum",
            "hoover",
            "garden",
            "repair",
            "fix",
            "plumber",
            "electrician",
            "kitchen",
            "bathroom",
            "house",
            "flat",
            "apartment",
            "furniture",
            "trash",
            "bin",
            "rubbish",
            "mow",
            "lawn",
            "tidy",
            "landlord",
            "fridge",
            "boiler",
            "lightbulb",
            "cook",
            "plant",
        ],
    ),
    (
        "Health",
        &[
            "doctor",
            "gp",
            "dentist",
            "gym",
            "workout",
            "jog",
            "running",
            "exercise",
            "medicine",
            "pill",
            "prescription",
            "pharmacy",
            "hospital",
            "therapy",
            "therapist",
            "yoga",
            "vitamin",
            "checkup",
            "check up",
            "vaccine",
            "physio",
            "meditate",
            "meditation",
            "blood test",
            "eye test",
            "optician",
            "swim",
        ],
    ),
    (
        "Finance",
        &[
            "bank",
            "pay",
            "bill",
            "tax",
            "invoice",
            "budget",
            "rent",
            "mortgage",
            "insurance",
            "transfer",
            "salary",
            "refund",
            "accountant",
            "credit card",
            "loan",
            "saving",
            "invest",
            "expense",
            "pension",
            "subscription",
        ],
    ),
    (
        "Errands",
        &[
            "buy",
            "errand",
            "pick up",
            "pickup",
            "grocery",
            "groceries",
            "shopping",
            "shop",
            "store",
            "post office",
            "parcel",
            "package",
            "milk",
            "bread",
            "egg",
            "supermarket",
            "return",
            "collect",
            "drop off",
            "car wash",
            "fuel",
            "petrol",
            "dry cleaning",
            "order",
        ],
    ),
    (
        "Learning",
        &[
            "learn",
            "study",
            "read",
            "course",
            "class",
            "lesson",
            "practice",
            "practise",
            "tutorial",
            "homework",
            "exam",
            "lecture",
            "research",
            "article",
            "paper",
            "chapter",
            "language",
            "vocabulary",
        ],
    ),
    (
        "Social",
        &[
            "party",
            "friend",
            "dinner",
            "drink",
            "wedding",
            "birthday",
            "invite",
            "catch up",
            "coffee",
            "visit",
            "hang out",
            "bbq",
            "barbecue",
            "brunch",
            "meetup",
            "date night",
            "gift",
            "present",
        ],
    ),
];

/// Openings dropped from a request before it becomes a title, longest first
/// where one is the start of another
const FILLER_PREFIXES: &[&str] = &[
    "please",
    "can you",
    "could you",
    "would you",
    "hey",
    "ok",
    "okay",
    "so",
    "remind me to",
    "remind me",
    "don't forget to",
    "dont forget to",
    "do not forget to",
    "remember to",
    "i need to",
    "i have to",
    "i must",
    "i should",
    "i want to",
    "i'd like to",
    "i would like to",
    "i gotta",
    "i've got to",
    "need to",
    "have to",
    "got to",
    "gotta",
    "should",
    "must",
    "let me",
    "add a task to",
    "add a task",
    "add task",
    "add a todo to",
    "add a todo",
    "add todo",
    "add",
    "new task",
    "create a task to",
    "create a task",
    "todo",
    "to do",
    "task",
    "to",
];

const COMPLETE_PREFIXES: &[&str] = &[
    "i have finished",
    "i've finished",
    "i finished",
    "finished",
    "i have completed",
    "i've completed",
    "i completed",
    "completed",
    "complete",
    "i'm done with",
    "i am done with",
    "done with",
    "done",
    "i have done",
    "i've done",
    "i did",
    "tick off",
    "check off",
    "cross off",
    "mark off",
];

const COMPLETE_SUFFIXES: &[&str] = &[
    "is done",
    "is finished",
    "is complete",
    "are done",
    "are finished",
    "has been done",
    "was done",
    "is sorted",
];

const MARK_SUFFIXES: &[&str] = &[
    "as done",
    "as complete",
    "as completed",
    "as finished",
    "done",
    "complete",
    "completed",
    "finished",
];

/// Verbs that delete a task the rest of the request names. Everyday verbs
/// like "drop" (the kids at school) or "scratch" are left out: a task to add
/// would be taken for a task to delete.
const DELETE_PREFIXES: &[&str] = &[
    "delete",
    "remove",
    "forget about",
    "forget",
    "never mind",
    "nevermind",
    "get rid of",
];

/// Verbs that delete a task only when the request names its whole title:
/// "cancel the gym class" may well be a task to add next to "Book gym class"
const WEAK_DELETE_PREFIXES: &[&str] = &["cancel"];

const MOVE_PREFIXES: &[&str] = &[
    "move",
    "reschedule",
    "postpone",
    "push back",
    "push",
    "change the date of",
    "change",
];

/// Words that carry no meaning when matching a request to a title
const STOPWORDS: &[&str] = &[
    "the", "a", "an", "my", "to", "with", "for", "of", "on", "at", "in", "i", "it", "and", "task",
    "todo", "that", "this", "about", "one", "thing", "is", "as", "be", "me", "up", "please", "now",
];

/// Words of a clause that say nothing beyond its date and priority
const FLUFF: &[&str] = &[
    "it's", "its", "it", "is", "this", "that", "very", "really", "super", "quite", "pretty", "so",
    "please", "and", "but", "be", "must", "should", "needs", "need", "to", "done", "a", "bit",
    "kind", "of", "thanks", "thank", "you", "by", "on", "before", "due", "for", "until", "till",
    "at", "the", "latest", "or", "ok", "okay", "also",
];

/// Prepositions that go with a date ("by friday", "due on the 12th")
const DATE_PREPOSITIONS: &[&str] = &[
    "by", "on", "before", "until", "till", "due", "for", "at", "from", "the",
];

fn done(today: NaiveDate, ops: Vec<Op>, tasks: &[Task]) -> Interpretation {
    let reply = describe_ops(&ops, tasks, today);
    Interpretation { ops, reply }
}

fn not_found(query: &str) -> Interpretation {
    Interpretation {
        ops: Vec::new(),
        reply: format!("I couldn't find a task matching \"{}\".", query.trim()),
    }
}

/// Lowercase, single-spaced, typographic apostrophes made plain
fn normalize(text: &str) -> String {
    text.replace(['\u{2019}', '\u{2018}'], "'")
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// A word of a request: as written (for titles) and normalized for matching
struct Word<'a> {
    raw: &'a str,
    norm: String,
}

fn words(text: &str) -> Vec<Word<'_>> {
    text.split_whitespace()
        .map(|raw| {
            let norm = raw
                .trim_matches(|c: char| !c.is_alphanumeric())
                .replace(['\u{2019}', '\u{2018}'], "'")
                .to_lowercase();
            Word { raw, norm }
        })
        .collect()
}

/// `word` with a plural "s" dropped ("bills" -> "bill"), leaving short words
/// and "ss" endings alone
fn stem(word: &str) -> String {
    if word.len() > 3 && word.ends_with('s') && !word.ends_with("ss") {
        word[..word.len() - 1].to_string()
    } else {
        word.to_string()
    }
}

/// Whether the space-separated `phrase` occurs in `words`, word for word
fn has_phrase(words: &[&str], phrase: &str) -> bool {
    phrase_at_any(words, phrase).is_some()
}

fn phrase_at_any(words: &[&str], phrase: &str) -> Option<Range<usize>> {
    let parts: Vec<&str> = phrase.split(' ').collect();
    (0..words.len().saturating_sub(parts.len() - 1))
        .find(|&i| words[i..].starts_with(&parts))
        .map(|i| i..i + parts.len())
}

/// `text` after one of `prefixes` followed by a word boundary
fn strip_any_prefix<'a>(text: &'a str, prefixes: &[&str]) -> Option<&'a str> {
    prefixes.iter().find_map(|p| {
        let rest = text.strip_prefix(p)?;
        match rest.chars().next() {
            None => Some(rest),
            Some(c) if !c.is_alphanumeric() && c != '\'' => {
                Some(rest.trim_start_matches(|c: char| c.is_whitespace() || c == ':' || c == ','))
            }
            _ => None,
        }
    })
}

/// The task part of "I finished the report", "mark the report as done",
/// "the report is done"
fn completion_target(lower: &str) -> Option<&str> {
    let lower = lower.trim_end_matches(['.', '!']);
    if let Some(rest) = strip_any_prefix(lower, &["mark"]) {
        for suffix in MARK_SUFFIXES {
            if let Some(what) = rest.strip_suffix(suffix)
                && what.ends_with(' ')
            {
                return Some(what.trim());
            }
        }
    }
    for suffix in COMPLETE_SUFFIXES {
        if let Some(what) = lower.strip_suffix(suffix)
            && what.ends_with(' ')
        {
            return Some(what.trim());
        }
    }
    strip_any_prefix(lower, COMPLETE_PREFIXES).filter(|rest| !rest.is_empty())
}

/// The task `query` names, by "#id" or by the words it shares with titles.
/// With `all_words` every meaningful word of the query must be in the title
/// (deleting on a loose match would lose a task); otherwise half of them.
fn best_match<'a>(query: &str, tasks: &'a [Task], all_words: bool) -> Option<&'a Task> {
    let query_words = words(query);
    for w in &query_words {
        if let Some(id) = w.raw.strip_prefix('#').and_then(|n| {
            n.trim_end_matches(|c: char| !c.is_ascii_digit())
                .parse::<u64>()
                .ok()
        }) && let Some(task) = tasks.iter().find(|t| t.id.0 == id)
        {
            return Some(task);
        }
    }
    let wanted = significant(query);
    if wanted.is_empty() {
        return None;
    }
    let mut best: Option<(&Task, usize, f64)> = None;
    for task in tasks {
        let title = significant(&task.title);
        let shared = wanted.iter().filter(|w| title.contains(w)).count();
        let enough = if all_words {
            shared == wanted.len()
        } else {
            shared * 2 >= wanted.len()
        };
        if shared == 0 || !enough {
            continue;
        }
        let coverage = shared as f64 / title.len().max(1) as f64;
        let better = match best {
            None => true,
            Some((b, s, c)) => {
                (shared, coverage, task.is_open(), task.id) > (s, c, b.is_open(), b.id)
            }
        };
        if better {
            best = Some((task, shared, coverage));
        }
    }
    best.map(|(t, _, _)| t)
}

/// The words of `text` that matter when matching it to a title, stemmed
fn significant(text: &str) -> Vec<String> {
    words(text)
        .into_iter()
        .map(|w| stem(&w.norm))
        .filter(|w| !w.is_empty() && !STOPWORDS.contains(&w.as_str()))
        .collect()
}

/// Whether `query` names `task` by its id or by every meaningful word of its title
fn names_whole_title(query: &str, task: &Task) -> bool {
    let by_id = words(query)
        .iter()
        .any(|w| w.raw.strip_prefix('#') == Some(task.id.0.to_string().as_str()));
    let wanted = significant(query);
    by_id || significant(&task.title).iter().all(|w| wanted.contains(w))
}

/// The title and notes of a task from a request to add it: filler openings,
/// dates and priority words are dropped, the first clause is the title and
/// the clauses after it that say more than a date or a priority are notes
fn title_and_notes(request: &str, today: NaiveDate) -> (String, String) {
    let mut clauses = request
        .split([',', ';', '!', '?', '\n'])
        .flat_map(|c| c.split(". "))
        .map(str::trim)
        .filter(|c| !c.is_empty());
    let mut title = String::new();
    for clause in clauses.by_ref() {
        title = clean_clause(clause, today);
        if !title.is_empty() {
            break;
        }
    }
    let notes: Vec<String> = clauses
        .map(|c| clean_clause(c, today))
        .filter(|c| {
            let words = words(c);
            words
                .iter()
                .any(|w| !w.norm.is_empty() && !FLUFF.contains(&w.norm.as_str()))
        })
        .collect();
    (capitalize(&title), capitalize(&notes.join("; ")))
}

/// `clause` without filler openings, dates (with their prepositions) and
/// priority words, and without dangling connectives at either end
fn clean_clause(clause: &str, today: NaiveDate) -> String {
    let mut words = words(clause);
    while let Some(len) = FILLER_PREFIXES.iter().find_map(|p| {
        let parts: Vec<&str> = p.split(' ').collect();
        let fits =
            words.len() >= parts.len() && words.iter().zip(&parts).all(|(w, p)| w.norm == *p);
        fits.then_some(parts.len())
    }) {
        words.drain(..len);
    }

    let norms: Vec<&str> = words.iter().map(|w| w.norm.as_str()).collect();
    let mut drop = vec![false; words.len()];
    let mark = |range: Range<usize>, drop: &mut Vec<bool>| {
        let mut start = range.start;
        while start > 0 && DATE_PREPOSITIONS.contains(&norms[start - 1]) && !drop[start - 1] {
            start -= 1;
        }
        drop[start..range.end].iter_mut().for_each(|d| *d = true);
    };
    for (range, _) in find_dates(&norms, today) {
        mark(range, &mut drop);
    }
    for phrase in URGENT_WORDS
        .iter()
        .chain(HIGH_WORDS)
        .chain(LOW_WORDS)
        .chain(&["not urgent", "no rush", "no hurry"])
    {
        if matches!(*phrase, "must" | "priority" | "really need") {
            continue;
        }
        let mut from = 0;
        while let Some(r) = phrase_at_any(&norms[from..], phrase) {
            let range = from + r.start..from + r.end;
            from = range.end;
            drop[range].iter_mut().for_each(|d| *d = true);
        }
    }

    let mut kept: Vec<&Word> = words
        .iter()
        .zip(&drop)
        .filter(|(_, d)| !**d)
        .map(|(w, _)| w)
        .collect();
    let dangling = [
        "and", "or", "it's", "its", "it", "is", "please", "by", "on", "to", "the", "at", "due",
        "for", "very", "really", "but", "so", "also",
    ];
    while kept
        .last()
        .is_some_and(|w| w.norm.is_empty() || dangling.contains(&w.norm.as_str()))
    {
        kept.pop();
    }
    while kept.first().is_some_and(|w| {
        w.norm.is_empty() || ["and", "or", "but", "so", "also", "please"].contains(&w.norm.as_str())
    }) {
        kept.remove(0);
    }
    let joined = kept.iter().map(|w| w.raw).collect::<Vec<_>>().join(" ");
    joined
        .trim_end_matches(['.', ',', ';', ':', '!', '?', '-'])
        .trim()
        .to_string()
}

fn capitalize(text: &str) -> String {
    let mut chars = text.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().chain(chars).collect(),
        None => String::new(),
    }
}

/// The dates named in `words` (normalized), left to right: the words each
/// spans and the day it means
fn find_dates(words: &[&str], today: NaiveDate) -> Vec<(Range<usize>, NaiveDate)> {
    let mut found = Vec::new();
    let mut i = 0;
    while i < words.len() {
        match date_at(words, i, today) {
            Some((len, date)) => {
                found.push((i..i + len, date));
                i += len;
            }
            None => i += 1,
        }
    }
    found
}

/// A date starting at `words[i]`: how many words it takes and the day
fn date_at(words: &[&str], i: usize, today: NaiveDate) -> Option<(usize, NaiveDate)> {
    let at = |k: usize| words.get(i + k).copied().unwrap_or("");
    let plus = |n: u64| today.checked_add_days(Days::new(n));
    let word = at(0);

    if let Ok(date) = NaiveDate::parse_from_str(word, "%Y-%m-%d") {
        return Some((1, date));
    }
    match word {
        "today" | "tonight" | "eod" => return Some((1, today)),
        "tomorrow" | "tmrw" | "tmr" | "tomorow" | "tommorow" | "tommorrow" => {
            return Some((1, plus(1)?));
        }
        "day" if at(1) == "after" && matches!(at(2), "tomorrow" | "tmrw") => {
            return Some((3, plus(2)?));
        }
        "weekend" => return Some((1, weekend(today))),
        "this" => {
            return match at(1) {
                "morning" | "afternoon" | "evening" => Some((2, today)),
                "weekend" => Some((2, weekend(today))),
                "week" => Some((2, coming(today, Weekday::Fri))),
                "month" => Some((2, end_of_month(today))),
                w => weekday_and_date(words, i + 1, today)
                    .map(|(n, d)| (n + 1, d))
                    .or_else(|| weekday(w, true).map(|d| (2, coming(today, d)))),
            };
        }
        "next" => {
            return match at(1) {
                "week" => Some((2, coming_after(today, Weekday::Mon))),
                "weekend" => Some((2, weekend(today).checked_add_days(Days::new(7))?)),
                "month" => Some((2, end_of_month(today).succ_opt()?)),
                "year" => Some((2, NaiveDate::from_ymd_opt(today.year() + 1, 1, 1)?)),
                w => weekday_and_date(words, i + 1, today)
                    .map(|(n, d)| (n + 1, d))
                    .or_else(|| Some((2, next_weekday(today, weekday(w, true)?)?))),
            };
        }
        "end" if at(1) == "of" => {
            let skip = if at(2) == "the" { 3 } else { 2 };
            let date = match at(skip) {
                "day" => today,
                "week" => coming(today, Weekday::Fri),
                "month" => end_of_month(today),
                "year" => NaiveDate::from_ymd_opt(today.year(), 12, 31)?,
                _ => return None,
            };
            return Some((skip + 1, date));
        }
        "in" | "within" => {
            let (n, unit_at) = if at(1) == "a" && at(2) == "couple" && at(3) == "of" {
                (2, 4)
            } else {
                (count(at(1))?, 2)
            };
            let date = match at(unit_at) {
                "day" | "days" => plus(n)?,
                "week" | "weeks" => plus(7 * n)?,
                "fortnight" => plus(14 * n)?,
                "month" | "months" => {
                    today.checked_add_months(Months::new(u32::try_from(n).ok()?))?
                }
                _ => return None,
            };
            return Some((unit_at + 1, date));
        }
        "the" if ordinal(at(1)).is_some() => {
            return day_month_at(words, i + 1, today).map(|(n, d)| (n + 1, d));
        }
        _ => {}
    }

    if let Some(found) = day_month_at(words, i, today) {
        return Some(found);
    }
    if let Some(found) = weekday_and_date(words, i, today) {
        return Some(found);
    }
    // "sat", "sun" and "wed" are words too: they count only after "on", "by"...
    let cued = i > 0
        && matches!(
            words[i - 1],
            "on" | "by" | "until" | "till" | "before" | "due" | "for"
        );
    weekday(word, cued).map(|d| (1, coming(today, d)))
}

/// A weekday followed by the date it names, which is what counts: "friday 6
/// november", "fri, nov 13", "monday the 9th"
fn weekday_and_date(words: &[&str], i: usize, today: NaiveDate) -> Option<(usize, NaiveDate)> {
    weekday(words.get(i)?, true)?;
    let skip = if words.get(i + 1) == Some(&"the") {
        2
    } else {
        1
    };
    day_month_at(words, i + skip, today).map(|(n, date)| (n + skip, date))
}

/// "12th", "12th of march", "march 12", "12 march", "march 12th 2027"
fn day_month_at(words: &[&str], i: usize, today: NaiveDate) -> Option<(usize, NaiveDate)> {
    let at = |k: usize| words.get(i + k).copied().unwrap_or("");
    let year_at = |k: usize| {
        at(k)
            .parse::<i32>()
            .ok()
            .filter(|y| (2000..=2100).contains(y))
    };
    let in_year = |month: u32, day: u32, len: usize| -> Option<(usize, NaiveDate)> {
        if let Some(year) = year_at(len) {
            return Some((len + 1, NaiveDate::from_ymd_opt(year, month, day)?));
        }
        // the next one, up to four years ahead for February 29
        (today.year()..=today.year() + 4)
            .filter_map(|year| NaiveDate::from_ymd_opt(year, month, day))
            .find(|date| *date >= today)
            .map(|date| (len, date))
    };

    if let Some(month) = month(at(0)) {
        let day = ordinal(at(1)).or_else(|| at(1).parse().ok().filter(|d| (1..=31).contains(d)))?;
        return in_year(month, day, 2);
    }
    let day = ordinal(at(0)).or_else(|| at(0).parse().ok().filter(|d| (1..=31).contains(d)))?;
    if let Some(month) = month(at(1)) {
        return in_year(month, day, 2);
    }
    if at(1) == "of"
        && let Some(month) = month(at(2))
    {
        return in_year(month, day, 3);
    }
    // a bare number is not a date; an ordinal is a day of this or next month
    ordinal(at(0))?;
    let mut first = today.with_day(1)?;
    for _ in 0..12 {
        if let Some(date) = first.with_day(day)
            && date >= today
        {
            return Some((1, date));
        }
        first = first.checked_add_months(Months::new(1))?;
    }
    None
}

/// "12th", "1st", "3rd" as a day of the month
fn ordinal(word: &str) -> Option<u32> {
    let digits = word
        .strip_suffix("st")
        .or_else(|| word.strip_suffix("nd"))
        .or_else(|| word.strip_suffix("rd"))
        .or_else(|| word.strip_suffix("th"))?;
    digits.parse().ok().filter(|d| (1..=31).contains(d))
}

fn count(word: &str) -> Option<u64> {
    if let Ok(n) = word.parse::<u64>() {
        return (n <= 3650).then_some(n);
    }
    let n = [
        "a", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve",
    ]
    .iter()
    .position(|w| *w == word)?;
    Some(if n == 0 { 1 } else { n as u64 })
}

fn month(word: &str) -> Option<u32> {
    let n = match word {
        "january" | "jan" => 1,
        "february" | "feb" => 2,
        "march" | "mar" => 3,
        "april" | "apr" => 4,
        "may" => 5,
        "june" | "jun" => 6,
        "july" | "jul" => 7,
        "august" | "aug" => 8,
        "september" | "sep" | "sept" => 9,
        "october" | "oct" => 10,
        "november" | "nov" => 11,
        "december" | "dec" => 12,
        _ => return None,
    };
    Some(n)
}

/// A weekday name; the abbreviations that are also words only when `cued`
fn weekday(word: &str, cued: bool) -> Option<Weekday> {
    let day = match word {
        "monday" | "mon" => Weekday::Mon,
        "tuesday" | "tue" | "tues" => Weekday::Tue,
        "wednesday" => Weekday::Wed,
        "thursday" | "thu" | "thur" | "thurs" => Weekday::Thu,
        "friday" | "fri" => Weekday::Fri,
        "saturday" => Weekday::Sat,
        "sunday" => Weekday::Sun,
        "wed" if cued => Weekday::Wed,
        "sat" if cued => Weekday::Sat,
        "sun" if cued => Weekday::Sun,
        _ => return None,
    };
    Some(day)
}

/// The next `day`, today included
fn coming(today: NaiveDate, day: Weekday) -> NaiveDate {
    let ahead = (7 + day.num_days_from_monday() - today.weekday().num_days_from_monday()) % 7;
    today + Days::new(u64::from(ahead))
}

/// "next friday": the friday of next week, unless this week's is already
/// past, then the coming one (a Saturday's "next friday" is six days ahead)
fn next_weekday(today: NaiveDate, day: Weekday) -> Option<NaiveDate> {
    let date = coming(today, day);
    let this_week = date.iso_week() == today.iso_week();
    if this_week {
        date.checked_add_days(Days::new(7))
    } else {
        Some(date)
    }
}

/// The next `day` after today
fn coming_after(today: NaiveDate, day: Weekday) -> NaiveDate {
    coming(today + Days::new(1), day)
}

/// The Saturday of this weekend (today when it is the weekend)
fn weekend(today: NaiveDate) -> NaiveDate {
    if today.weekday() == Weekday::Sun {
        today
    } else {
        coming(today, Weekday::Sat)
    }
}

fn end_of_month(today: NaiveDate) -> NaiveDate {
    let first = today.with_day(1).unwrap_or(today);
    first
        .checked_add_months(Months::new(1))
        .and_then(|d| d.pred_opt())
        .unwrap_or(today)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::task::{Status, TaskId};
    use chrono::TimeZone;

    /// Friday 30 October 2026
    fn today() -> NaiveDate {
        NaiveDate::from_ymd_opt(2026, 10, 30).unwrap()
    }

    fn now() -> DateTime<Local> {
        Local
            .from_local_datetime(&today().and_hms_opt(9, 0, 0).unwrap())
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
            created: now(),
            completed: None,
            source: String::new(),
        }
    }

    #[test]
    fn due_dates() {
        let t = today();
        let cases: &[(&str, Option<NaiveDate>)] = &[
            ("call mum today", Some(t)),
            ("tonight: pack", Some(t)),
            ("call mum tomorrow", Some(d(10, 31))),
            ("the day after tomorrow", Some(d(11, 1))),
            ("by friday", Some(d(10, 30))),
            ("this friday", Some(d(10, 30))),
            ("next friday", Some(d(11, 6))),
            ("monday", Some(d(11, 2))),
            ("on Wednesday.", Some(d(11, 4))),
            ("on wed", Some(d(11, 4))),
            ("we wed", None),
            ("buy sun cream", None),
            ("this weekend", Some(d(10, 31))),
            ("next week", Some(d(11, 2))),
            ("in 3 days", Some(d(11, 2))),
            ("in two weeks", Some(d(11, 13))),
            ("in a week", Some(d(11, 6))),
            ("in a couple of days", Some(d(11, 1))),
            ("in 1 month", Some(d(11, 30))),
            ("end of month", Some(d(10, 31))),
            ("by the end of the month", Some(d(10, 31))),
            ("on the 12th", Some(d(11, 12))),
            ("30th", Some(d(10, 30))),
            ("the 31st", Some(d(10, 31))),
            (
                "March 3",
                Some(NaiveDate::from_ymd_opt(2027, 3, 3).unwrap()),
            ),
            (
                "3 march",
                Some(NaiveDate::from_ymd_opt(2027, 3, 3).unwrap()),
            ),
            (
                "3rd of March",
                Some(NaiveDate::from_ymd_opt(2027, 3, 3).unwrap()),
            ),
            ("dec 24th", Some(d(12, 24))),
            (
                "march 3 2028",
                Some(NaiveDate::from_ymd_opt(2028, 3, 3).unwrap()),
            ),
            ("due 2026-11-15.", Some(d(11, 15))),
            ("buy 2 apples", None),
            ("finish the slides", None),
            ("i may go", None),
        ];
        for (text, want) in cases {
            assert_eq!(due_date(text, t), *want, "{text}");
        }
    }

    #[test]
    fn a_weekday_before_a_date_gives_way_to_it() {
        let t = today();
        let cases: &[(&str, Option<NaiveDate>)] = &[
            ("Friday 6 November", Some(d(11, 6))),
            ("Fri, Nov 13", Some(d(11, 13))),
            ("Monday 9 Nov", Some(d(11, 9))),
            ("sat 7 nov", Some(d(11, 7))),
            ("dentist on Friday 13th November", Some(d(11, 13))),
            ("monday the 9th", Some(d(11, 9))),
            ("next friday 13 november", Some(d(11, 13))),
            ("monday 2 apples", Some(d(11, 2))),
        ];
        for (text, want) in cases {
            assert_eq!(due_date(text, t), *want, "{text}");
        }
        let i = interpret("dentist on Friday 13th November", &[], now());
        assert!(
            matches!(i.ops.as_slice(), [Op::Add(n)] if n.title == "Dentist" && n.due == Some(d(11, 13))),
            "{:?}",
            i.ops
        );
    }

    #[test]
    fn next_weekday_is_in_next_week_unless_this_weeks_is_past() {
        let saturday = d(10, 31);
        let sunday = d(11, 1);
        let monday = d(10, 26);
        assert_eq!(due_date("next friday", saturday), Some(d(11, 6)));
        assert_eq!(due_date("next monday", sunday), Some(d(11, 2)));
        assert_eq!(due_date("next friday", monday), Some(d(11, 6)));
        assert_eq!(due_date("next monday", monday), Some(d(11, 2)));
        assert_eq!(due_date("next sunday", monday), Some(d(11, 8)));
    }

    #[test]
    fn february_29_is_found_in_the_next_leap_year() {
        let leap = NaiveDate::from_ymd_opt(2028, 2, 29);
        assert_eq!(due_date("feb 29", today()), leap);
        assert_eq!(due_date("29th of february", today()), leap);
        assert_eq!(due_date("feb 30", today()), None);
    }

    #[test]
    fn ordinal_rolls_to_next_month_and_skips_short_months() {
        assert_eq!(due_date("on the 5th", today()), Some(d(11, 5)));
        let jan31 = NaiveDate::from_ymd_opt(2027, 1, 31).unwrap();
        assert_eq!(
            due_date("the 30th", jan31),
            NaiveDate::from_ymd_opt(2027, 3, 30)
        );
    }

    #[test]
    fn priorities() {
        assert_eq!(priority("call the bank ASAP"), Some(Priority::Urgent));
        assert_eq!(priority("it's critical"), Some(Priority::Urgent));
        assert_eq!(priority("it's important"), Some(Priority::High));
        assert_eq!(priority("high-priority fix"), Some(Priority::High));
        assert_eq!(priority("whenever"), Some(Priority::Low));
        assert_eq!(priority("maybe learn piano"), Some(Priority::Low));
        assert_eq!(priority("not urgent at all"), Some(Priority::Low));
        assert_eq!(priority("buy milk"), None);
    }

    #[test]
    fn categories() {
        assert_eq!(category("buy milk"), Some("Errands"));
        assert_eq!(category("call the bank"), Some("Finance"));
        assert_eq!(category("finish the slides"), Some("Work"));
        assert_eq!(category("book the dentist"), Some("Health"));
        assert_eq!(category("pay the bills"), Some("Finance"));
        assert_eq!(category("call mum"), Some("Personal"));
        assert_eq!(category("clean the kitchen"), Some("Home"));
        assert_eq!(category("dinner with friends"), Some("Social"));
        assert_eq!(category("study for the exam"), Some("Learning"));
        assert_eq!(category("xyzzy"), None);
    }

    #[test]
    fn adds_with_title_date_priority() {
        let i = interpret("Remind me to call mum tomorrow, it's important", &[], now());
        let [Op::Add(new)] = i.ops.as_slice() else {
            panic!("{:?}", i.ops)
        };
        assert_eq!(new.title, "Call mum");
        assert_eq!(new.notes, "");
        assert_eq!(new.due, Some(d(10, 31)));
        assert_eq!(new.priority, Some(Priority::High));
        assert_eq!(new.category.as_deref(), Some("Personal"));
        assert!(i.reply.contains("Call mum"), "{}", i.reply);
    }

    #[test]
    fn add_strips_filler_and_keeps_notes() {
        let i = interpret(
            "todo: finish the slides by next friday, the deck for Acme",
            &[],
            now(),
        );
        let [Op::Add(new)] = i.ops.as_slice() else {
            panic!("{:?}", i.ops)
        };
        assert_eq!(new.title, "Finish the slides");
        assert_eq!(new.notes, "The deck for Acme");
        assert_eq!(new.due, Some(d(11, 6)));

        let i = interpret("I need to pay rent urgently", &[], now());
        let [Op::Add(new)] = i.ops.as_slice() else {
            panic!("{:?}", i.ops)
        };
        assert_eq!(new.title, "Pay rent");
        assert_eq!(new.priority, Some(Priority::Urgent));
    }

    #[test]
    fn completes_by_title_words() {
        let tasks = [
            task(1, "Write quarterly report", Status::Todo),
            task(2, "Buy milk", Status::Todo),
        ];
        for request in [
            "I finished the report",
            "done: report",
            "mark the report as done",
            "The report is done.",
        ] {
            let i = interpret(request, &tasks, now());
            assert_eq!(i.ops, vec![Op::Complete { id: TaskId(1) }], "{request}");
        }
        let i = interpret("finished the taxes", &tasks, now());
        assert!(i.ops.is_empty());
        assert!(i.reply.contains("couldn't find"));
    }

    #[test]
    fn deletes_only_on_a_full_match_else_adds() {
        let tasks = [
            task(1, "Clean bathroom", Status::Todo),
            task(2, "Gym membership renewal", Status::Todo),
        ];
        let i = interpret("delete the bathroom one", &tasks, now());
        assert_eq!(i.ops, vec![Op::Delete { id: TaskId(1) }]);
        let i = interpret("remove #2", &tasks, now());
        assert_eq!(i.ops, vec![Op::Delete { id: TaskId(2) }]);
        let i = interpret("remove the mould in the bathroom", &tasks, now());
        assert!(
            matches!(i.ops.as_slice(), [Op::Add(n)] if n.title == "Remove the mould in the bathroom"),
            "{:?}",
            i.ops
        );
    }

    #[test]
    fn everyday_verbs_add_rather_than_delete() {
        let tasks = [
            task(1, "Pick up the kids from school", Status::Todo),
            task(3, "Book gym class", Status::Todo),
        ];
        for request in [
            "Drop the kids at school",
            "scratch the kids from school",
            "cancel the gym class",
        ] {
            let i = interpret(request, &tasks, now());
            assert!(
                matches!(i.ops.as_slice(), [Op::Add(_)]),
                "{request}: {:?}",
                i.ops
            );
        }
        let i = interpret("cancel book gym class", &tasks, now());
        assert_eq!(i.ops, vec![Op::Delete { id: TaskId(3) }]);
        let i = interpret("cancel #3", &tasks, now());
        assert_eq!(i.ops, vec![Op::Delete { id: TaskId(3) }]);
        let i = interpret("forget about the gym class", &tasks, now());
        assert_eq!(i.ops, vec![Op::Delete { id: TaskId(3) }]);
    }

    #[test]
    fn moves_a_task() {
        let tasks = [task(4, "Dentist appointment", Status::Todo)];
        let i = interpret("move the dentist to friday", &tasks, now());
        let patch = TaskPatch {
            due: Some(Some(d(10, 30))),
            ..TaskPatch::default()
        };
        assert_eq!(
            i.ops,
            vec![Op::Update {
                id: TaskId(4),
                patch
            }]
        );
    }
}
