//! Runs a few requests through the model on an in-memory list and prints the
//! operations, the plan and the timings.
//!
//! `cargo run --release -p assistant --example try [-- --think] [--cpu] [--no-cache] [--greedy] [--compare] [--reviews N] [-r REQUEST]... [model]`
//! (each `-r` replaces the built-in requests; `--reviews` asks for the review
//! that many times)
//!
//! `--no-cache` processes the system prompt with every request instead of
//! starting from its saved state. `--compare` answers the requests greedily
//! from the saved system prompt, then again with the same lists and times
//! without it, and checks that the answers are the same text.

use std::io::Write;
use std::time::Instant;

use assistant::{Assistant, Config, ModelStatus, Request, Update};
use chrono::{DateTime, Local};
use todo_core::{Op, Store, Task};

fn main() -> anyhow::Result<()> {
    let mut config = Config::default();
    let mut requests: Vec<String> = Vec::new();
    let mut compare = false;
    let mut reviews = 1;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--think" => config.thinking = true,
            "--cpu" => config.cpu = true,
            "--no-cache" => config.cache_prefix = false,
            "--greedy" => config.greedy = true,
            "--compare" => compare = true,
            "--reviews" => reviews = args.next().and_then(|n| n.parse().ok()).unwrap_or(1),
            "-r" => requests.extend(args.next()),
            model => config.model = model.into(),
        }
    }
    if requests.is_empty() {
        requests = REQUESTS.iter().map(|r| r.to_string()).collect();
    }
    let (assistant, updates) = Assistant::spawn();
    let mut model = Model { assistant, updates, next_id: 0 };
    if compare {
        config.greedy = true;
        config.cache_prefix = true;
    }
    model.load(&config)?;

    let mut store = Store::in_memory();
    let mut asked = Vec::new();
    let mut latencies = Vec::new();
    for request in &requests {
        let now = Local::now();
        println!("\n> {request}");
        let asking = Asking::Interpret { request: request.clone(), tasks: store.tasks.clone(), now };
        let answer = model.ask(&asking)?;
        match &answer.update {
            Update::Interpreted { interpretation, elapsed, .. } => {
                for op in &interpretation.ops {
                    println!("  {}", describe(op));
                }
                println!("  reply: {}", interpretation.reply);
                println!("  ({:.1}s)", elapsed.as_secs_f32());
                latencies.push(elapsed.as_secs_f32());
                let touched = store.apply(&interpretation.ops, request, now);
                println!("  touched {touched:?}");
            }
            Update::Failed { error, .. } => println!("  failed: {error}"),
            _ => unreachable!(),
        }
        asked.push((asking, answer));
    }
    summary("requests", &latencies);

    println!("\nlist:");
    for t in &store.tasks {
        let due = t.due.map(|d| d.to_string()).unwrap_or_else(|| "-".into());
        println!("  #{} [{}] {} | {} | {} | due {due}", t.id.0, t.status.label(), t.title, t.category, t.priority.label());
    }

    let now = Local::now();
    for _ in 0..reviews {
        println!("\nreview:");
        let asking = Asking::Review { tasks: store.tasks.clone(), now };
        let answer = model.ask(&asking)?;
        match &answer.update {
            Update::Reviewed { plan, elapsed, .. } => {
                println!("  {}", plan.summary);
                for (i, item) in plan.next.iter().enumerate() {
                    let title = store.get(item.id).map(|t| t.title.as_str()).unwrap_or("?");
                    println!("  {}. #{} {title}: {}", i + 1, item.id.0, item.reason);
                }
                for s in &plan.suggestions {
                    println!("  suggest #{} -> {}: {}", s.id.0, s.priority.label(), s.reason);
                }
                println!("  ({:.1}s)", elapsed.as_secs_f32());
            }
            Update::Failed { error, .. } => println!("  failed: {error}"),
            _ => unreachable!(),
        }
        asked.push((asking, answer));
    }

    if compare {
        self::compare(&mut model, &config, &asked)?;
    }
    Ok(())
}

/// Asks the same things again, with the system prompt processed with every
/// request, and compares the answers with `asked`
fn compare(model: &mut Model, config: &Config, asked: &[(Asking, Answer)]) -> anyhow::Result<()> {
    println!("\nagain, the system prompt processed with each request:");
    model.load(&Config { cache_prefix: false, ..config.clone() })?;
    let (mut same, mut cached, mut plain) = (0, Vec::new(), Vec::new());
    for (asking, before) in asked {
        let after = model.ask(asking)?;
        let (a, b) = (before.elapsed(), after.elapsed());
        let what = match asking {
            Asking::Interpret { request, .. } => format!("> {request}"),
            Asking::Review { .. } => "review".into(),
        };
        if before.text == after.text && !before.text.is_empty() {
            same += 1;
            println!("  same text ({a:.1}s from the saved prompt, {b:.1}s without): {what}");
        } else {
            println!("  DIFFERENT: {what}\n    saved prompt: {:?}\n    without:      {:?}", before.text, after.text);
        }
        cached.push(a);
        plain.push(b);
    }
    println!("\n{same} of {} answers are the same text", asked.len());
    summary("from the saved system prompt", &cached);
    summary("with the system prompt processed each time", &plain);
    anyhow::ensure!(same == asked.len(), "the answers differ");
    Ok(())
}

fn summary(what: &str, latencies: &[f32]) {
    if !latencies.is_empty() {
        let mean = latencies.iter().sum::<f32>() / latencies.len() as f32;
        let max = latencies.iter().copied().fold(0.0, f32::max);
        println!("\n{} {what}, mean {mean:.1}s, max {max:.1}s", latencies.len());
    }
}

/// A session exercising several tasks per request, relative dates,
/// completions, moves and deletions
const REQUESTS: &[&str] = &[
    "buy milk, call the bank and finish the slides by friday",
    "remind me to call mum tomorrow, it's important",
    "dentist appointment next tuesday at 3pm",
    "renew my passport next month, no rush",
    "I finished the slides",
    "move the dentist to friday",
    "I bought the milk and called the bank",
    "actually forget about the passport",
];

/// What a request asks, kept to ask it again
enum Asking {
    Interpret { request: String, tasks: Vec<Task>, now: DateTime<Local> },
    Review { tasks: Vec<Task>, now: DateTime<Local> },
}

struct Answer {
    /// the final update of the request
    update: Update,
    /// the model's whole answer, the last partial text
    text: String,
}

impl Answer {
    fn elapsed(&self) -> f32 {
        match &self.update {
            Update::Interpreted { elapsed, .. } | Update::Reviewed { elapsed, .. } => elapsed.as_secs_f32(),
            _ => f32::NAN,
        }
    }
}

struct Model {
    assistant: Assistant,
    updates: async_channel::Receiver<Update>,
    next_id: u64,
}

impl Model {
    fn load(&mut self, config: &Config) -> anyhow::Result<()> {
        let started = Instant::now();
        self.assistant.send(Request::Load(config.clone()));
        loop {
            match self.updates.recv_blocking()? {
                Update::Status(ModelStatus::Ready { description, path, load_time }) => {
                    println!("loaded {description} from {} in {:.1}s", path.display(), load_time.as_secs_f32());
                    return Ok(());
                }
                Update::Status(ModelStatus::Failed(e)) => anyhow::bail!("loading {}: {e}", config.model),
                Update::Status(status) => println!("{status:?} ({:.1}s)", started.elapsed().as_secs_f32()),
                _ => {}
            }
        }
    }

    /// Asks and waits for the answer, partial answers shown as dots
    fn ask(&mut self, asking: &Asking) -> anyhow::Result<Answer> {
        self.next_id += 1;
        let id = self.next_id;
        self.assistant.send(match asking {
            Asking::Interpret { request, tasks, now } => {
                Request::Interpret { id, request: request.clone(), tasks: tasks.clone(), now: *now }
            }
            Asking::Review { tasks, now } => Request::Review { id, tasks: tasks.clone(), now: *now },
        });
        let mut text = String::new();
        loop {
            match self.updates.recv_blocking()? {
                Update::Partial { id: i, text: partial } if i == id => {
                    text = partial;
                    print!(".");
                    let _ = std::io::stdout().flush();
                }
                update @ (Update::Interpreted { id: i, .. }
                | Update::Reviewed { id: i, .. }
                | Update::Failed { id: i, .. })
                    if i == id =>
                {
                    println!();
                    return Ok(Answer { update, text });
                }
                _ => {}
            }
        }
    }
}

fn describe(op: &Op) -> String {
    match op {
        Op::Add(t) => format!(
            "add {:?} category {:?} priority {:?} due {:?} estimate {:?} tags {:?}",
            t.title, t.category, t.priority, t.due, t.estimate_minutes, t.tags
        ),
        Op::Update { id, patch } => format!("update #{} {patch:?}", id.0),
        Op::Complete { id } => format!("complete #{}", id.0),
        Op::Reopen { id } => format!("reopen #{}", id.0),
        Op::Delete { id } => format!("delete #{}", id.0),
    }
}
