//! The worker thread: it waits for a [`Request::Load`], loads the model, then
//! answers requests one at a time until another load or the end of the channel.
//!
//! The GGUF, the tokenizer, the model, the engine and the chat borrow from one
//! another, so they are all locals of [`serve`]: returning from it unloads the
//! model.
//!
//! The system prompts are constant (see [`todo_core::prompt`]), so the model's
//! state after each one is saved ([`Prefixes`]) and every request starts from
//! a copy of it: only the user message is processed.

use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Once;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use biv::chat::{Chat, ChatOptions, Event, Prefix, ReasoningEffort};
use biv::engine::Engine;
use biv::gguf::Value;
use biv::model::Model;
use biv::sampler::{Sampler, SamplerConfig};
use biv::tokenizer::Tokenizer;
use chrono::{DateTime, Local};
use todo_core::Task;
use todo_core::plan;
use todo_core::prompt::{self, Prompt};

use crate::{Cancelled, Config, ModelStatus, Request, Update};

/// The shortest time between two [`Update::Partial`] of a request
const PARTIAL_INTERVAL: Duration = Duration::from_millis(100);

/// Tokens the chat template adds around the system and user texts, with room
/// for the reasoning instructions
const TEMPLATE_TOKENS: usize = 96;

pub(crate) struct Worker {
    pub requests: async_channel::Receiver<Request>,
    pub updates: async_channel::Sender<Update>,
    pub cancelled: Cancelled,
}

impl Worker {
    fn send(&self, update: Update) {
        // the channel is unbounded: this never waits; a closed one means nobody listens
        let _ = self.updates.send_blocking(update);
    }

    fn status(&self, status: ModelStatus) {
        self.send(Update::Status(status));
    }

    fn cancelled(&self, id: u64) -> bool {
        self.cancelled.contains(id)
    }

    /// Answers `Failed` "cancelled" for request `id` if it was cancelled
    /// before it was reached; returns whether it was
    fn skip_cancelled(&self, id: u64) -> bool {
        let cancelled = self.cancelled(id);
        if cancelled {
            self.cancelled.forget_up_to(id);
            self.send(Update::Failed { id, error: "cancelled".into() });
        }
        cancelled
    }
}

pub(crate) fn run(worker: Worker) {
    let mut next: Option<Config> = None;
    loop {
        let config = match next.take() {
            Some(config) => config,
            None => match worker.requests.recv_blocking() {
                Ok(Request::Load(config)) => config,
                Ok(Request::Interpret { id, .. } | Request::Review { id, .. }) => {
                    if !worker.skip_cancelled(id) {
                        worker.cancelled.forget_up_to(id);
                        worker.send(Update::Failed { id, error: "model not loaded".into() });
                    }
                    continue;
                }
                Err(_) => return,
            },
        };
        worker.status(ModelStatus::Loading);
        match catch_unwind(AssertUnwindSafe(|| serve(&config, &worker))) {
            Ok(Ok(Some(config))) => next = Some(config),
            Ok(Ok(None)) => return,
            Ok(Err(error)) => worker.status(ModelStatus::Failed(error)),
            Err(panic) => worker.status(ModelStatus::Failed(format!("the model crashed: {}", panic_message(&*panic)))),
        }
    }
}

/// Loads the model of `config` and answers requests with it. Returns the
/// configuration of the next load, or `None` once the channel is closed.
fn serve(config: &Config, worker: &Worker) -> Result<Option<Config>, String> {
    init_rayon();
    let started = Instant::now();
    let path = biv_cli::hf::resolve(&config.model)?;
    let gguf = biv_cli::open_gguf(&path).map_err(|e| e.to_string())?;
    let tok = Tokenizer::from_gguf(&gguf).map_err(|e| format!("{}: {e}", path.display()))?;
    let model = Model::load(&gguf).map_err(|e| format!("{}: {e}", path.display()))?;
    let engine = make_engine(&model, config.cpu)?;
    let name = gguf.get("general.name").and_then(Value::as_str).unwrap_or("model");
    let description = format!("{name}, {}", engine.description());
    let mut chat = Chat::new(engine.as_ref(), &tok, options(config, &model)).map_err(|e| e.to_string())?;
    // declared after the chat and the engine, so dropped before them: a prefix
    // is the state of a sequence of this model on this engine, and means
    // nothing to another
    let mut prefixes = Prefixes::default();
    let mut config = config.clone();

    // The interpretation's system prompt is processed before saying Ready, so
    // that the first request is as fast as the next ones: until Ready the app
    // answers requests with its rules, without waiting. The review's is saved
    // by the first review, which comes once the list has been quiet a while.
    // A load asked for meanwhile is only seen between those steps: the
    // prompt's processing can't be interrupted.
    let mut waiting = match warm(&mut chat, &mut prefixes, &config, worker) {
        None => return Ok(None),
        Some(Some(Request::Load(next))) if !same_weights(&next, &config) => return Ok(Some(next)),
        Some(waiting) => waiting,
    };
    let ready = ModelStatus::Ready { description, path: path.clone(), load_time: started.elapsed() };
    worker.status(ready.clone());

    loop {
        let request = match waiting.take() {
            Some(request) => request,
            None => match worker.requests.recv_blocking() {
                Ok(request) => request,
                Err(_) => break,
            },
        };
        let (id, job) = match request {
            // the same weights: only the chat options change
            Request::Load(next) if same_weights(&next, &config) => {
                let new_session = next.context != config.context;
                chat.opts = options(&next, &model);
                if next.thinking != config.thinking || !next.cache_prefix {
                    prefixes.saved.clear();
                }
                // a load is a chance to try saving prefixes again
                prefixes.broken = false;
                config = next;
                // the session has the size of the context it was made with (the
                // plain path makes a new one for every request anyway)
                if new_session && let Err(e) = chat.reset() {
                    prefixes.fail(&e);
                }
                waiting = match warm(&mut chat, &mut prefixes, &config, worker) {
                    None => break,
                    Some(Some(Request::Load(next))) if !same_weights(&next, &config) => return Ok(Some(next)),
                    Some(waiting) => waiting,
                };
                worker.status(ready.clone());
                continue;
            }
            Request::Load(next) => return Ok(Some(next)),
            Request::Interpret { id, request, tasks, now } => (id, Job::Interpret { request, tasks, now }),
            Request::Review { id, tasks, now } => (id, Job::Review { tasks, now }),
        };
        if worker.skip_cancelled(id) {
            continue;
        }
        let started = Instant::now();
        let update = match catch_unwind(AssertUnwindSafe(|| {
            answer(&mut chat, &mut prefixes, &tok, &config, id, job, worker)
        })) {
            Ok(Ok(Answer::Interpreted(interpretation))) => {
                Update::Interpreted { id, interpretation, elapsed: started.elapsed() }
            }
            Ok(Ok(Answer::Reviewed(plan))) => Update::Reviewed { id, plan, elapsed: started.elapsed() },
            Ok(Err(error)) => Update::Failed { id, error },
            Err(panic) => Update::Failed { id, error: format!("the model crashed: {}", panic_message(&*panic)) },
        };
        worker.cancelled.forget_up_to(id);
        worker.send(update);
    }
    Ok(None)
}

/// Whether `next` loads the same weights as `config` on the same engine, so
/// that only the chat options change
fn same_weights(next: &Config, config: &Config) -> bool {
    next.model == config.model && next.cpu == config.cpu
}

/// Saves the interpretation's prefix if it is not, unless a request is
/// already waiting (it is answered first, and saves the prefix itself if it
/// needs it) or prefixes are off or broken. Returns the request waiting after that, if any, or `None` once
/// the channel is closed.
fn warm(chat: &mut Chat<'_>, prefixes: &mut Prefixes, config: &Config, worker: &Worker) -> Option<Option<Request>> {
    let waiting = |worker: &Worker| match worker.requests.try_recv() {
        Ok(request) => Some(Some(request)),
        Err(async_channel::TryRecvError::Empty) => Some(None),
        Err(async_channel::TryRecvError::Closed) => None,
    };
    let before = waiting(worker)?;
    // broken: a reset on reload failed, a prefix saved now would not be used
    if before.is_some() || !config.cache_prefix || prefixes.broken {
        return Some(before);
    }
    chat.opts.system = Some(prompt::INTERPRET_SYSTEM.into());
    match prefixes.start(chat) {
        Ok(Start::Saved { tokens, bytes, time }) => trace(|| {
            format!("saved the interpretation's system prompt, {tokens} tokens, {} MB in {:.1}s", bytes >> 20, time.as_secs_f32())
        }),
        Ok(_) => {}
        Err(e) => prefixes.fail(&e),
    }
    waiting(worker)
}

/// The model's state after each system prompt, saved to start requests from
/// instead of processing the system prompt again. Each is about 150 MB plus
/// 64 KB a token on Metal (twice that per token on the CPU).
#[derive(Default)]
struct Prefixes {
    saved: Vec<Saved>,
    /// a save or restore failed: the system prompt is processed with every
    /// message, as without prefixes, until the next load
    broken: bool,
}

struct Saved {
    /// what the system block was made of: a restored chat answers as the
    /// one that saved it, whatever its own options say
    system: String,
    thinking: bool,
    effort: ReasoningEffort,
    prefix: Prefix,
}

/// How a request's turn started
#[derive(Debug, Clone, Copy)]
enum Start {
    /// from the saved state after its system prompt, of that many tokens
    Restored { tokens: usize, time: Duration },
    /// the system prompt was processed and saved first
    Saved { tokens: usize, bytes: usize, time: Duration },
    /// a new session: the system prompt is processed with the message
    Plain,
}

impl Prefixes {
    /// Puts `chat` at the end of its system block (`chat.opts.system`):
    /// restored from the prefix saved with these options, or saved now
    fn start(&mut self, chat: &mut Chat<'_>) -> Result<Start, biv::Error> {
        catch_unwind(AssertUnwindSafe(|| self.start_unguarded(chat)))
            .unwrap_or_else(|panic| Err(biv::Error::Backend(format!("crashed: {}", panic_message(&*panic)))))
    }

    fn start_unguarded(&mut self, chat: &mut Chat<'_>) -> Result<Start, biv::Error> {
        let system = chat.opts.system.clone().unwrap_or_default();
        let (thinking, effort) = (chat.opts.thinking, chat.opts.effort);
        let started = Instant::now();
        if let Some(saved) =
            self.saved.iter().find(|s| s.system == system && s.thinking == thinking && s.effort == effort)
        {
            chat.restore(&saved.prefix)?;
            return Ok(Start::Restored { tokens: saved.prefix.n_past(), time: started.elapsed() });
        }

        // saved with other options: replaced
        self.saved.retain(|s| s.system != system);
        // save() processes the system block only at the start of a session
        chat.reset()?;
        let prefix = chat.save()?;
        // the chat is now where the prefix leaves it: no restore needed
        let (tokens, bytes, time) = (prefix.n_past(), prefix.bytes(), started.elapsed());
        let start = Start::Saved { tokens, bytes, time };
        self.saved.push(Saved { system, thinking, effort, prefix });
        Ok(start)
    }

    /// Stops using prefixes after `error`, until the next load
    fn fail(&mut self, error: &biv::Error) {
        eprintln!("assistant: cannot save or restore the system prompt's state, it is processed with each request: {error}");
        self.broken = true;
        self.saved.clear();
    }
}

/// Puts `chat` at the start of a turn under `system`: from its saved prefix
/// if prefixes are on and work, else in a new session
fn start_turn(chat: &mut Chat<'_>, prefixes: &mut Prefixes, system: &str, cache: bool) -> Result<Start, String> {
    chat.opts.system = Some(system.into());
    if cache && !prefixes.broken {
        match prefixes.start(chat) {
            Ok(start) => return Ok(start),
            Err(e) => prefixes.fail(&e),
        }
    }
    // a new session: the system block is only emitted on the first turn of one
    chat.reset().map_err(|e| e.to_string())?;
    Ok(Start::Plain)
}

/// Prints the message of `f` to stderr if `TODO_BIV_TRACE` is set
fn trace(f: impl FnOnce() -> String) {
    if std::env::var_os("TODO_BIV_TRACE").is_some() {
        eprintln!("assistant: {}", f());
    }
}

/// What a request asks of the model
enum Job {
    Interpret { request: String, tasks: Vec<Task>, now: DateTime<Local> },
    Review { tasks: Vec<Task>, now: DateTime<Local> },
}

enum Answer {
    Interpreted(prompt::Interpretation),
    Reviewed(todo_core::Plan),
}

impl Job {
    fn tasks(&self) -> &[Task] {
        match self {
            Job::Interpret { tasks, .. } | Job::Review { tasks, .. } => tasks,
        }
    }

    fn now(&self) -> DateTime<Local> {
        match self {
            Job::Interpret { now, .. } | Job::Review { now, .. } => *now,
        }
    }

    fn prompt(&self, tasks: &[Task]) -> Prompt {
        match self {
            Job::Interpret { request, now, .. } => prompt::interpret_prompt(request, tasks, *now),
            Job::Review { now, .. } => prompt::review_prompt(tasks, *now),
        }
    }

    fn parse(&self, reply: &str, tasks: &[Task]) -> anyhow::Result<Answer> {
        Ok(match self {
            Job::Interpret { now, .. } => Answer::Interpreted(prompt::parse_interpretation(reply, tasks, *now)?),
            Job::Review { now, .. } => {
                let mut plan = prompt::parse_review(reply, tasks, *now)?;
                plan.from_model = true;
                Answer::Reviewed(plan)
            }
        })
    }
}

/// One request: the prompt, the model's answer, its parsing
fn answer(
    chat: &mut Chat<'_>,
    prefixes: &mut Prefixes,
    tok: &Tokenizer,
    config: &Config,
    id: u64,
    job: Job,
    worker: &Worker,
) -> Result<Answer, String> {
    let room = answer_room(chat.opts.context, chat.opts.max_tokens);
    // the system block counts in the context whether it is restored or not
    let (prompt, tasks) = fit(&job, tok, chat.opts.context.saturating_sub(room))?;

    let start = start_turn(chat, prefixes, &prompt.system, config.cache_prefix)?;
    let mut sampler = Sampler::new(sampler_config(config));

    let mut reasoning = String::new();
    let mut reply = String::new();
    let mut last_partial = Instant::now();
    let mut shown = 0;
    let started = Instant::now();
    let mut prefill = None;
    let result = chat.send(
        &prompt.user,
        &mut sampler,
        |event| {
            match event {
                Event::Thinking(t) => reasoning.push_str(t),
                Event::Answer(t) => reply.push_str(t),
                Event::PromptDone => {
                    prefill = Some(started.elapsed());
                    return;
                }
                // the end of the reasoning (and no tools are given)
                _ => return,
            }
            let text = if reply.is_empty() { &reasoning } else { &reply };
            if last_partial.elapsed() >= PARTIAL_INTERVAL && text.len() != shown {
                last_partial = Instant::now();
                shown = text.len();
                worker.send(Update::Partial { id, text: text.clone() });
            }
        },
        || worker.cancelled(id),
    );
    if worker.cancelled(id) {
        return Err("cancelled".into());
    }
    let stats = result.map_err(|e| match e {
        biv::Error::ContextFull { .. } => format!("the list is too long for the model's context: {e}"),
        e => e.to_string(),
    })?;
    // the whole answer, the last partial text
    let text = if reply.is_empty() { &reasoning } else { &reply };
    if text.len() != shown {
        worker.send(Update::Partial { id, text: text.clone() });
    }
    trace(|| {
        let prefill = prefill.unwrap_or_default().as_secs_f32();
        let total = started.elapsed().as_secs_f32();
        let system = match start {
            Start::Restored { tokens, time } => {
                format!("system prompt of {tokens} tokens restored in {:.0} ms", time.as_secs_f32() * 1000.0)
            }
            Start::Saved { tokens, bytes, time } => {
                format!("system prompt of {tokens} tokens saved first, {} MB in {:.1}s", bytes >> 20, time.as_secs_f32())
            }
            Start::Plain => "system prompt included".into(),
        };
        format!(
            "#{id} prompt {} tokens in {prefill:.1}s ({system}), answer {} tokens in {:.1}s",
            stats.prompt_tokens,
            stats.gen_tokens,
            total - prefill
        )
    });
    job.parse(&reply, &tasks).map_err(|e| {
        eprintln!("assistant: cannot parse the model's answer ({e:#}):\n{reply}");
        if stats.truncated {
            format!("the answer was cut at {} tokens: {e:#}", stats.gen_tokens)
        } else {
            format!("the model's answer could not be understood: {e:#}")
        }
    })
}

/// Tokens to leave free for the answer
fn answer_room(context: usize, max_tokens: usize) -> usize {
    max_tokens.min(context / 2)
}

/// The prompt and the tasks it shows the model: all of them if that leaves
/// `budget` tokens or fewer for the prompt, else the open ones, the most
/// pressing first, as many as fit
fn fit(job: &Job, tok: &Tokenizer, budget: usize) -> Result<(Prompt, Vec<Task>), String> {
    let size = |p: &Prompt| tok.encode(&p.system).len() + tok.encode(&p.user).len() + TEMPLATE_TOKENS;
    let all = job.tasks();
    let prompt = job.prompt(all);
    if size(&prompt) <= budget {
        return Ok((prompt, all.to_vec()));
    }

    let now = job.now();
    let mut ranked: Vec<&Task> = all.iter().filter(|t| t.is_open()).collect();
    ranked.sort_by(|a, b| plan::score(b, now).total_cmp(&plan::score(a, now)));
    // the first `n` ranked tasks, in the list's order
    let subset = |n: usize| -> Vec<Task> {
        let keep: Vec<_> = ranked[..n].iter().map(|t| t.id).collect();
        all.iter().filter(|t| keep.contains(&t.id)).cloned().collect()
    };
    // the largest count that fits, by bisection: prompts grow with the tasks
    let (mut lo, mut hi) = (0, ranked.len());
    let mut best = None;
    while lo <= hi {
        let mid = (lo + hi) / 2;
        let tasks = subset(mid);
        let prompt = job.prompt(&tasks);
        if size(&prompt) <= budget {
            best = Some((prompt, tasks));
            lo = mid + 1;
        } else if mid == 0 {
            break;
        } else {
            hi = mid - 1;
        }
    }
    best.ok_or_else(|| format!("the request does not fit in the model's context ({budget} tokens)"))
}

fn options(config: &Config, model: &Model<'_>) -> ChatOptions {
    // no system prompt (each request sets its own), the engine's batch size,
    // no tools
    ChatOptions {
        thinking: config.thinking,
        // a to-do list rarely needs long deliberation
        effort: ReasoningEffort::Low,
        max_tokens: config.max_tokens.max(1),
        context: config.context.min(model.cfg.context_length),
        draft: 1,
        ..ChatOptions::default()
    }
}

/// Near-greedy sampling, for well-formed JSON; reasoning loops less when it
/// is given a little more room
fn sampler_config(config: &Config) -> SamplerConfig {
    if config.greedy {
        return SamplerConfig { temperature: 0.0, ..SamplerConfig::default() };
    }
    let seed = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0);
    let (temperature, top_p) = if config.thinking { (0.6, 0.95) } else { (0.2, 0.8) };
    SamplerConfig { temperature, top_k: 20, top_p, seed, ..SamplerConfig::default() }
}

fn make_engine<'m>(model: &'m Model<'m>, cpu: bool) -> Result<Box<dyn Engine + 'm>, String> {
    #[cfg(target_os = "macos")]
    if !cpu {
        return Ok(Box::new(biv::metal::MetalEngine::new(model, 256).map_err(|e| e.to_string())?));
    }
    // elsewhere the CPU is the only backend
    let _ = cpu;
    Ok(Box::new(model))
}

/// rayon's global pool runs the CPU compute; it can only be built once
fn init_rayon() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        // efficiency cores included: the matmuls gain from them
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        // an error means the app built the pool already, which is as good
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|i| format!("biv-{i}"))
            .build_global();
    });
}

fn panic_message(panic: &(dyn Any + Send)) -> String {
    if let Some(s) = panic.downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = panic.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn answer_room_leaves_half_the_context_at_most() {
        assert_eq!(answer_room(8192, 1536), 1536);
        assert_eq!(answer_room(2048, 1536), 1024);
    }

    #[test]
    fn only_the_model_and_the_engine_need_a_new_load() {
        let config = Config::default();
        let options = Config { thinking: true, context: 4096, cache_prefix: false, greedy: true, ..config.clone() };
        assert!(same_weights(&options, &config));
        assert!(!same_weights(&Config { cpu: true, ..config.clone() }, &config));
        assert!(!same_weights(&Config { model: "other.gguf".into(), ..config.clone() }, &config));
    }

    #[test]
    fn greedy_sampling_takes_the_most_likely_token() {
        let config = Config { greedy: true, ..Config::default() };
        assert_eq!(sampler_config(&config).temperature, 0.0);
        assert!(sampler_config(&Config::default()).temperature > 0.0);
    }

    #[test]
    fn panic_messages() {
        let p = catch_unwind(|| panic!("boom")).unwrap_err();
        assert_eq!(panic_message(&*p), "boom");
        let p = catch_unwind(|| panic!("{}", 42)).unwrap_err();
        assert_eq!(panic_message(&*p), "42");
    }
}
