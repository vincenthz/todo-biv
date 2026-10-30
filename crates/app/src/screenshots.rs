//! Renders the main window with a sample list, in light and dark, to PNG
//! files for looking at the layout: `cargo test -p todo-biv screenshots --
//! --ignored`. Files go to `target/screenshots`, or `SCREENSHOTS` if set.

use std::{path::PathBuf, sync::Arc, time::Duration};

use assistant::{Config, ModelStatus};
use chrono::{Duration as Days, Local};
use gpui::{AnyWindowHandle, AppContext as _, Entity, HeadlessAppContext, Window, px, size};
use gpui_component::{Root, Theme, ThemeMode};
use speech::whisper::hub;
use todo_core::{NewTask, Op, Plan, PlanItem, Priority, PrioritySuggestion, Status, Store, TaskId, TaskPatch};

use crate::{
    lists::View,
    services::{
        model::ModelService,
        reviewer::Reviewer,
        speech::{SpeechService, SpeechStatus},
    },
    settings::Settings,
    state::TodoState,
    views::shell::Shell,
};

/// A list with something of everything: overdue, due today, upcoming and
/// undated tasks of every priority, one in progress, two done, and a plan
/// the model wrote five minutes ago with a priority suggestion
pub fn sample_store() -> Store {
    let now = Local::now();
    let today = now.date_naive();
    let day = |n: i64| Some(today + Days::days(n));
    let mut store = Store::in_memory();
    let add = |title: &str, notes: &str, category: &str, priority: Priority, due, tags: &[&str], estimate| {
        Op::Add(NewTask {
            title: title.into(),
            notes: notes.into(),
            category: Some(category.into()),
            priority: Some(priority),
            due,
            tags: tags.iter().map(|t| t.to_string()).collect(),
            estimate_minutes: estimate,
        })
    };
    let ops = [
        add("Finish the quarterly slides", "Numbers from finance; 12 slides at most", "Work", Priority::Urgent, day(1), &["q4"], Some(120)),
        add("File the tax return", "", "Finance", Priority::High, day(-2), &[], Some(60)),
        add("Call mum", "Ask about the weekend", "Personal", Priority::High, day(0), &["phone"], None),
        add("Buy milk and eggs", "", "Errands", Priority::Low, day(0), &[], None),
        add("Dentist appointment", "Dr. Moreau, 3rd floor", "Health", Priority::Medium, day(4), &[], None),
        add("Fix the leaking kitchen tap", "", "Home", Priority::Medium, None, &[], Some(30)),
        add("Renew the passport", "Photos first", "Personal", Priority::Medium, day(30), &["admin"], None),
        add("Book a table for Sam's birthday", "", "Social", Priority::Medium, day(9), &[], None),
        add("Read chapter 5 of the systems book", "", "Learning", Priority::Low, None, &[], Some(45)),
        add("Send the report to Anna", "", "Work", Priority::High, None, &[], None),
        add("Pay the electricity bill", "", "Finance", Priority::Medium, None, &[], None),
    ];
    let ids = store.apply(&ops, "", now);
    let id = |n: usize| ids[n];
    let started = TaskPatch { status: Some(Status::InProgress), ..Default::default() };
    store.apply(
        &[
            Op::Update { id: id(6), patch: started },
            Op::Complete { id: id(9) },
            Op::Complete { id: id(10) },
        ],
        "",
        now,
    );
    if let Some(task) = store.tasks.iter_mut().find(|t| t.id == id(2)) {
        task.source = "remind me to call mum today, it's important".into();
    }
    let item = |n: usize, reason: &str| PlanItem { id: id(n), reason: reason.into() };
    store.plan = Some(Plan {
        generated: now - Days::minutes(5),
        summary: "A busy Friday: the slides are due tomorrow and the tax return is overdue. Clear the two quick calls first.".into(),
        next: vec![
            item(2, "Due today and quick: five minutes on the phone."),
            item(1, "Overdue by two days; penalties start next week."),
            item(0, "Due tomorrow and the largest task left."),
            item(3, "On the way home."),
            item(4, "Confirm the time before Tuesday."),
            item(6, "Already started; photos are the next step."),
            item(5, "A small job for the weekend."),
            item(7, "Restaurants fill up a week ahead."),
            item(8, "Whenever there is time."),
        ],
        suggestions: vec![PrioritySuggestion {
            id: id(7),
            priority: Priority::High,
            reason: "Saturday tables at the usual place go a week ahead.".into(),
        }],
        from_model: true,
    });
    store
}

struct Shooter {
    cx: HeadlessAppContext,
    window: AnyWindowHandle,
    shell: Entity<Shell>,
    state: Entity<TodoState>,
    dir: PathBuf,
}

impl Shooter {
    fn update<R>(&mut self, f: impl FnOnce(&Entity<Shell>, &mut Window, &mut gpui::App) -> R) -> R {
        let shell = self.shell.clone();
        let result = self.cx.update_window(self.window, |_, window, cx| f(&shell, window, cx)).expect("the window is open");
        self.cx.run_until_parked();
        result
    }

    fn state(&mut self, f: impl FnOnce(&mut TodoState, &mut gpui::Context<TodoState>)) {
        let state = self.state.clone();
        self.update(|_, _, cx| state.update(cx, f));
    }

    fn shoot(&mut self, name: &str) {
        // Two frames: the first lays out, the second paints what it measured.
        for _ in 0..2 {
            self.update(|_, window, cx| window.draw(cx).clear(cx));
        }
        let image = self.cx.capture_screenshot(self.window).expect("rendering the window");
        let path = self.dir.join(format!("{name}.png"));
        image.save(&path).expect("saving the screenshot");
        println!("{}", path.display());
    }

    fn theme(&mut self, mode: ThemeMode) {
        self.update(|_, window, cx| {
            Theme::change(mode, Some(window), cx);
            crate::palette::apply(cx);
            window.refresh();
        });
    }
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "writes PNG files for inspection"]
fn screenshots() {
    let dir = std::env::var_os("SCREENSHOTS")
        .map_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/screenshots"), PathBuf::from);
    std::fs::create_dir_all(&dir).expect("creating the screenshot directory");
    let width: f32 = std::env::var("SCREENSHOT_WIDTH").ok().and_then(|w| w.parse().ok()).unwrap_or(1240.);
    let height: f32 = std::env::var("SCREENSHOT_HEIGHT").ok().and_then(|h| h.parse().ok()).unwrap_or(820.);
    // The platform itself must be made on the main thread, which tests
    // are not run on; its text system and renderer need not.
    let text_system = Arc::new(gpui_apple::AppleTextSystem::new());
    let mut cx = HeadlessAppContext::with_platform(text_system, Arc::new(gpui_kit_assets::AllAssets), gpui_platform::current_headless_renderer);
    cx.update(|cx| {
        gpui_component::init(cx);
        crate::views::shell::bind_keys(cx);
        Theme::change(ThemeMode::Light, None, cx);
        crate::palette::apply(cx);
    });

    let settings = Settings { microphone: true, ..Settings::default() };
    let state = cx.update(|cx| cx.new(|_| TodoState::in_memory(sample_store(), settings)));
    let model = cx.update(|cx| {
        cx.new(|_| {
            let mut model = ModelService::without_model();
            let ready = ModelStatus::Ready {
                description: "Qwen3.8 27B, Metal".into(),
                path: PathBuf::from("~/.cache/huggingface/hub/Qwen3.8-27B-Q4_1.gguf"),
                load_time: Duration::from_millis(8400),
            };
            model.set_status(ready, Some(Config::default()));
            model
        })
    });
    let speech = cx.update(|cx| {
        cx.new(|_| {
            let mut speech = SpeechService::without_worker(hub::checkpoint_or_default(hub::DEFAULT_CHECKPOINT));
            speech.set_status(SpeechStatus::Ready { device: "GPU".into() });
            speech
        })
    });
    let reviewer = cx.update(|cx| cx.new(|cx| Reviewer::new(state.clone(), model.clone(), cx)));
    let mut shell = None;
    let window = cx
        .open_window(size(px(width), px(height)), |window, cx| {
            let view = cx.new(|cx| Shell::new(state.clone(), model, speech, reviewer, window, cx));
            shell = Some(view.clone());
            cx.new(|cx| Root::new(view, window, cx))
        })
        .expect("opening a window");
    cx.run_until_parked();
    let mut shooter = Shooter { cx, window: window.into(), shell: shell.expect("the window was built"), state, dir };

    for (mode, theme) in [(ThemeMode::Light, "light"), (ThemeMode::Dark, "dark")] {
        shooter.theme(mode);
        shooter.state(|state, cx| state.select_view(View::NextUp, cx));
        shooter.shoot(&format!("{theme}-next-up"));

        // A request just answered: its tasks highlighted, with its undo.
        shooter.update(|shell, _, cx| {
            let bar = shell.read(cx).request_bar.clone();
            let state = shell.read(cx).state();
            let ops = [Op::Add(NewTask {
                title: "Call the bank about the card".into(),
                category: Some("Finance".into()),
                priority: Some(Priority::High),
                due: Some(Local::now().date_naive() + Days::days(1)),
                ..Default::default()
            })];
            state.update(cx, |state, cx| state.apply_request(&ops, "call the bank tomorrow, it's important", cx));
            bar.update(cx, |bar, cx| bar.show_answer("Added “Call the bank about the card”, high priority, due tomorrow.", 1, cx));
            state.update(cx, |state, cx| state.select_view(View::AllOpen, cx));
        });
        shooter.shoot(&format!("{theme}-all-open-after-request"));
        shooter.state(|state, cx| {
            state.undo(cx);
        });

        let first = TaskId(1);
        shooter.state(|state, cx| state.reveal(first, cx));
        shooter.shoot(&format!("{theme}-editor"));
        shooter.state(|state, cx| state.toggle_expanded(first, cx));

        for (name, view) in [("today", View::Today), ("upcoming", View::Upcoming), ("done", View::Done), ("work", View::Category("Work".into()))] {
            shooter.state(|state, cx| state.select_view(view, cx));
            shooter.shoot(&format!("{theme}-{name}"));
        }

        shooter.update(|shell, window, cx| shell.update(cx, |shell, cx| shell.open_settings(true, window, cx)));
        shooter.shoot(&format!("{theme}-settings"));
        shooter.update(|shell, window, cx| shell.update(cx, |shell, cx| shell.open_settings(false, window, cx)));
    }

    // A list that could not be read entirely at startup
    shooter.theme(ThemeMode::Light);
    shooter.state(|state, cx| {
        state.select_view(View::NextUp, cx);
        state.show_load_problem(
            "The list could not be read entirely (reading tasks.json: unknown variant `critical`). It was copied to \
             ~/Library/Application Support/todo-biv/tasks.json.unreadable-20261030-090000, and the 9 tasks that could be read were kept."
                .into(),
            cx,
        );
    });
    shooter.shoot("light-load-problem");
    shooter.state(|state, cx| state.dismiss_load_problems(cx));

    // A first start: nothing to do, the model loading, ranked by rules.
    shooter.state(|state, cx| {
        state.store = Store::in_memory();
        state.select_view(View::NextUp, cx);
    });
    shooter.update(|shell, _, cx| {
        let model = shell.read(cx).model();
        model.update(cx, |model, cx| {
            model.set_status(ModelStatus::Loading, None);
            cx.notify();
        });
        let bar = shell.read(cx).request_bar.clone();
        bar.update(cx, |bar, cx| bar.clear_reply(cx));
    });
    shooter.shoot("light-empty");
}
