//! Where requests are made: a text field (and, in microphone mode, a record
//! button), and under it what came of the last request, with its undo.
//!
//! A request goes to the model when it is loaded, otherwise (or if the model
//! fails) the rules of [`todo_core::heuristics`] read it at once.

use std::time::{Duration, Instant};

use chrono::Local;
use gpui::{Context, Entity, Subscription, Task, Window, div, prelude::*, px};
use gpui_component::{
    ActiveTheme as _, Disableable as _, Icon, Sizable as _, StyledExt as _,
    button::{Button, ButtonRounded, ButtonVariants as _},
    h_flex,
    input::{Input, InputEvent, InputState},
    spinner::Spinner,
    v_flex,
};
use gpui_kit_assets::IconName;
use speech::audio::capture::Recorder;
use todo_core::{heuristics, prompt::Interpretation};

use crate::{
    palette,
    services::{
        model::{ModelEvent, ModelService},
        speech::{SpeechEvent, SpeechService, SpeechStatus},
    },
    state::TodoState,
    ui,
};

pub const PLACEHOLDER: &str = "Tell me what to do… e.g. “call the dentist friday, urgent”";

/// Recordings shorter than this are taken for a slip and dropped
const SHORTEST_RECORDING: f32 = 0.3;
/// A recording this long is stopped and transcribed: it was likely forgotten
const LONGEST_RECORDING: f32 = 120.0;

/// Shown when a recording holds nothing but zeros
const NO_INPUT: &str = "Nothing reached the app from the microphone. If it is not muted, allow Todo Biv in System Settings › Privacy & Security › Microphone.";

/// A request sent to the model and not answered yet
struct Pending {
    id: u64,
    request: String,
    started: Instant,
}

/// What came of the last request, or of the microphone
enum Reply {
    None,
    Answered { text: String, changes: usize, quick: Option<String>, elapsed: Option<Duration> },
    Failed(String),
    Notice(String),
}

pub struct RequestBar {
    state: Entity<TodoState>,
    model: Entity<ModelService>,
    speech: Entity<SpeechService>,
    pub input: Entity<InputState>,
    recorder: Option<Recorder>,
    transcribing: Option<u64>,
    pending: Option<Pending>,
    reply: Reply,
    /// A cancelled request, put back in the field at the next render
    restore_input: Option<String>,
    /// Redraws the level meter or the elapsed time
    ticker: Option<Task<()>>,
    _subscriptions: Vec<Subscription>,
}

impl RequestBar {
    pub fn new(
        state: Entity<TodoState>,
        model: Entity<ModelService>,
        speech: Entity<SpeechService>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let input = cx.new(|cx| InputState::new(window, cx).placeholder(PLACEHOLDER));
        let subscriptions = vec![
            cx.subscribe_in(&input, window, |this, _, event: &InputEvent, window, cx| {
                if let InputEvent::PressEnter { .. } = event {
                    if this.recorder.is_some() {
                        this.toggle_recording(window, cx);
                    } else {
                        let text = this.input.read(cx).value().to_string();
                        this.submit(text, window, cx);
                    }
                }
            }),
            cx.subscribe_in(&speech, window, |this, _, event: &SpeechEvent, window, cx| this.on_speech(event, window, cx)),
            cx.subscribe(&model, |this, _, event: &ModelEvent, cx| this.on_model(event, cx)),
            cx.observe(&model, |_, _, cx| cx.notify()),
            cx.observe(&speech, |_, _, cx| cx.notify()),
            cx.observe(&state, |this, state, cx| {
                // Microphone mode turned off while recording
                if !state.read(cx).settings.microphone {
                    this.discard_recording(cx);
                }
                cx.notify();
            }),
        ];
        Self {
            state,
            model,
            speech,
            input,
            recorder: None,
            transcribing: None,
            pending: None,
            reply: Reply::None,
            restore_input: None,
            ticker: None,
            _subscriptions: subscriptions,
        }
    }

    pub fn focus(&self, window: &mut Window, cx: &mut Context<Self>) {
        self.input.update(cx, |input, cx| input.focus(window, cx));
    }

    /// Sends `text` to the model, or reads it by the rules if the model is
    /// not loaded. While recording, stops and transcribes instead; nothing is
    /// sent while a transcript is awaited, or another request answered.
    pub fn submit(&mut self, text: String, window: &mut Window, cx: &mut Context<Self>) {
        if self.recorder.is_some() {
            self.toggle_recording(window, cx);
            return;
        }
        let request = text.trim().to_string();
        if request.is_empty() || self.pending.is_some() || self.transcribing.is_some() {
            return;
        }
        self.input.update(cx, |input, cx| input.set_value("", window, cx));
        let tasks = self.state.read(cx).store.tasks.clone();
        let sent = self.model.update(cx, |model, cx| model.interpret(request.clone(), tasks, cx));
        match sent {
            Some(id) => {
                self.pending = Some(Pending { id, request, started: Instant::now() });
                self.reply = Reply::None;
                self.ticker = Some(redraw_every(Duration::from_millis(250), cx));
            }
            None => {
                let note = if self.model.read(cx).is_loading() { "model still loading" } else { "model not ready" };
                self.quick_parse(&request, note.into(), cx);
            }
        }
        cx.notify();
    }

    fn quick_parse(&mut self, request: &str, why: String, cx: &mut Context<Self>) {
        let interpretation = heuristics::interpret(request, &self.state.read(cx).store.tasks, Local::now());
        self.apply(request, interpretation, Some(why), None, cx);
    }

    fn apply(&mut self, request: &str, interpretation: Interpretation, quick: Option<String>, elapsed: Option<Duration>, cx: &mut Context<Self>) {
        let touched = self.state.update(cx, |state, cx| state.apply_request(&interpretation.ops, request, cx));
        let changes = touched.len();
        let text = if !interpretation.ops.is_empty() && touched.is_empty() {
            // The list changed while the model was answering.
            "Nothing changed: the tasks it was about are gone.".to_string()
        } else if interpretation.reply.trim().is_empty() {
            match changes {
                0 => "Nothing to change.".to_string(),
                1 => "Done: 1 change.".to_string(),
                n => format!("Done: {n} changes."),
            }
        } else {
            interpretation.reply
        };
        self.reply = Reply::Answered { text, changes, quick, elapsed };
    }

    fn on_model(&mut self, event: &ModelEvent, cx: &mut Context<Self>) {
        let Some(pending) = &self.pending else { return };
        let (id, request) = (pending.id, pending.request.clone());
        match event {
            ModelEvent::Interpreted { id: answered, interpretation, elapsed } if *answered == id => {
                self.pending = None;
                self.apply(&request, interpretation.clone(), None, Some(*elapsed), cx);
            }
            ModelEvent::Failed { id: failed, error, .. } if *failed == id => {
                self.pending = None;
                if error == "cancelled" {
                    self.reply = Reply::Notice(format!("Cancelled “{request}”."));
                    self.restore_input = Some(request);
                } else {
                    self.quick_parse(&request, format!("the model failed: {error}"), cx);
                }
            }
            _ => return,
        }
        self.ticker = None;
        cx.notify();
    }

    /// Stops and drops the recording, if there is one
    pub fn discard_recording(&mut self, cx: &mut Context<Self>) {
        if let Some(recorder) = self.recorder.take() {
            drop(recorder.finish());
            self.reply = Reply::Notice("Recording discarded.".into());
            self.ticker = None;
            cx.notify();
        }
    }

    /// Discards the recording, or interrupts the request being answered and
    /// puts it back in the field; returns false if there is neither
    pub fn cancel(&mut self, cx: &mut Context<Self>) -> bool {
        if self.recorder.is_some() {
            self.discard_recording(cx);
        } else if let Some(pending) = self.pending.take() {
            // Whatever answer still comes is ignored.
            self.model.update(cx, |model, _| model.cancel(pending.id));
            self.reply = Reply::Notice(format!("Cancelled “{}”.", pending.request));
            self.restore_input = Some(pending.request);
        } else {
            return false;
        }
        self.ticker = None;
        cx.notify();
        true
    }

    /// Puts the list back as it was before the last request
    pub fn undo(&mut self, cx: &mut Context<Self>) -> bool {
        let label = self.state.read(cx).undo_label().map(str::to_string);
        let Some(label) = label else { return false };
        self.state.update(cx, |state, cx| state.undo(cx));
        self.reply = Reply::Notice(format!("Undid {label}."));
        cx.notify();
        true
    }

    /// Starts recording, or stops and transcribes what was recorded. A
    /// request being answered must be answered (or cancelled) first: its
    /// transcript could not be sent.
    pub fn toggle_recording(&mut self, _: &mut Window, cx: &mut Context<Self>) {
        if self.recorder.is_some() {
            self.stop_recording(cx);
        } else if self.pending.is_some() {
            self.reply = Reply::Notice("Wait for the answer to the request (or press Esc to cancel it), then speak.".into());
        } else if self.state.read(cx).settings.microphone && self.transcribing.is_none() {
            if !self.speech.read(cx).is_ready() {
                self.reply = Reply::Notice(speech_hint(&self.speech.read(cx).status).into());
            } else {
                let device = self.state.read(cx).settings.device().map(str::to_string);
                match Recorder::start(device.as_deref()) {
                    Ok(recorder) => {
                        self.recorder = Some(recorder);
                        self.reply = Reply::None;
                        self.ticker = Some(watch_recording(cx));
                    }
                    Err(error) => self.reply = Reply::Failed(format!("Could not record: {error:#}")),
                }
            }
        }
        cx.notify();
    }

    /// Stops recording and has what was recorded transcribed
    fn stop_recording(&mut self, cx: &mut Context<Self>) {
        let Some(recorder) = self.recorder.take() else { return };
        self.ticker = None;
        let recording = recorder.finish();
        if recording.seconds() < SHORTEST_RECORDING {
            self.reply = Reply::None;
        } else if recording.peak() == 0.0 {
            self.reply = Reply::Failed(NO_INPUT.into());
        } else {
            self.transcribing = Some(self.speech.update(cx, |speech, _| speech.transcribe(recording)));
            self.ticker = Some(redraw_every(Duration::from_millis(250), cx));
        }
        cx.notify();
    }

    fn on_speech(&mut self, event: &SpeechEvent, window: &mut Window, cx: &mut Context<Self>) {
        match event {
            SpeechEvent::Transcribed { id, transcript, .. } if Some(*id) == self.transcribing => {
                self.transcribing = None;
                self.ticker = None;
                let text = transcript.text.trim().to_string();
                if text.is_empty() {
                    self.reply = Reply::Notice("No speech heard.".into());
                } else if self.state.read(cx).settings.auto_submit {
                    self.input.update(cx, |input, cx| input.set_value(text.clone(), window, cx));
                    self.submit(text, window, cx);
                } else {
                    self.input.update(cx, |input, cx| input.set_value(text, window, cx));
                    self.reply = Reply::Notice("Heard. Press Enter to send it.".into());
                }
            }
            SpeechEvent::Failed { id, error } if Some(*id) == self.transcribing => {
                self.transcribing = None;
                self.ticker = None;
                self.reply = Reply::Failed(format!("Could not transcribe: {error}"));
            }
            _ => return,
        }
        cx.notify();
    }

    fn mic_button(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let status = self.speech.read(cx).status.clone();
        let ready = matches!(status, SpeechStatus::Ready { .. });
        let recording = self.recorder.is_some();
        let button = Button::new("record")
            .icon(if recording { IconName::Square } else { IconName::Mic })
            .rounded(ButtonRounded::Size(px(18.)))
            .disabled((!ready && !recording) || self.transcribing.is_some() || self.pending.is_some())
            .tooltip(if recording { "Stop (Enter)".to_string() } else if ready { "Speak a request (⌘⇧M)".to_string() } else { speech_hint(&status).to_string() })
            .on_click(cx.listener(|this, _, window, cx| this.toggle_recording(window, cx)));
        let button = if recording { button.danger() } else { button.ghost() };
        div().child(button)
    }

    fn render_status(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme().clone();
        let muted = theme.muted_foreground;
        let row = h_flex().gap_2().items_center().min_h(px(28.)).text_sm();
        if let Some(recorder) = &self.recorder {
            let level = recorder.level().clamp(0.0, 1.0).sqrt();
            let red = palette::danger(cx);
            return row
                .child(div().size(px(8.)).rounded_full().bg(red))
                .child(div().text_color(red).font_medium().child(format!("Recording {}", clock(recorder.duration()))))
                .child(div().w(px(160.)).child(ui::meter(level, red, cx)))
                .child(div().text_color(muted).child("Click the button or press Enter to stop, Esc to discard"))
                .into_any_element();
        }
        if self.transcribing.is_some() {
            return row.child(Spinner::new().small().color(palette::accent(cx))).child(div().text_color(muted).child("Transcribing…")).into_any_element();
        }
        if let Some(pending) = &self.pending {
            let partial = self.model.read(cx).partial.filter(|(id, _)| *id == pending.id).map(|(_, chars)| chars);
            let progress = match partial {
                Some(chars) => format!("Thinking… {chars} characters · {:.0} s", pending.started.elapsed().as_secs_f32()),
                None => format!("Thinking… {:.0} s", pending.started.elapsed().as_secs_f32()),
            };
            return row
                .child(Spinner::new().small().color(palette::accent(cx)))
                .child(div().font_medium().child(format!("“{}”", pending.request)))
                .child(div().text_color(muted).child(progress))
                .child(
                    Button::new("cancel")
                        .xsmall()
                        .ghost()
                        .label("Cancel")
                        .tooltip("Esc")
                        .on_click(cx.listener(|this, _, _, cx| {
                            this.cancel(cx);
                        })),
                )
                .into_any_element();
        }
        let undo = self.state.read(cx).undo_label().map(|label| {
            Button::new("undo")
                .xsmall()
                .ghost()
                .icon(IconName::Undo2)
                .label("Undo")
                .tooltip(format!("Undo {label} (⌘Z)"))
                .on_click(cx.listener(|this, _, _, cx| {
                    this.undo(cx);
                }))
        });
        let row = match &self.reply {
            Reply::None => {
                let hint = (self.state.read(cx).settings.microphone && !self.speech.read(cx).is_ready())
                    .then(|| speech_hint(&self.speech.read(cx).status));
                row.child(div().text_color(muted).child(hint.unwrap_or("Add, finish, move or reprioritize tasks in your own words.")))
            }
            Reply::Answered { text, changes, quick, elapsed } => {
                let (icon, color) = if quick.is_some() { (IconName::Zap, palette::warning(cx)) } else { (IconName::Sparkles, palette::accent(cx)) };
                let detail = match (quick, elapsed) {
                    (Some(why), _) => format!("(quick parse — {why})"),
                    (None, Some(elapsed)) => format!("{} · {:.1} s", plural(*changes, "change"), elapsed.as_secs_f32()),
                    (None, None) => plural(*changes, "change"),
                };
                row.child(Icon::new(icon).size(px(14.)).text_color(color))
                    .child(div().min_w_0().child(text.clone()))
                    .child(div().text_color(muted).text_xs().child(detail))
            }
            Reply::Failed(error) => row
                .child(Icon::new(IconName::CircleAlert).size(px(14.)).text_color(palette::danger(cx)))
                .child(div().text_color(palette::danger(cx)).child(error.clone())),
            Reply::Notice(text) => row.child(div().text_color(muted).child(text.clone())),
        };
        row.child(div().flex_1()).children(undo).into_any_element()
    }
}

/// Redraws the bar for the level meter while recording, and stops a
/// recording that went on for too long
fn watch_recording(cx: &mut Context<RequestBar>) -> Task<()> {
    cx.spawn(async move |bar, cx| {
        loop {
            cx.background_executor().timer(Duration::from_millis(50)).await;
            let watched = bar.update(cx, |bar, cx| {
                if bar.recorder.as_ref().is_some_and(|r| r.duration() >= LONGEST_RECORDING) {
                    bar.stop_recording(cx);
                }
                cx.notify();
            });
            if watched.is_err() {
                break;
            }
        }
    })
}

/// Redraws `view` periodically, for a timer or a meter
fn redraw_every<V: 'static>(period: Duration, cx: &mut Context<V>) -> Task<()> {
    cx.spawn(async move |view, cx| {
        loop {
            cx.background_executor().timer(period).await;
            if view.update(cx, |_, cx| cx.notify()).is_err() {
                break;
            }
        }
    })
}

/// What speech recognition is doing, for a hint
pub fn speech_hint(status: &SpeechStatus) -> &'static str {
    match status {
        SpeechStatus::Off => "Speech recognition is off.",
        SpeechStatus::Downloading { .. } => "Downloading the speech model…",
        SpeechStatus::Loading => "Loading the speech model…",
        SpeechStatus::Ready { .. } => "Speech recognition is ready.",
        SpeechStatus::Failed(_) => "Speech recognition failed; see Settings.",
    }
}

fn clock(seconds: f32) -> String {
    let seconds = seconds.max(0.0) as u32;
    format!("{}:{:02}", seconds / 60, seconds % 60)
}

fn plural(n: usize, word: &str) -> String {
    if n == 1 { format!("1 {word}") } else { format!("{n} {word}s") }
}

impl Render for RequestBar {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        if let Some(request) = self.restore_input.take() {
            self.input.update(cx, |input, cx| input.set_value(request, window, cx));
        }
        let theme = cx.theme().clone();
        let microphone = self.state.read(cx).settings.microphone;
        let busy = self.pending.is_some();
        let send = Button::new("send")
            .primary()
            .icon(IconName::ArrowUp)
            .rounded(ButtonRounded::Size(px(18.)))
            .disabled(busy || self.transcribing.is_some())
            .tooltip("Send (Enter)")
            .on_click(cx.listener(|this, _, window, cx| {
                let text = this.input.read(cx).value().to_string();
                this.submit(text, window, cx);
            }));
        v_flex()
            .gap_1()
            .child(
                h_flex()
                    .gap_1()
                    .items_center()
                    .pl_3()
                    .pr_1p5()
                    .py_1p5()
                    .rounded_xl()
                    .border_1()
                    .border_color(theme.border)
                    .bg(theme.background)
                    .shadow_sm()
                    .child(Icon::new(IconName::Sparkles).size(px(18.)).text_color(palette::accent(cx)))
                    .child(
                        div()
                            .key_context("RequestInput")
                            .flex_1()
                            .min_w_0()
                            .child(Input::new(&self.input).appearance(false).large().disabled(busy)),
                    )
                    .when(microphone, |row| row.child(self.mic_button(cx)))
                    .child(send),
            )
            .child(div().px_2().child(self.render_status(cx)))
    }
}

/// The request shown in the screenshots' reply line
#[cfg(test)]
impl RequestBar {
    pub fn show_answer(&mut self, text: &str, changes: usize, cx: &mut Context<Self>) {
        self.reply = Reply::Answered { text: text.into(), changes, quick: None, elapsed: Some(Duration::from_millis(2300)) };
        cx.notify();
    }

    pub fn clear_reply(&mut self, cx: &mut Context<Self>) {
        self.reply = Reply::None;
        cx.notify();
    }
}
