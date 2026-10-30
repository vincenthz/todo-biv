//! Preferences: the language model, the periodic review, and the microphone.

use assistant::ModelStatus;
use gpui::{App, Context, Entity, SharedString, Subscription, Window, div, prelude::*, px};
use gpui_component::{
    ActiveTheme as _, Disableable as _, Sizable as _, StyledExt as _,
    button::{Button, ButtonVariants as _},
    h_flex,
    input::{Input, InputEvent, InputState},
    menu::{DropdownMenu as _, PopupMenuItem},
    progress::Progress,
    switch::Switch,
    v_flex,
};
use gpui_kit_assets::IconName;
use speech::whisper::hub;

use crate::{
    palette,
    services::{
        model::ModelService,
        speech::{SpeechService, SpeechStatus},
    },
    settings::Settings,
    state::TodoState,
    ui,
};

/// Review intervals offered, in minutes
const INTERVALS: [u32; 5] = [5, 10, 15, 30, 60];

pub struct SettingsPage {
    state: Entity<TodoState>,
    model: Entity<ModelService>,
    speech: Entity<SpeechService>,
    model_input: Entity<InputState>,
    /// Microphones, listed when the page opens
    devices: Result<Vec<String>, String>,
    _subscriptions: Vec<Subscription>,
}

impl SettingsPage {
    pub fn new(
        state: Entity<TodoState>,
        model: Entity<ModelService>,
        speech: Entity<SpeechService>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let current = state.read(cx).settings.model.clone();
        let model_input = cx.new(|cx| InputState::new(window, cx).placeholder(assistant::DEFAULT_MODEL).default_value(current));
        let subscriptions = vec![
            cx.observe(&state, |_, _, cx| cx.notify()),
            cx.observe(&model, |_, _, cx| cx.notify()),
            cx.observe(&speech, |_, _, cx| cx.notify()),
            cx.subscribe(&model_input, |this, _, event: &InputEvent, cx| {
                if let InputEvent::PressEnter { .. } = event {
                    this.apply_model(cx);
                }
            }),
        ];
        let devices = if state.read(cx).settings.microphone { list_devices() } else { Ok(Vec::new()) };
        Self { state, model, speech, model_input, devices, _subscriptions: subscriptions }
    }

    /// Saves a model setting, reloading the model with it unless no model
    /// was asked for yet
    fn change_model(&mut self, edit: impl FnOnce(&mut Settings), cx: &mut Context<Self>) {
        self.state.update(cx, |state, cx| state.update_settings(edit, cx));
        if !matches!(self.model.read(cx).status, ModelStatus::Idle) {
            self.load(cx);
        }
    }

    fn load(&mut self, cx: &mut Context<Self>) {
        let config = self.state.read(cx).settings.model_config();
        self.model.update(cx, |model, cx| model.load(config, cx));
    }

    fn apply_model(&mut self, cx: &mut Context<Self>) {
        let typed = self.model_input.read(cx).value().trim().to_string();
        let model = if typed.is_empty() { assistant::DEFAULT_MODEL.to_string() } else { typed };
        self.state.update(cx, |state, cx| state.update_settings(|s| s.model = model, cx));
        self.load(cx);
    }

    fn set_microphone(&mut self, on: bool, cx: &mut Context<Self>) {
        self.state.update(cx, |state, cx| state.update_settings(|s| s.microphone = on, cx));
        self.speech.update(cx, |speech, cx| speech.set_enabled(on, cx));
        if on {
            self.devices = list_devices();
        }
    }

    fn model_card(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme().clone();
        let settings = self.state.read(cx).settings.clone();
        let status = self.model.read(cx).status.clone();
        let status_row = match &status {
            ModelStatus::Idle => h_flex().gap_2().items_center().child(ui::muted("Not loaded.", cx)).child(
                Button::new("load")
                    .small()
                    .primary()
                    .icon(IconName::Play)
                    .label("Load now")
                    .on_click(cx.listener(|this, _, _, cx| this.load(cx))),
            ),
            ModelStatus::Loading => h_flex().gap_2().items_center().child(ui::muted("Loading… (a large model takes a moment to map and warm up)", cx)),
            ModelStatus::Ready { description, path, load_time } => v_flex()
                .gap_0p5()
                .child(div().text_sm().child(format!("Ready: {description}, loaded in {:.1} s", load_time.as_secs_f32())))
                .child(div().text_xs().text_color(theme.muted_foreground).child(path.display().to_string())),
            ModelStatus::Failed(error) => h_flex()
                .gap_2()
                .items_center()
                .child(div().flex_1().text_sm().text_color(palette::danger(cx)).child(format!("Failed: {error}")))
                .child(Button::new("retry").small().outline().icon(IconName::RefreshCw).label("Retry").on_click(cx.listener(|this, _, _, cx| this.load(cx)))),
        };
        let intervals = h_flex().gap_1().children(INTERVALS.iter().enumerate().map(|(i, &minutes)| {
            let button = Button::new(("interval", i)).small().label(format!("{minutes} min")).on_click(
                cx.listener(move |this, _, _, cx| this.state.update(cx, |state, cx| state.update_settings(|s| s.review_minutes = minutes, cx))),
            );
            if settings.review_minutes == minutes { button.primary() } else { button.ghost() }
        }));

        ui::colored_card(palette::accent(cx), cx)
            .gap_3()
            .child(ui::section_title("Language model"))
            .child(ui::muted(
                "biv runs the model on this computer: requests are read, classified and prioritized without leaving it.",
                cx,
            ))
            .child(
                v_flex().gap_1().child(ui::caption("MODEL", cx)).child(
                    h_flex()
                        .gap_2()
                        .child(div().flex_1().child(Input::new(&self.model_input)))
                        .child(Button::new("apply-model").outline().label("Apply").on_click(cx.listener(|this, _, _, cx| this.apply_model(cx)))),
                ),
            )
            .child(div().text_xs().text_color(theme.muted_foreground).child("The full path of a GGUF file (~/ for the home folder), or hf.co/<owner>/<repo>:<quant> from the Hugging Face cache."))
            .child(status_row)
            .child(switch_row("cpu", "Run on the CPU", "Slower; for when Metal is unavailable.", settings.cpu, cx, |this, on, cx| {
                this.change_model(|s| s.cpu = on, cx)
            }))
            .child(switch_row("thinking", "Thinking", "Let the model reason before answering: slower, sometimes better.", settings.thinking, cx, |this, on, cx| {
                this.change_model(|s| s.thinking = on, cx)
            }))
            .child(switch_row("startup", "Load at startup", "Otherwise requests are read by simple rules until you load it.", settings.load_at_startup, cx, |this, on, cx| {
                this.state.update(cx, |state, cx| state.update_settings(|s| s.load_at_startup = on, cx))
            }))
            .child(v_flex().gap_1().child(ui::caption("REVIEW THE LIST EVERY", cx)).child(intervals).child(
                div().text_xs().text_color(theme.muted_foreground).child("And about 20 seconds after it changes, while the model is loaded."),
            ))
    }

    fn microphone_card(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme().clone();
        let settings = self.state.read(cx).settings.clone();
        let speech = self.speech.read(cx);
        let (model, status) = (speech.model, speech.status.clone());
        let busy = speech.is_busy();

        let status_row = match &status {
            SpeechStatus::Off => ui::muted("Off. The speech model loads when the microphone mode is turned on.", cx).into_any_element(),
            SpeechStatus::Downloading { file, percent } => h_flex()
                .gap_2()
                .items_center()
                .child(ui::muted(format!("Downloading {file} {percent:.0}%"), cx))
                .child(div().w(px(200.)).child(Progress::new("download").value(*percent)))
                .into_any_element(),
            SpeechStatus::Loading => ui::muted("Loading the speech model…", cx).into_any_element(),
            SpeechStatus::Ready { device } => ui::muted(format!("Ready on the {device}."), cx).into_any_element(),
            SpeechStatus::Failed(error) => h_flex()
                .gap_2()
                .items_center()
                .child(div().flex_1().text_sm().text_color(palette::danger(cx)).child(format!("Failed: {error}")))
                .child(
                    Button::new("speech-retry")
                        .small()
                        .outline()
                        .label("Retry")
                        .on_click(cx.listener(|this, _, _, cx| this.speech.update(cx, |speech, cx| speech.load(cx)))),
                )
                .into_any_element(),
        };

        let checkpoints = h_flex().gap_1().flex_wrap().children(hub::CHECKPOINTS.iter().enumerate().map(|(i, checkpoint)| {
            let downloaded = if hub::cached(checkpoint).is_some() { "" } else { " ↓" };
            let label = format!("{} · {} MB{downloaded}", checkpoint.label, checkpoint.download_mb);
            let button = Button::new(("whisper", i)).small().label(label).disabled(busy).on_click(cx.listener(move |this, _, _, cx| {
                this.state.update(cx, |state, cx| state.update_settings(|s| s.whisper = checkpoint.name.into(), cx));
                this.speech.update(cx, |speech, cx| speech.select(checkpoint, cx));
            }));
            if checkpoint == model { button.primary() } else { button.ghost() }
        }));

        let device_label: SharedString = if settings.input_device.is_empty() { "System default".into() } else { settings.input_device.clone().into() };
        let devices = self.devices.clone().unwrap_or_default();
        let current = settings.input_device.clone();
        let page = cx.entity().downgrade();
        let device_menu = Button::new("device").small().outline().icon(IconName::Mic).label(device_label).dropdown_caret(true).dropdown_menu(move |mut menu, _, _| {
            for name in std::iter::once(String::new()).chain(devices.iter().cloned()) {
                let label = if name.is_empty() { "System default".to_string() } else { name.clone() };
                let page = page.clone();
                let checked = name == current;
                menu = menu.item(PopupMenuItem::new(label).checked(checked).on_click(move |_, _, cx| {
                    let name = name.clone();
                    let _ = page.update(cx, |page, cx| page.state.update(cx, |state, cx| state.update_settings(|s| s.input_device = name, cx)));
                }));
            }
            menu
        });

        ui::colored_card(palette::vivid(175.0, cx), cx)
            .gap_3()
            .child(ui::section_title("Microphone"))
            .child(ui::muted("Speak a request instead of typing it. Whisper transcribes it on this computer.", cx))
            .child(switch_row("microphone", "Microphone mode", "Shows a record button next to the request field (⌘⇧M).", settings.microphone, cx, |this, on, cx| {
                this.set_microphone(on, cx)
            }))
            .when(settings.microphone, |card| {
                card.child(v_flex().gap_1().child(ui::caption("SPEECH MODEL", cx)).child(checkpoints))
                    .child(status_row)
                    .child(
                        v_flex()
                            .gap_1()
                            .child(ui::caption("INPUT DEVICE", cx))
                            .child(h_flex().gap_2().items_center().child(device_menu).child(
                                Button::new("refresh-devices").small().ghost().icon(IconName::RefreshCw).tooltip("List the microphones again").on_click(cx.listener(|this, _, _, cx| {
                                    this.devices = list_devices();
                                    cx.notify();
                                })),
                            ))
                            .when_some(self.devices.as_ref().err(), |col, error| col.child(div().text_xs().text_color(palette::danger(cx)).child(error.clone()))),
                    )
                    .child(switch_row("auto-submit", "Send what is heard", "Otherwise the transcript waits in the field for Enter.", settings.auto_submit, cx, |this, on, cx| {
                        this.state.update(cx, |state, cx| state.update_settings(|s| s.auto_submit = on, cx))
                    }))
            })
            .when(!settings.microphone, |card| card.child(div().text_xs().text_color(theme.muted_foreground).child(format!("Uses about {} MB of memory when on.", model.download_mb))))
    }
}

fn list_devices() -> Result<Vec<String>, String> {
    speech::audio::capture::input_devices().map_err(|e| format!("Could not list the microphones: {e:#}"))
}

/// A switch with a label and a line of explanation
fn switch_row(
    id: &'static str,
    label: &'static str,
    description: &'static str,
    checked: bool,
    cx: &mut Context<SettingsPage>,
    on_change: impl Fn(&mut SettingsPage, bool, &mut Context<SettingsPage>) + 'static,
) -> impl IntoElement {
    let page = cx.entity().downgrade();
    h_flex()
        .gap_3()
        .items_center()
        .child(v_flex().flex_1().child(div().text_sm().font_medium().child(label)).child(div().text_xs().text_color(cx.theme().muted_foreground).child(description)))
        .child(Switch::new(id).checked(checked).on_click(move |on, _, cx: &mut App| {
            let on = *on;
            let _ = page.update(cx, |page, cx| on_change(page, on, cx));
        }))
}

const SHORTCUTS: [(&str, &str); 7] = [
    ("⌘K / ⌘L", "Focus the request field"),
    ("Enter", "Send the request (or stop recording)"),
    ("⌘⇧M", "Start or stop recording"),
    ("Esc", "Cancel the request being answered"),
    ("⌘Z", "Undo the last request"),
    ("⌘R", "Review the list now"),
    ("⌘,", "Settings"),
];

impl Render for SettingsPage {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme().clone();
        let data_dir = self.state.read(cx).data_dir().map(|d| d.display().to_string());
        let problems: Vec<String> = self.state.read(cx).problems().into_iter().map(str::to_string).collect();
        let shortcuts = ui::card(cx).gap_1().child(ui::section_title("Keyboard")).children(SHORTCUTS.map(|(keys, what)| {
            h_flex()
                .gap_3()
                .text_sm()
                .child(div().w(px(110.)).font_semibold().text_color(theme.muted_foreground).child(keys))
                .child(what)
        }));
        div().id("settings").size_full().overflow_y_scroll().child(
            v_flex()
                .w_full()
                .max_w(px(760.))
                .mx_auto()
                .px_8()
                .py_6()
                .gap_5()
                .child(div().text_2xl().font_bold().child("Settings"))
                .child(self.model_card(cx))
                .child(self.microphone_card(cx))
                .child(shortcuts)
                .when_some(data_dir, |col, dir| col.child(ui::muted(format!("Tasks and settings are saved in {dir}"), cx)))
                .children(problems.into_iter().map(|problem| div().text_sm().text_color(palette::danger(cx)).child(problem))),
        )
    }
}
