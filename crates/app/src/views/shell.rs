//! The window's frame: a title bar with the model's status, a sidebar of
//! views and categories, the request bar over the task list, and the plan;
//! or the settings in place of the list and plan.

use chrono::Local;
use gpui::{App, Context, Entity, FocusHandle, Focusable as _, Hsla, KeyBinding, Subscription, Window, actions, div, prelude::*, px};
use gpui_component::{
    ActiveTheme as _, Disableable as _, Icon, Sizable as _, StyledExt as _, TitleBar, WindowExt as _,
    button::{Button, ButtonVariants as _},
    h_flex,
    notification::Notification,
    sidebar::{Sidebar, SidebarGroup, SidebarMenu, SidebarMenuItem},
    spinner::Spinner,
    v_flex,
};
use gpui_kit_assets::IconName;
use assistant::ModelStatus;

use crate::{
    lists::{self, View},
    palette,
    services::{model::ModelService, reviewer::Reviewer, speech::SpeechService},
    state::TodoState,
    views::{plan_panel::PlanPanel, request_bar::RequestBar, settings::SettingsPage, task_list::TaskList},
};

actions!(todo_biv, [Quit, FocusRequest, ToggleRecording, CancelRequest, UndoRequest, ToggleSettings, ReviewNow]);

const CONTEXT: &str = "TodoApp";

pub fn bind_keys(cx: &mut App) {
    cx.bind_keys([
        KeyBinding::new("cmd-q", Quit, None),
        KeyBinding::new("cmd-k", FocusRequest, Some(CONTEXT)),
        KeyBinding::new("cmd-l", FocusRequest, Some(CONTEXT)),
        KeyBinding::new("cmd-shift-m", ToggleRecording, Some(CONTEXT)),
        KeyBinding::new("escape", CancelRequest, Some(CONTEXT)),
        KeyBinding::new("cmd-z", UndoRequest, Some(CONTEXT)),
        // Registered after the text field's own, so it wins there; it undoes
        // typing while there is text, the last request otherwise.
        KeyBinding::new("cmd-z", UndoRequest, Some("RequestInput > Input")),
        KeyBinding::new("cmd-,", ToggleSettings, Some(CONTEXT)),
        KeyBinding::new("cmd-r", ReviewNow, Some(CONTEXT)),
    ]);
    cx.on_action(|_: &Quit, cx| cx.quit());
}

pub struct Shell {
    state: Entity<TodoState>,
    model: Entity<ModelService>,
    speech: Entity<SpeechService>,
    reviewer: Entity<Reviewer>,
    pub request_bar: Entity<RequestBar>,
    task_list: Entity<TaskList>,
    plan_panel: Entity<PlanPanel>,
    settings: Option<Entity<SettingsPage>>,
    focus: FocusHandle,
    _subscriptions: Vec<Subscription>,
}

impl Shell {
    pub fn new(
        state: Entity<TodoState>,
        model: Entity<ModelService>,
        speech: Entity<SpeechService>,
        reviewer: Entity<Reviewer>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let request_bar = cx.new(|cx| RequestBar::new(state.clone(), model.clone(), speech.clone(), window, cx));
        let task_list = cx.new(|cx| TaskList::new(state.clone(), cx));
        let plan_panel = cx.new(|cx| PlanPanel::new(state.clone(), model.clone(), reviewer.clone(), cx));
        let subscriptions = vec![
            cx.observe(&state, |_, _, cx| cx.notify()),
            cx.observe(&model, |_, _, cx| cx.notify()),
            cx.observe(&reviewer, |_, _, cx| cx.notify()),
            // When the focused element goes (the editor closed while one of its
            // fields had focus), keys would reach no view and the shortcuts
            // would stop working: focus goes back where it belongs.
            cx.on_focus_lost(window, |this, window, cx| this.restore_focus(window, cx)),
        ];
        request_bar.update(cx, |bar, cx| bar.focus(window, cx));
        Self {
            state,
            model,
            speech,
            reviewer,
            request_bar,
            task_list,
            plan_panel,
            settings: None,
            focus: cx.focus_handle(),
            _subscriptions: subscriptions,
        }
    }

    fn restore_focus(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.settings.is_some() {
            self.focus.focus(window, cx);
        } else {
            self.request_bar.update(cx, |bar, cx| bar.focus(window, cx));
        }
    }

    pub fn open_settings(&mut self, open: bool, window: &mut Window, cx: &mut Context<Self>) {
        if open && self.settings.is_none() {
            // A recording would go on unseen, and microphone mode may be turned off there.
            self.request_bar.update(cx, |bar, cx| bar.discard_recording(cx));
            let (state, model, speech) = (self.state.clone(), self.model.clone(), self.speech.clone());
            self.settings = Some(cx.new(|cx| SettingsPage::new(state, model, speech, window, cx)));
            self.focus.focus(window, cx);
        } else if !open && self.settings.is_some() {
            self.settings = None;
            self.request_bar.update(cx, |bar, cx| bar.focus(window, cx));
        }
        cx.notify();
    }

    #[cfg(test)]
    pub fn state(&self) -> Entity<TodoState> {
        self.state.clone()
    }

    #[cfg(test)]
    pub fn model(&self) -> Entity<ModelService> {
        self.model.clone()
    }

    fn select(&mut self, view: View, window: &mut Window, cx: &mut Context<Self>) {
        self.state.update(cx, |state, cx| state.select_view(view, cx));
        self.open_settings(false, window, cx);
    }

    fn focus_request(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.open_settings(false, window, cx);
        self.request_bar.update(cx, |bar, cx| bar.focus(window, cx));
    }

    fn toggle_recording(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if !self.state.read(cx).settings.microphone {
            return;
        }
        self.focus_request(window, cx);
        self.request_bar.update(cx, |bar, cx| bar.toggle_recording(window, cx));
    }

    fn cancel(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.request_bar.update(cx, |bar, cx| bar.cancel(cx)) {
            return;
        }
        if self.settings.is_some() {
            self.open_settings(false, window, cx);
        } else if let Some(id) = self.state.read(cx).expanded {
            self.state.update(cx, |state, cx| state.toggle_expanded(id, cx));
        } else {
            cx.propagate();
        }
    }

    fn undo(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let input = self.request_bar.read(cx).input.clone();
        let typing = input.read(cx).focus_handle(cx).is_focused(window) && !input.read(cx).value().is_empty();
        if typing {
            window.dispatch_action(Box::new(gpui_component::input::Undo), cx);
        } else if !self.request_bar.update(cx, |bar, cx| bar.undo(cx)) {
            cx.propagate();
        }
    }

    fn review_now(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.reviewer.update(cx, |reviewer, cx| reviewer.review(cx));
        if !self.reviewer.read(cx).is_reviewing() {
            window.push_notification(Notification::warning("The model is busy or not loaded; the plan is ranked by rules meanwhile."), cx);
        }
    }

    /// The title bar's model status: a colored dot and a word; clicking it
    /// explains, loads, or shows the error
    fn model_status(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme().clone();
        let model = self.model.read(cx);
        let label = model.status_label();
        let (color, loading) = match &model.status {
            ModelStatus::Idle => (theme.muted_foreground, false),
            ModelStatus::Loading => (palette::warning(cx), true),
            ModelStatus::Ready { .. } => (palette::success(cx), false),
            ModelStatus::Failed(_) => (palette::danger(cx), false),
        };
        let indicator = if loading {
            Spinner::new().xsmall().color(color).into_any_element()
        } else {
            div().size(px(7.)).rounded_full().bg(color).into_any_element()
        };
        let tooltip = match &model.status {
            ModelStatus::Idle => "Click to load the model".to_string(),
            ModelStatus::Loading => "The model is loading; requests are read by rules meanwhile".to_string(),
            ModelStatus::Ready { description, .. } => description.clone(),
            ModelStatus::Failed(_) => "Click to see why".to_string(),
        };
        h_flex()
            .id("model-status")
            .gap_1p5()
            .items_center()
            .px_2p5()
            .py_0p5()
            .rounded_full()
            .border_1()
            .border_color(color.opacity(0.35))
            .bg(palette::tint(color, cx))
            .text_xs()
            .font_medium()
            .cursor_pointer()
            .tooltip(move |window, cx| gpui_component::tooltip::Tooltip::new(tooltip.clone()).build(window, cx))
            .child(indicator)
            .child(label)
            .on_click(cx.listener(|this, _, window, cx| {
                let status = this.model.read(cx).status.clone();
                match status {
                    ModelStatus::Failed(error) => {
                        window.push_notification(Notification::error(error).title("The model could not load").autohide(false), cx)
                    }
                    ModelStatus::Idle => {
                        let config = this.state.read(cx).settings.model_config();
                        this.model.update(cx, |model, cx| model.load(config, cx));
                    }
                    ModelStatus::Ready { description, load_time, .. } => window.push_notification(
                        Notification::info(format!("{description}, loaded in {:.1} s.", load_time.as_secs_f32())).title("The model is ready"),
                        cx,
                    ),
                    ModelStatus::Loading => {}
                }
            }))
    }

    fn title_bar(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let ready = self.model.read(cx).is_ready();
        let reviewing = self.reviewer.read(cx).is_reviewing();
        let settings_open = self.settings.is_some();
        TitleBar::new()
            .child(
                h_flex()
                    .gap_2()
                    .items_center()
                    .child(Icon::new(IconName::ListTodo).size(px(16.)).text_color(palette::accent(cx)))
                    .child(div().text_sm().font_semibold().child("Todo Biv")),
            )
            .child(
                h_flex()
                    .gap_1()
                    .items_center()
                    .pr_3()
                    .child(self.model_status(cx))
                    .child(
                        Button::new("review-now")
                            .xsmall()
                            .ghost()
                            .icon(IconName::RefreshCw)
                            .label(if reviewing { "Reviewing…" } else { "Review now" })
                            .loading(reviewing)
                            .disabled(!ready || reviewing)
                            .tooltip("Ask the model what to do next")
                            .on_click(cx.listener(|this, _, window, cx| this.review_now(window, cx))),
                    )
                    .child(
                        Button::new("settings")
                            .xsmall()
                            .ghost()
                            .icon(IconName::Settings)
                            .toggled(settings_open)
                            .tooltip("Settings (⌘,)")
                            .on_click(cx.listener(move |this, _, window, cx| this.open_settings(!settings_open, window, cx))),
                    ),
            )
    }

    fn view_item(&self, view: View, icon: IconName, color: Hsla, count: usize, alert: bool, cx: &mut Context<Self>) -> SidebarMenuItem {
        let active = self.settings.is_none() && self.state.read(cx).view == view;
        let muted = cx.theme().muted_foreground;
        let count_color = if alert { palette::danger(cx) } else { muted };
        let shell = cx.entity().downgrade();
        SidebarMenuItem::new(view.label().to_string())
            .icon(Icon::new(icon).text_color(color))
            .active(active)
            .suffix(move |_, _| div().text_xs().font_medium().text_color(count_color).when(count > 0, |d| d.child(count.to_string())))
            .on_click(move |_, window, cx| {
                let view = view.clone();
                let _ = shell.update(cx, |shell, cx| shell.select(view, window, cx));
            })
    }

    fn sidebar(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let now = Local::now();
        let today = now.date_naive();
        let (counts, overdue, categories) = {
            let state = self.state.read(cx);
            let plan = state.plan(now);
            let tasks = &state.store.tasks;
            let counts = View::SMART.map(|view| lists::count(&view, tasks, &plan, now));
            let overdue = tasks.iter().any(|t| t.is_open() && t.due.is_some_and(|d| d < today));
            let mut categories = lists::categories(tasks);
            // The selected category stays listed when its last task is done.
            if let View::Category(name) = &state.view
                && !categories.iter().any(|(c, _)| c.eq_ignore_ascii_case(name))
            {
                categories.push((name.clone(), 0));
            }
            (counts, overdue, categories)
        };
        let accent = palette::accent(cx);
        let icons = [
            (IconName::Sparkles, accent),
            (IconName::Sun, palette::warning(cx)),
            (IconName::CalendarDays, palette::vivid(200.0, cx)),
            (IconName::ListTodo, palette::vivid(270.0, cx)),
            (IconName::CircleCheckBig, palette::success(cx)),
        ];
        let smart: Vec<SidebarMenuItem> = View::SMART
            .into_iter()
            .zip(icons)
            .zip(counts)
            .map(|((view, (icon, color)), count)| {
                let alert = view == View::Today && overdue;
                self.view_item(view, icon, color, count, alert, cx)
            })
            .collect();
        let categories: Vec<SidebarMenuItem> = categories
            .into_iter()
            .map(|(name, count)| {
                let color = palette::category(&name, cx);
                self.view_item(View::Category(name.clone()), category_icon(&name), color, count, false, cx)
            })
            .collect();
        let has_categories = !categories.is_empty();
        Sidebar::new("sidebar")
            .w(px(224.))
            .child(SidebarGroup::new("Views").child(SidebarMenu::new().children(smart)))
            .when(has_categories, |sidebar| sidebar.child(SidebarGroup::new("Categories").child(SidebarMenu::new().children(categories))))
    }
}

impl Shell {
    /// Failures to save, and what could not be read at startup (which can be
    /// dismissed), over the list
    fn problems(&self, cx: &mut Context<Self>) -> Option<impl IntoElement + use<>> {
        let state = self.state.read(cx);
        let problems: Vec<String> = state.problems().into_iter().map(str::to_string).collect();
        let dismissable = state.has_load_problems();
        if problems.is_empty() {
            return None;
        }
        let danger = palette::danger(cx);
        Some(
            h_flex()
                .gap_2()
                .items_start()
                .px_3()
                .py_1()
                .rounded_md()
                .text_sm()
                .text_color(danger)
                .bg(palette::tint(danger, cx))
                .child(v_flex().flex_1().min_w_0().gap_0p5().children(problems.into_iter().map(|p| div().child(p))))
                .when(dismissable, |row| {
                    row.child(Button::new("dismiss-problems").xsmall().ghost().icon(IconName::X).tooltip("Dismiss").on_click(cx.listener(
                        |this, _, _, cx| this.state.update(cx, |state, cx| state.dismiss_load_problems(cx)),
                    )))
                }),
        )
    }
}

pub fn category_icon(name: &str) -> IconName {
    match name.to_ascii_lowercase().as_str() {
        "work" => IconName::Briefcase,
        "personal" => IconName::User,
        "home" => IconName::House,
        "health" => IconName::HeartPulse,
        "finance" => IconName::Wallet,
        "errands" => IconName::ShoppingCart,
        "learning" => IconName::GraduationCap,
        "social" => IconName::Users,
        _ => IconName::Tag,
    }
}

impl Render for Shell {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme().clone();
        let problems = self.problems(cx);
        let main = match &self.settings {
            Some(settings) => div().flex_1().min_w_0().h_full().child(settings.clone()).into_any_element(),
            None => h_flex()
                .flex_1()
                .min_w_0()
                .h_full()
                .child(
                    v_flex()
                        .flex_1()
                        .min_w_0()
                        .h_full()
                        .child(div().px_6().pt_5().pb_1().child(self.request_bar.clone()))
                        .when_some(problems, |col, problems| col.child(div().mx_6().child(problems)))
                        .child(div().flex_1().min_h_0().child(self.task_list.clone())),
                )
                .child(div().w(px(330.)).flex_none().h_full().border_l_1().border_color(theme.border).bg(theme.sidebar).child(self.plan_panel.clone()))
                .into_any_element(),
        };
        v_flex()
            .size_full()
            .bg(theme.background)
            .text_color(theme.foreground)
            .key_context(CONTEXT)
            .track_focus(&self.focus)
            .on_action(cx.listener(|this, _: &FocusRequest, window, cx| this.focus_request(window, cx)))
            .on_action(cx.listener(|this, _: &ToggleRecording, window, cx| this.toggle_recording(window, cx)))
            .on_action(cx.listener(|this, _: &CancelRequest, window, cx| this.cancel(window, cx)))
            .on_action(cx.listener(|this, _: &UndoRequest, window, cx| this.undo(window, cx)))
            .on_action(cx.listener(|this, _: &ReviewNow, window, cx| this.review_now(window, cx)))
            .on_action(cx.listener(|this, _: &ToggleSettings, window, cx| {
                let open = this.settings.is_none();
                this.open_settings(open, window, cx)
            }))
            .child(self.title_bar(cx))
            .child(h_flex().flex_1().min_h_0().child(self.sidebar(cx)).child(main))
    }
}
