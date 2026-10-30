//! The plan: the task to take up next, those after it with the reasons for
//! their order, and the priority changes the model's review suggests.

use chrono::Local;
use gpui::{Context, Entity, Subscription, Window, div, prelude::*, px};
use gpui_component::{
    ActiveTheme as _, Disableable as _, Icon, Sizable as _, StyledExt as _,
    button::{Button, ButtonVariants as _},
    h_flex,
    spinner::Spinner,
    v_flex,
};
use gpui_kit_assets::IconName;
use todo_core::{Priority, Status, TaskId, TaskPatch};

use crate::{
    lists, palette,
    services::{model::ModelService, reviewer::Reviewer},
    state::TodoState,
    ui,
    views::task_row,
};

/// How many tasks the plan lists after the first
const LISTED: usize = 7;

pub struct PlanPanel {
    state: Entity<TodoState>,
    model: Entity<ModelService>,
    reviewer: Entity<Reviewer>,
    _subscriptions: Vec<Subscription>,
}

impl PlanPanel {
    pub fn new(state: Entity<TodoState>, model: Entity<ModelService>, reviewer: Entity<Reviewer>, cx: &mut Context<Self>) -> Self {
        let subscriptions = vec![
            cx.observe(&state, |_, _, cx| cx.notify()),
            cx.observe(&model, |_, _, cx| cx.notify()),
            cx.observe(&reviewer, |_, _, cx| cx.notify()),
        ];
        Self { state, model, reviewer, _subscriptions: subscriptions }
    }

    fn reveal(&self, id: TaskId, cx: &mut Context<Self>) {
        self.state.update(cx, |state, cx| state.reveal(id, cx));
    }
}

fn direction(from: Priority, to: Priority) -> &'static str {
    if to > from { "Raise" } else { "Lower" }
}

impl Render for PlanPanel {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme().clone();
        let now = Local::now();
        let state = self.state.read(cx);
        let plan = state.plan(now);
        let tasks = state.store.tasks.clone();
        let reviewing = self.reviewer.read(cx).is_reviewing();
        let review_error = self.reviewer.read(cx).last_error.clone();
        let ready = self.model.read(cx).is_ready();
        let accent = palette::accent(cx);

        let provenance = if reviewing {
            h_flex().gap_1p5().child(Spinner::new().xsmall().color(accent)).child("The model is reviewing the list…")
        } else if plan.from_model {
            h_flex().gap_1p5().child(Icon::new(IconName::Sparkles).size(px(12.)).text_color(accent)).child(format!("Reviewed {} by the model", lists::ago(plan.generated, now)))
        } else {
            h_flex().gap_1p5().child(Icon::new(IconName::ListChecks).size(px(12.))).child(if ready { "Ranked by rules until the model reviews it" } else { "Ranked by rules" })
        };
        let header = v_flex()
            .gap_1()
            .child(h_flex().items_center().child(div().text_lg().font_bold().child("Plan")))
            .child(provenance.text_xs().text_color(theme.muted_foreground))
            .when(!plan.next.is_empty() && !plan.summary.trim().is_empty(), |col| col.child(div().pt_1().text_sm().child(plan.summary.clone())));

        let find = |id: TaskId| tasks.iter().find(|t| t.id == id);
        let mut items = plan.next.iter().filter_map(|item| find(item.id).map(|task| (task, item.reason.clone())));

        let hero = items.next().map(|(task, reason)| {
            let id = task.id;
            let started = task.status == Status::InProgress;
            ui::colored_card(accent, cx)
                .id("up-next")
                .cursor_pointer()
                .on_click(cx.listener(move |this, _, _, cx| this.reveal(id, cx)))
                .child(h_flex().gap_1p5().items_center().child(Icon::new(IconName::ArrowRight).size(px(14.)).text_color(accent)).child(div().text_xs().font_semibold().text_color(accent).child("UP NEXT")))
                .child(div().text_lg().font_semibold().line_height(px(24.)).child(task.title.clone()))
                .child(
                    h_flex()
                        .gap_1p5()
                        .flex_wrap()
                        .child(ui::pill(task.category.clone(), palette::category(&task.category, cx), cx))
                        .child(ui::priority_badge(task.priority, cx))
                        .children(task_row::due_pill(task, now, cx))
                        .children(task.estimate_minutes.map(|m| ui::icon_pill(IconName::Clock, task_row::estimate(m), theme.muted_foreground, cx))),
                )
                .when(!reason.trim().is_empty(), |card| card.child(div().text_sm().text_color(theme.muted_foreground).child(reason)))
                .child(
                    h_flex()
                        .gap_2()
                        .pt_1()
                        .child(
                            Button::new("hero-done")
                                .small()
                                .primary()
                                .icon(IconName::Check)
                                .label("Done")
                                .on_click(cx.listener(move |this, _, _, cx| {
                                    cx.stop_propagation();
                                    this.state.update(cx, |state, cx| state.set_done(id, true, cx));
                                })),
                        )
                        .child(
                            Button::new("hero-start")
                                .small()
                                .outline()
                                .icon(IconName::Play)
                                .label(if started { "In progress" } else { "Start" })
                                .disabled(started)
                                .on_click(cx.listener(move |this, _, _, cx| {
                                    cx.stop_propagation();
                                    let patch = TaskPatch { status: Some(Status::InProgress), ..Default::default() };
                                    this.state.update(cx, |state, cx| state.patch(id, patch, cx));
                                })),
                        ),
                )
        });

        let rest: Vec<_> = items.collect();
        let more = rest.len().saturating_sub(LISTED);
        let then = (!rest.is_empty()).then(|| {
            v_flex()
                .gap_1()
                .child(ui::caption("THEN", cx))
                .children(rest.into_iter().take(LISTED).enumerate().map(|(i, (task, reason))| {
                    let id = task.id;
                    h_flex()
                        .id(("plan", id.0))
                        .gap_2()
                        .items_start()
                        .px_2()
                        .py_1p5()
                        .rounded_md()
                        .cursor_pointer()
                        .hover(|row| row.bg(theme.secondary))
                        .on_click(cx.listener(move |this, _, _, cx| this.reveal(id, cx)))
                        .child(div().w(px(18.)).flex_none().text_sm().font_semibold().text_color(theme.muted_foreground).child(format!("{}", i + 2)))
                        .child(
                            v_flex()
                                .flex_1()
                                .min_w_0()
                                .child(
                                    h_flex()
                                        .gap_1p5()
                                        .items_center()
                                        .child(div().size(px(7.)).flex_none().rounded_full().bg(palette::priority(task.priority, cx)))
                                        .child(div().text_sm().font_medium().truncate().child(task.title.clone())),
                                )
                                .when(!reason.trim().is_empty(), |col| col.child(div().text_xs().text_color(theme.muted_foreground).line_clamp(2).child(reason))),
                        )
                }))
                .when(more > 0, |col| col.child(div().px_2().text_xs().text_color(theme.muted_foreground).child(format!("and {more} more"))))
        });

        let suggestions = (!plan.suggestions.is_empty()).then(|| {
            v_flex().gap_2().child(ui::caption("SUGGESTED PRIORITIES", cx)).children(plan.suggestions.iter().filter_map(|s| {
                let task = find(s.id)?;
                let id = s.id;
                let color = palette::priority(s.priority, cx);
                Some(
                    ui::card(cx)
                        .gap_1p5()
                        .p_3()
                        .child(
                            h_flex()
                                .gap_1()
                                .flex_wrap()
                                .items_center()
                                .text_sm()
                                .child(direction(task.priority, s.priority))
                                .child(div().font_semibold().child(format!("“{}”", task.title)))
                                .child("to")
                                .child(ui::priority_badge(s.priority, cx)),
                        )
                        .when(!s.reason.trim().is_empty(), |card| card.child(div().text_xs().text_color(theme.muted_foreground).child(s.reason.clone())))
                        .child(
                            h_flex()
                                .gap_1()
                                .child(
                                    Button::new(("accept", id.0))
                                        .xsmall()
                                        .outline()
                                        .icon(Icon::new(IconName::Check).text_color(color))
                                        .label("Accept")
                                        .on_click(cx.listener(move |this, _, _, cx| this.state.update(cx, |state, cx| state.accept_suggestion(id, cx)))),
                                )
                                .child(
                                    Button::new(("dismiss", id.0))
                                        .xsmall()
                                        .ghost()
                                        .label("Dismiss")
                                        .on_click(cx.listener(move |this, _, _, cx| this.state.update(cx, |state, cx| state.dismiss_suggestion(id, cx)))),
                                ),
                        ),
                )
            }))
        });

        let empty = plan.next.is_empty().then(|| {
            v_flex()
                .gap_1()
                .py_6()
                .items_center()
                .child(ui::badge(IconName::Sun, palette::success(cx), 40., cx))
                .child(div().pt_1().text_sm().font_medium().child("All clear"))
                .child(ui::muted("Nothing open. Enjoy it.", cx))
        });

        v_flex()
            .id("plan-panel")
            .size_full()
            .overflow_y_scroll()
            .gap_4()
            .p_4()
            .child(header)
            .children(hero)
            .children(suggestions)
            .children(then)
            .children(empty)
            .when_some(review_error, |col, error| {
                col.child(div().text_xs().text_color(palette::danger(cx)).child(format!("Last review failed: {error}")))
            })
    }
}
