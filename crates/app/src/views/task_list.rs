//! The tasks of the selected view, the expanded one with its editor, or an
//! empty state with examples of what to ask.

use chrono::Local;
use gpui::{Context, Entity, Subscription, Window, div, prelude::*, px};
use gpui_component::{ActiveTheme as _, StyledExt as _, h_flex, v_flex};
use gpui_kit_assets::IconName;

use crate::{
    lists::{self, View},
    state::TodoState,
    ui,
    views::{task_editor::TaskEditor, task_row},
};

pub struct TaskList {
    state: Entity<TodoState>,
    editor: Option<Entity<TaskEditor>>,
    _observe: Subscription,
}

impl TaskList {
    pub fn new(state: Entity<TodoState>, cx: &mut Context<Self>) -> Self {
        let observe = cx.observe(&state, |_, _, cx| cx.notify());
        Self { state, editor: None, _observe: observe }
    }

    /// The editor of the expanded task, made when another is expanded
    fn editor(&mut self, window: &mut Window, cx: &mut Context<Self>) -> Option<Entity<TaskEditor>> {
        let expanded = self.state.read(cx).expanded;
        match (expanded, &self.editor) {
            (None, _) => self.editor = None,
            (Some(id), Some(editor)) if editor.read(cx).id == id => {}
            (Some(id), _) => {
                let state = self.state.clone();
                self.editor = Some(cx.new(|cx| TaskEditor::new(id, state, window, cx)));
            }
        }
        self.editor.clone()
    }
}

fn empty(view: &View, cx: &gpui::App) -> impl IntoElement {
    let examples = [
        "Try: “remind me to call mum tomorrow, it's important”",
        "or: “buy milk, call the bank and finish the slides by friday”",
    ];
    match view {
        View::NextUp | View::AllOpen => ui::empty_state(IconName::ListTodo, "Nothing to do", &examples, cx).into_any_element(),
        View::Today => ui::empty_state(
            IconName::Sun,
            "Nothing due today",
            &["Tasks due today or overdue show here.", "Try: “pay the electricity bill today”"],
            cx,
        )
        .into_any_element(),
        View::Upcoming => ui::empty_state(
            IconName::CalendarDays,
            "Nothing scheduled",
            &["Tasks with a due date after today show here.", "Try: “move the dentist to friday”"],
            cx,
        )
        .into_any_element(),
        View::Done => ui::empty_state(
            IconName::CircleCheckBig,
            "Nothing done yet",
            &["Tick a task, or say “I finished the report”."],
            cx,
        )
        .into_any_element(),
        View::Category(name) => {
            ui::empty_state(IconName::Folder, &format!("No open {name} tasks"), &examples[..1], cx).into_any_element()
        }
    }
}

fn subtitle(view: &View, from_model: bool) -> &'static str {
    match view {
        View::NextUp if from_model => "In the order the model suggests",
        View::NextUp => "Ranked by priority and due date",
        View::Today => "Due today or overdue",
        View::Upcoming => "By due date",
        View::AllOpen => "Most pressing first",
        View::Done => "Most recently finished first",
        View::Category(_) => "Open tasks, most pressing first",
    }
}

impl Render for TaskList {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let editor = self.editor(window, cx);
        let theme = cx.theme().clone();
        let now = Local::now();
        let state = self.state.read(cx);
        let plan = state.plan(now);
        let view = state.view.clone();
        let tasks = lists::tasks_in(&view, &state.store.tasks, &plan, now);

        let header = h_flex()
            .gap_2()
            .items_baseline()
            .child(div().text_xl().font_bold().child(view.label().to_string()))
            .child(div().text_sm().text_color(theme.muted_foreground).child(format!("{}", tasks.len())))
            .child(div().flex_1())
            .child(div().text_xs().text_color(theme.muted_foreground).child(subtitle(&view, plan.from_model)));

        let rows = tasks.iter().enumerate().map(|(i, task)| {
            let expanded = state.expanded == Some(task.id);
            let look = task_row::RowLook {
                expanded,
                highlighted: state.is_highlighted(task.id),
                rank: (view == View::NextUp).then_some(i),
            };
            v_flex()
                .child(task_row::render(task, look, &self.state, now, cx))
                .when(expanded, |col| col.children(editor.clone()))
        });
        let body = if tasks.is_empty() { empty(&view, cx).into_any_element() } else { v_flex().gap_0p5().children(rows).into_any_element() };

        v_flex().size_full().child(div().px_6().pt_4().pb_2().child(header)).child(
            div()
                .id("tasks")
                .flex_1()
                .min_h_0()
                .overflow_y_scroll()
                .child(div().px_4().pb(px(80.)).child(body)),
        )
    }
}
