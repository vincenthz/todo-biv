//! One task in the list: its done checkbox, title and notes, and pills for
//! its category, priority and due date.

use chrono::{DateTime, Local};
use gpui::{AnyElement, App, Entity, div, prelude::*, px};
use gpui_component::{ActiveTheme as _, Icon, StyledExt as _, checkbox::Checkbox, h_flex, v_flex};
use gpui_kit_assets::IconName;
use todo_core::{Status, Task};

use crate::{
    lists::{self, Due},
    palette,
    state::TodoState,
    ui,
};

/// How a row shows, besides its task
pub struct RowLook {
    pub expanded: bool,
    pub highlighted: bool,
    /// The task's place in the plan, shown in "Next up"
    pub rank: Option<usize>,
}

/// The due date pill, colored by how pressing it is
pub fn due_pill(task: &Task, now: DateTime<Local>, cx: &App) -> Option<gpui::Div> {
    let due = task.due?;
    let (label, urgency) = lists::due_label(due, now.date_naive());
    let color = match urgency {
        Due::Overdue => palette::danger(cx),
        Due::Today => palette::warning(cx),
        Due::Soon => palette::accent(cx),
        Due::Later => cx.theme().muted_foreground,
    };
    Some(ui::icon_pill(IconName::Calendar, label, color, cx))
}

pub fn render(task: &Task, look: RowLook, state: &Entity<TodoState>, now: DateTime<Local>, cx: &App) -> AnyElement {
    let theme = cx.theme();
    let id = task.id;
    let open = task.is_open();
    let accent = palette::accent(cx);

    let checkbox = {
        let state = state.clone();
        Checkbox::new(("done", id.0)).checked(!open).on_click(move |checked, _, cx| {
            cx.stop_propagation();
            let checked = *checked;
            state.update(cx, |state, cx| state.set_done(id, checked, cx));
        })
    };

    let title = div()
        .text_base()
        .font_medium()
        .line_clamp(2)
        .when(!open, |t| t.line_through().text_color(theme.muted_foreground))
        .child(task.title.clone());
    let notes = task.notes.lines().next().filter(|line| !line.trim().is_empty()).map(|line| {
        div().text_sm().text_color(theme.muted_foreground).truncate().child(line.to_string())
    });
    let tags = (!task.tags.is_empty()).then(|| {
        h_flex().gap_1p5().children(
            task.tags.iter().map(|tag| div().text_xs().text_color(theme.muted_foreground).child(format!("#{tag}"))),
        )
    });

    let mut pills = h_flex().gap_1p5().flex_none().items_center();
    if task.status == Status::InProgress {
        pills = pills.child(ui::icon_pill(IconName::LoaderCircle, "In progress", palette::success(cx), cx));
    }
    if let Some(minutes) = task.estimate_minutes {
        pills = pills.child(ui::icon_pill(IconName::Clock, estimate(minutes), theme.muted_foreground, cx));
    }
    pills = pills.child(ui::pill(task.category.clone(), palette::category(&task.category, cx), cx));
    if open {
        pills = pills.child(ui::priority_badge(task.priority, cx)).children(due_pill(task, now, cx));
    } else if let Some(completed) = task.completed {
        pills = pills.child(ui::icon_pill(IconName::CircleCheck, lists::ago(completed, now), palette::success(cx), cx));
    }

    let rank = look.rank.map(|rank| {
        div()
            .flex_none()
            .w(px(20.))
            .text_sm()
            .font_semibold()
            .text_color(if rank == 0 { accent } else { theme.muted_foreground })
            .child(format!("{}", rank + 1))
    });

    let state = state.clone();
    h_flex()
        .id(("task", id.0))
        .gap_3()
        .items_center()
        .px_3()
        .py_2p5()
        .rounded_lg()
        .border_1()
        .border_color(gpui::transparent_black())
        .cursor_pointer()
        .when(look.expanded, |row| row.bg(theme.secondary).border_color(theme.border).rounded_b_none())
        .when(look.highlighted, |row| row.bg(palette::tint(accent, cx)).border_color(accent.opacity(0.5)))
        .when(!look.expanded && !look.highlighted, |row| row.hover(|row| row.bg(theme.secondary.opacity(0.6))))
        .on_click(move |_, _, cx| state.update(cx, |state, cx| state.toggle_expanded(id, cx)))
        .children(rank)
        .child(checkbox)
        .child(v_flex().flex_1().min_w_0().gap_0p5().child(title).children(notes).children(tags))
        .child(pills)
        .child(Icon::new(if look.expanded { IconName::ChevronUp } else { IconName::ChevronDown }).size(px(14.)).text_color(theme.muted_foreground.opacity(0.6)))
        .into_any_element()
}

/// An effort estimate, briefly: "20 min", "1 h", "1 h 30"
pub fn estimate(minutes: u32) -> String {
    match (minutes / 60, minutes % 60) {
        (0, m) => format!("{m} min"),
        (h, 0) => format!("{h} h"),
        (h, m) => format!("{h} h {m:02}"),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn estimates_read_briefly() {
        assert_eq!(super::estimate(20), "20 min");
        assert_eq!(super::estimate(60), "1 h");
        assert_eq!(super::estimate(90), "1 h 30");
    }
}
