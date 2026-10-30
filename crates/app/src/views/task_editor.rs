//! The inline editor under an expanded task. Every change is saved as it is
//! made, and a change made elsewhere (a request, an undo) is shown in the
//! fields, so that typing in one does not write its old text back.

use chrono::{Datelike as _, Duration, Local, NaiveDate, Weekday};
use gpui::{Context, Entity, SharedString, Subscription, Window, div, prelude::*, px};
use gpui_component::{
    ActiveTheme as _, Sizable as _,
    button::{Button, ButtonVariants as _},
    calendar::Date,
    date_picker::{DatePicker, DatePickerEvent, DatePickerState, DateRangePreset},
    h_flex,
    input::{Input, InputEvent, InputState, Textarea, TextareaState},
    menu::{DropdownMenu as _, PopupMenuItem},
    v_flex,
};
use gpui_kit_assets::IconName;
use todo_core::{Priority, Status, Task, TaskId, TaskPatch};

use crate::{lists, palette, state::TodoState, ui};

pub struct TaskEditor {
    pub id: TaskId,
    state: Entity<TodoState>,
    title: Entity<InputState>,
    notes: Entity<TextareaState>,
    tags: Entity<InputState>,
    due: Entity<DatePickerState>,
    /// The task's fields when last shown, to tell a change made elsewhere
    seen: Fields,
    _subscriptions: Vec<Subscription>,
}

/// What the editor's text fields and date picker show of a task
#[derive(Clone, Default, PartialEq)]
struct Fields {
    title: String,
    notes: String,
    tags: Vec<String>,
    due: Option<NaiveDate>,
}

impl Fields {
    fn of(task: &Task) -> Self {
        Self { title: task.title.clone(), notes: task.notes.clone(), tags: task.tags.clone(), due: task.due }
    }
}

impl TaskEditor {
    pub fn new(id: TaskId, state: Entity<TodoState>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let seen = state.read(cx).store.get(id).map(Fields::of).unwrap_or_default();
        let title = cx.new(|cx| InputState::new(window, cx).placeholder("Title").default_value(seen.title.clone()));
        let notes = cx.new(|cx| TextareaState::new(window, cx).auto_grow(2, 8).placeholder("Notes").default_value(seen.notes.clone()));
        let tags = cx.new(|cx| InputState::new(window, cx).placeholder("Tags, separated by commas").default_value(seen.tags.join(", ")));
        let due = cx.new(|cx| DatePickerState::new(window, cx).date_format("%a %b %-d, %Y"));
        if let Some(date) = seen.due {
            due.update(cx, |picker, cx| picker.set_date(date, window, cx));
        }
        let subscriptions = vec![
            cx.observe_in(&state, window, |this, _, window, cx| this.sync(window, cx)),
            cx.subscribe(&title, |this, input, event: &InputEvent, cx| {
                if let InputEvent::Change = event {
                    let text = input.read(cx).value().trim().to_string();
                    if !text.is_empty() {
                        this.patch(TaskPatch { title: Some(text), ..Default::default() }, cx);
                    }
                }
            }),
            cx.subscribe(&notes, |this, input, event: &InputEvent, cx| {
                if let InputEvent::Change = event {
                    let text = input.read(cx).value().trim_end().to_string();
                    this.patch(TaskPatch { notes: Some(text), ..Default::default() }, cx);
                }
            }),
            cx.subscribe(&tags, |this, input, event: &InputEvent, cx| {
                if let InputEvent::Change = event {
                    let tags = parse_tags(&input.read(cx).value());
                    this.patch(TaskPatch { tags: Some(tags), ..Default::default() }, cx);
                }
            }),
            cx.subscribe(&due, |this, _, event: &DatePickerEvent, cx| {
                let DatePickerEvent::Change(value) = event;
                let date = value.start().map(|d| d.date());
                this.patch(TaskPatch { due: Some(date), ..Default::default() }, cx);
            }),
        ];
        Self { id, state, title, notes, tags, due, seen, _subscriptions: subscriptions }
    }

    /// Shows in the fields what changed in the task since they were last
    /// synced, unless they show it already (the change came from them)
    fn sync(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(now) = self.state.read(cx).store.get(self.id).map(Fields::of) else { return };
        let seen = std::mem::replace(&mut self.seen, now.clone());
        if now.title != seen.title && self.title.read(cx).value().trim() != now.title {
            self.title.update(cx, |input, cx| input.set_value(now.title.clone(), window, cx));
        }
        if now.notes != seen.notes && self.notes.read(cx).value().trim() != now.notes {
            self.notes.update(cx, |input, cx| input.set_value(now.notes.clone(), window, cx));
        }
        if now.tags != seen.tags && parse_tags(&self.tags.read(cx).value()) != now.tags {
            self.tags.update(cx, |input, cx| input.set_value(now.tags.join(", "), window, cx));
        }
        if now.due != seen.due {
            self.due.update(cx, |picker, cx| picker.set_date(Date::Single(now.due), window, cx));
        }
    }

    fn patch(&self, patch: TaskPatch, cx: &mut Context<Self>) {
        let id = self.id;
        self.state.update(cx, |state, cx| state.patch(id, patch, cx));
    }

    fn category_menu(&self, current: &str, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let choices = lists::category_choices(&self.state.read(cx).store.tasks);
        let current = current.to_string();
        let editor = cx.entity().downgrade();
        let color = palette::category(&current, cx);
        Button::new("category")
            .small()
            .outline()
            .child(h_flex().gap_1p5().items_center().child(div().size(px(8.)).rounded_full().bg(color)).child(current.clone()))
            .dropdown_caret(true)
            .dropdown_menu(move |mut menu, _, _| {
                for choice in &choices {
                    let editor = editor.clone();
                    let category = choice.clone();
                    let checked = choice.eq_ignore_ascii_case(&current);
                    menu = menu.item(PopupMenuItem::new(choice.clone()).checked(checked).on_click(move |_, _, cx| {
                        let category = category.clone();
                        let _ = editor.update(cx, |editor, cx| editor.patch(TaskPatch { category: Some(category), ..Default::default() }, cx));
                    }));
                }
                menu
            })
    }
}

/// Tags as typed: separated by commas or spaces, `#` optional
pub fn parse_tags(text: &str) -> Vec<String> {
    let mut tags: Vec<String> = Vec::new();
    for tag in text.split([',', ' ']).map(|t| t.trim().trim_start_matches('#').to_lowercase()).filter(|t| !t.is_empty()) {
        if !tags.contains(&tag) {
            tags.push(tag);
        }
    }
    tags
}

/// Due date shortcuts offered by the date picker
fn preset_dates(today: NaiveDate) -> [(&'static str, NaiveDate); 5] {
    // The coming `weekday`, a week ahead if it is today
    let next = |weekday: Weekday| {
        let ahead = (weekday.num_days_from_monday() + 7 - today.weekday().num_days_from_monday()) % 7;
        today + Duration::days(if ahead == 0 { 7 } else { ahead as i64 })
    };
    [
        ("Today", today),
        ("Tomorrow", today + Duration::days(1)),
        ("Friday", next(Weekday::Fri)),
        ("Monday", next(Weekday::Mon)),
        ("In a week", today + Duration::days(7)),
    ]
}

fn presets(today: NaiveDate) -> Vec<DateRangePreset> {
    preset_dates(today).into_iter().map(|(label, date)| DateRangePreset::single(label, date)).collect()
}

/// A row of mutually exclusive buttons
fn choices<T: Copy + PartialEq + 'static>(
    id: &'static str,
    options: impl IntoIterator<Item = (T, SharedString)>,
    current: T,
    cx: &mut Context<TaskEditor>,
    on_select: impl Fn(&mut TaskEditor, T, &mut Context<TaskEditor>) + Copy + 'static,
) -> gpui::Div {
    h_flex().gap_1().flex_wrap().children(options.into_iter().enumerate().map(|(i, (value, label))| {
        let button = Button::new((id, i)).small().label(label).on_click(cx.listener(move |this, _, _, cx| on_select(this, value, cx)));
        if value == current { button.primary() } else { button.ghost() }
    }))
}

fn field(label: &'static str, content: impl IntoElement, cx: &gpui::App) -> gpui::Div {
    v_flex().gap_1().child(ui::caption(label, cx)).child(content)
}

impl Render for TaskEditor {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme().clone();
        let Some(task) = self.state.read(cx).store.get(self.id).cloned() else { return div().into_any_element() };
        let today = Local::now().date_naive();
        let id = self.id;

        let priority = choices(
            "priority",
            Priority::ALL.map(|p| (p, SharedString::from(p.label()))),
            task.priority,
            cx,
            |this, priority, cx| this.patch(TaskPatch { priority: Some(priority), ..Default::default() }, cx),
        );
        let status = choices(
            "status",
            [Status::Todo, Status::InProgress, Status::Done].map(|s| (s, SharedString::from(s.label()))),
            task.status,
            cx,
            |this, status, cx| this.patch(TaskPatch { status: Some(status), ..Default::default() }, cx),
        );
        let due = DatePicker::new(&self.due).small().cleanable(true).placeholder("No due date").presets(presets(today));
        let origin = {
            let added = format!("Added {}", lists::ago(task.created, Local::now()));
            match task.source.trim() {
                "" => added,
                source => format!("{added} from “{source}”"),
            }
        };

        v_flex()
            .gap_3()
            .p_4()
            .border_1()
            .border_t_0()
            .border_color(theme.border)
            .rounded_b_lg()
            .bg(theme.background)
            .child(field("Title", Input::new(&self.title), cx))
            .child(field("Notes", Textarea::new(&self.notes), cx))
            .child(
                h_flex()
                    .gap_4()
                    .flex_wrap()
                    .items_start()
                    .child(field("Category", self.category_menu(&task.category, cx), cx))
                    .child(field("Due", div().w(px(220.)).child(due), cx))
                    .child(field("Tags", div().w(px(220.)).child(Input::new(&self.tags).small()), cx)),
            )
            .child(h_flex().gap_4().flex_wrap().child(field("Priority", priority, cx)).child(field("Status", status, cx)))
            .child(
                h_flex()
                    .gap_2()
                    .items_center()
                    .child(div().flex_1().min_w_0().text_xs().text_color(theme.muted_foreground).truncate().child(origin))
                    .child(
                        Button::new("delete")
                            .small()
                            .ghost()
                            .icon(IconName::Trash)
                            .label("Delete")
                            .on_click(cx.listener(move |this, _, _, cx| this.state.update(cx, |state, cx| state.delete(id, cx)))),
                    )
                    .child(
                        Button::new("close")
                            .small()
                            .outline()
                            .label("Close")
                            .on_click(cx.listener(move |this, _, _, cx| this.state.update(cx, |state, cx| state.toggle_expanded(id, cx)))),
                    ),
            )
            .into_any_element()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tags_are_split_and_cleaned() {
        assert_eq!(parse_tags("#Phone, errands  phone,, "), ["phone", "errands"]);
        assert!(parse_tags(" , ").is_empty());
    }

    #[test]
    fn presets_land_on_the_right_days() {
        let date = |day| NaiveDate::from_ymd_opt(2026, 11, day).expect("a date");
        let friday = NaiveDate::from_ymd_opt(2026, 10, 30).expect("a date");
        let dates = preset_dates(friday).map(|(_, date)| date);
        assert_eq!(dates, [friday, date(1) - Duration::days(1), date(6), date(2), date(6)]);
        let tuesday = date(3);
        assert_eq!(preset_dates(tuesday).map(|(_, date)| date)[2..4], [date(6), date(9)]);
    }
}
