//! Small building blocks shared by the views.

use gpui::{App, Div, Hsla, IntoElement, ParentElement, SharedString, Styled, div, px};
use gpui_component::{ActiveTheme as _, Icon, StyledExt as _, h_flex, v_flex};
use gpui_kit_assets::IconName;
use todo_core::Priority;

use crate::palette;

/// A small colored label.
pub fn pill(text: impl Into<SharedString>, color: Hsla, cx: &App) -> Div {
    div()
        .flex_none()
        .px_2()
        .rounded_full()
        .text_xs()
        .font_medium()
        .text_color(color)
        .bg(palette::tint(color, cx))
        .child(text.into())
}

/// A pill with an icon before its text.
pub fn icon_pill(icon: IconName, text: impl Into<SharedString>, color: Hsla, cx: &App) -> Div {
    h_flex()
        .flex_none()
        .gap_1()
        .px_2()
        .rounded_full()
        .text_xs()
        .font_medium()
        .text_color(color)
        .bg(palette::tint(color, cx))
        .child(Icon::new(icon).size(px(11.)).text_color(color))
        .child(text.into())
}

/// The priority badge: a flag and the priority's name, in its color.
pub fn priority_badge(priority: Priority, cx: &App) -> Div {
    icon_pill(IconName::Flag, priority.label(), palette::priority(priority, cx), cx)
}

/// An icon on a disc of its color's tint.
pub fn badge(icon: IconName, color: Hsla, size: f32, cx: &App) -> Div {
    div()
        .flex()
        .flex_none()
        .items_center()
        .justify_center()
        .size(px(size))
        .rounded_full()
        .bg(palette::tint(color, cx))
        .child(Icon::new(icon).size(px(size * 0.55)).text_color(color))
}

/// A bordered panel.
pub fn card(cx: &App) -> Div {
    let theme = cx.theme();
    v_flex().gap_2().p_4().rounded_lg().border_1().border_color(theme.border).bg(theme.background)
}

/// A panel washed with `color`, with a colored edge.
pub fn colored_card(color: Hsla, cx: &App) -> Div {
    v_flex().gap_2().p_4().rounded_lg().border_1().border_l_4().border_color(color.opacity(0.6)).bg(palette::tint(color, cx))
}

/// A small uppercase heading over a group.
pub fn caption(text: impl Into<SharedString>, cx: &App) -> Div {
    div().text_xs().font_semibold().text_color(cx.theme().muted_foreground).child(text.into())
}

pub fn section_title(text: impl Into<SharedString>) -> Div {
    div().text_lg().font_semibold().child(text.into())
}

/// Secondary text.
pub fn muted(text: impl Into<SharedString>, cx: &App) -> Div {
    div().text_sm().text_color(cx.theme().muted_foreground).child(text.into())
}

/// A thin horizontal bar filled to `fraction`.
pub fn meter(fraction: f32, color: Hsla, cx: &App) -> Div {
    div()
        .h(px(6.))
        .w_full()
        .rounded_full()
        .bg(palette::tint(color, cx))
        .child(div().h_full().rounded_full().w(gpui::relative(fraction.clamp(0.0, 1.0))).bg(color))
}

/// A large icon over a title and some lines of help, for empty lists.
pub fn empty_state(icon: IconName, title: &str, lines: &[&str], cx: &App) -> impl IntoElement {
    let theme = cx.theme();
    v_flex()
        .items_center()
        .gap_2()
        .py_12()
        .px_6()
        .child(badge(icon, palette::accent(cx), 56., cx))
        .child(div().pt_2().text_base().font_semibold().child(title.to_string()))
        .children(lines.iter().map(|line| div().text_sm().text_color(theme.muted_foreground).child(line.to_string())))
}
