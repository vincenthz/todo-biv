//! Color: a vivid accent instead of the default monochrome, a hue for each
//! priority and category, used consistently everywhere.

use gpui::{App, Hsla, hsla};
use gpui_component::{ActiveTheme as _, Theme};
use todo_core::Priority;

const ACCENT_HUE: f32 = 252.0;

/// A saturated color of `hue` degrees, adjusted for the light or dark theme.
pub fn vivid(hue: f32, cx: &App) -> Hsla {
    if cx.theme().is_dark() { hsla(hue / 360.0, 0.80, 0.66, 1.0) } else { hsla(hue / 360.0, 0.72, 0.48, 1.0) }
}

/// A faint wash of `color`, for backgrounds.
pub fn tint(color: Hsla, cx: &App) -> Hsla {
    color.opacity(if cx.theme().is_dark() { 0.16 } else { 0.09 })
}

pub fn accent(cx: &App) -> Hsla {
    vivid(ACCENT_HUE, cx)
}

pub fn danger(cx: &App) -> Hsla {
    vivid(356.0, cx)
}

pub fn success(cx: &App) -> Hsla {
    vivid(145.0, cx)
}

pub fn warning(cx: &App) -> Hsla {
    vivid(32.0, cx)
}

/// Urgent red, high orange, medium blue, low gray
pub fn priority(priority: Priority, cx: &App) -> Hsla {
    match priority {
        Priority::Urgent => danger(cx),
        Priority::High => vivid(26.0, cx),
        Priority::Medium => vivid(212.0, cx),
        Priority::Low => cx.theme().muted_foreground,
    }
}

/// The same hue for a category wherever it shows; categories the model
/// invents get one from their name.
pub fn category_hue(name: &str) -> f32 {
    match name.to_ascii_lowercase().as_str() {
        "work" => 222.0,
        "personal" => 280.0,
        "home" => 28.0,
        "health" => 150.0,
        "finance" => 48.0,
        "errands" => 190.0,
        "learning" => 318.0,
        "social" => 340.0,
        other => {
            let hash = other.bytes().fold(2166136261u32, |h, b| (h ^ b as u32).wrapping_mul(16777619));
            (hash % 360) as f32
        }
    }
}

pub fn category(name: &str, cx: &App) -> Hsla {
    vivid(category_hue(name), cx)
}

/// Replaces the theme's neutral primary colors with the accent. Call again
/// after every theme change, which resets them.
pub fn apply(cx: &mut App) {
    let dark = Theme::global(cx).is_dark();
    let shade = |lightness_shift: f32| {
        let base = if dark { 0.66 } else { 0.52 };
        hsla(ACCENT_HUE / 360.0, 0.78, base + lightness_shift, 1.0)
    };
    let on_accent = if dark { hsla(0.0, 0.0, 0.07, 1.0) } else { hsla(0.0, 0.0, 1.0, 1.0) };
    Theme::update(cx, |theme| {
        let colors = &mut theme.colors;
        colors.primary = shade(0.0);
        colors.primary_hover = shade(if dark { 0.05 } else { -0.05 });
        colors.primary_active = shade(if dark { 0.10 } else { -0.10 });
        colors.primary_foreground = on_accent;
        colors.button_primary = colors.primary;
        colors.button_primary_hover = colors.primary_hover;
        colors.button_primary_active = colors.primary_active;
        colors.button_primary_foreground = on_accent;
        colors.ring = colors.primary;
        colors.progress_bar = colors.primary;
        colors.link = colors.primary;
        colors.link_hover = colors.primary_hover;
        colors.link_active = colors.primary_active;
        colors.sidebar_primary = colors.primary;
        colors.sidebar_primary_foreground = on_accent;
        colors.caret = colors.primary;
        colors.selection = colors.primary.opacity(0.25);
        colors.sidebar = colors.primary.opacity(if dark { 0.08 } else { 0.045 });
        colors.sidebar_accent = colors.primary.opacity(if dark { 0.22 } else { 0.13 });
        colors.accent = colors.primary.opacity(if dark { 0.18 } else { 0.10 });
        colors.title_bar = colors.primary.opacity(if dark { 0.08 } else { 0.045 });
    });
}
