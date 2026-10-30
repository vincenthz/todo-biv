//! A to-do list you talk to. Plain-language requests, typed or spoken, are
//! read by a local language model (biv) into tasks that it classifies,
//! prioritizes and dates; the model reviews the list periodically to say what
//! to do next.

mod lists;
mod palette;
#[cfg(test)]
mod screenshots;
mod services;
mod settings;
mod state;
mod ui;
mod views;

use gpui::{prelude::*, *};
use gpui_component::{Root, Theme, TitleBar};

use crate::{
    services::{model::ModelService, reviewer::Reviewer, speech::SpeechService},
    state::TodoState,
    views::shell::{self, Shell},
};

/// Follows the system's light or dark appearance, with the app's accent.
fn sync_appearance(window: &mut Window, cx: &mut App) {
    Theme::sync_system_appearance(Some(window), cx);
    palette::apply(cx);
}

fn main() {
    let state = TodoState::load();
    gpui_platform::application().with_assets(gpui_kit_assets::AllAssets).run(move |cx: &mut App| {
        gpui_component::init(cx);
        shell::bind_keys(cx);
        cx.on_window_closed(|cx, _| {
            if cx.windows().is_empty() {
                cx.quit();
            }
        })
        .detach();

        let settings = state.settings.clone();
        let state = cx.new(|_| state);
        let model = cx.new(ModelService::spawn);
        if settings.load_at_startup {
            model.update(cx, |model, cx| model.load(settings.model_config(), cx));
        }
        let speech = cx.new(|cx| SpeechService::new(settings.whisper_checkpoint(), settings.microphone, cx));
        let reviewer = cx.new(|cx| {
            let mut reviewer = Reviewer::new(state.clone(), model.clone(), cx);
            reviewer.start(cx);
            reviewer
        });

        let bounds = Bounds::centered(None, size(px(1240.), px(820.)), cx);
        let opened = cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                window_min_size: Some(size(px(900.), px(560.))),
                titlebar: Some(TitlebarOptions { title: Some("Todo Biv".into()), ..TitleBar::title_bar_options() }),
                app_owns_titlebar_drag: true,
                ..Default::default()
            },
            |window, cx| {
                sync_appearance(window, cx);
                window.observe_window_appearance(sync_appearance).detach();
                let shell = cx.new(|cx| Shell::new(state, model, speech, reviewer, window, cx));
                cx.new(|cx| Root::new(shell, window, cx))
            },
        );
        if let Err(error) = opened {
            eprintln!("Could not open the window: {error:#}");
            cx.quit();
            return;
        }
        cx.activate(true);
    });
}
