//! Preferences, saved in `<data dir>/todo-biv/settings.json`. Every field has
//! a default, so files from older versions (or hand-edited ones) still load.

use std::path::{Path, PathBuf};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    /// A GGUF path or an `hf.co/<owner>/<repo>:<quant>` name
    pub model: String,
    /// Run the model on the CPU even where Metal is available
    pub cpu: bool,
    /// Let the model reason before answering
    pub thinking: bool,
    pub load_at_startup: bool,
    /// How often the model reviews the list when nothing changed
    pub review_minutes: u32,
    pub microphone: bool,
    /// Short name of the Whisper checkpoint
    pub whisper: String,
    /// Empty for the system's default input
    pub input_device: String,
    /// Whether a transcript is sent as soon as it is heard
    pub auto_submit: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            model: assistant::DEFAULT_MODEL.into(),
            cpu: false,
            thinking: false,
            load_at_startup: true,
            review_minutes: 10,
            microphone: false,
            whisper: speech2text::whisper::hub::DEFAULT_CHECKPOINT.into(),
            input_device: String::new(),
            auto_submit: true,
        }
    }
}

impl Settings {
    /// `settings.json` next to the tasks
    pub fn default_path() -> PathBuf {
        todo_core::store::data_dir().join("settings.json")
    }

    /// The settings at `path`, or the defaults if there are none. A file
    /// that cannot be read entirely is copied to `settings.json.bad` (the
    /// next change would overwrite it), the fields that can be read are kept,
    /// and the second value says so, for the user.
    pub fn load(path: &Path) -> (Self, Option<String>) {
        let text = match std::fs::read_to_string(path) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return (Self::default(), None),
            Err(e) => return (Self::default(), Some(format!("Could not read the settings ({e}); the defaults are used."))),
        };
        let error = match serde_json::from_str(&text) {
            Ok(settings) => return (settings, None),
            Err(error) => error,
        };
        let (settings, ignored) = Self::salvage(&text);
        let reverted = match ignored {
            None => "the defaults are used".to_string(),
            Some(fields) => format!("{} reverted to the default", fields.join(", ")),
        };
        let backup = path.with_extension("json.bad");
        let copied = match std::fs::copy(path, &backup) {
            Ok(_) => format!("the file was copied to {}", backup.display()),
            Err(e) => format!("the file could not be copied aside ({e})"),
        };
        (settings, Some(format!("Could not read all the settings ({error}): {reverted}; {copied}.")))
    }

    /// The fields of `text` that can be read, over the defaults, and the
    /// names of those that cannot; `None` for the names if it is not a JSON object
    fn salvage(text: &str) -> (Self, Option<Vec<String>>) {
        let Ok(Value::Object(fields)) = serde_json::from_str::<Value>(text) else { return (Self::default(), None) };
        let Ok(Value::Object(mut merged)) = serde_json::to_value(Self::default()) else { return (Self::default(), None) };
        let mut ignored = Vec::new();
        for (name, value) in fields {
            if !merged.contains_key(&name) {
                continue;
            }
            let mut candidate = merged.clone();
            candidate.insert(name.clone(), value);
            if serde_json::from_value::<Self>(Value::Object(candidate.clone())).is_ok() {
                merged = candidate;
            } else {
                ignored.push(name);
            }
        }
        (serde_json::from_value(Value::Object(merged)).unwrap_or_default(), Some(ignored))
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(path, serde_json::to_vec_pretty(self)?)?;
        Ok(())
    }

    /// What the model service is asked to load, with a leading `~/` of a
    /// path made the home directory (nothing expands it otherwise)
    pub fn model_config(&self) -> assistant::Config {
        let model = self.model.trim();
        let model = match (model.strip_prefix("~/"), std::env::home_dir()) {
            (Some(rest), Some(home)) => home.join(rest).display().to_string(),
            _ => model.to_string(),
        };
        assistant::Config { model, cpu: self.cpu, thinking: self.thinking, ..Default::default() }
    }

    pub fn whisper_checkpoint(&self) -> &'static speech2text::whisper::hub::Checkpoint {
        speech2text::whisper::hub::checkpoint_or_default(&self.whisper)
    }

    pub fn device(&self) -> Option<&str> {
        Some(self.input_device.as_str()).filter(|d| !d.is_empty())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_fields_take_defaults() {
        let settings: Settings = serde_json::from_str(r#"{"microphone": true, "review_minutes": 30}"#).expect("parses");
        assert!(settings.microphone);
        assert_eq!(settings.review_minutes, 30);
        assert!(settings.auto_submit);
        assert!(settings.load_at_startup);
        assert_eq!(settings.model, assistant::DEFAULT_MODEL);
    }

    #[test]
    fn round_trips() {
        let dir = std::env::temp_dir().join(format!("todo-biv-settings-{}", std::process::id()));
        let path = dir.join("settings.json");
        let settings = Settings { thinking: true, input_device: "USB Mic".into(), ..Settings::default() };
        settings.save(&path).expect("saves");
        assert_eq!(Settings::load(&path), (settings, None));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_bad_file_keeps_what_it_can_and_is_copied_aside() {
        let dir = std::env::temp_dir().join(format!("todo-biv-bad-settings-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("a directory");
        let path = dir.join("settings.json");
        let text = r#"{"cpu": true, "review_minutes": "30", "model": "/m.gguf", "extra": 1}"#;
        std::fs::write(&path, text).expect("writes");
        let (settings, problem) = Settings::load(&path);
        assert_eq!(settings, Settings { cpu: true, model: "/m.gguf".into(), ..Settings::default() });
        let problem = problem.expect("a problem");
        assert!(problem.contains("review_minutes reverted"), "{problem}");
        assert_eq!(std::fs::read_to_string(dir.join("settings.json.bad")).expect("a copy"), text);

        std::fs::write(&path, "{\"cpu\": true,").expect("writes");
        let (settings, problem) = Settings::load(&path);
        assert_eq!(settings, Settings::default());
        assert!(problem.is_some_and(|p| p.contains("the defaults are used")));
        assert_eq!(Settings::load(&dir.join("none.json")), (Settings::default(), None));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_model_path_in_the_home_directory_is_expanded() {
        let home = std::env::home_dir().expect("a home directory");
        let settings = Settings { model: " ~/models/q.gguf ".into(), ..Settings::default() };
        assert_eq!(settings.model_config().model, home.join("models/q.gguf").display().to_string());
        let settings = Settings { model: "hf.co/a/b:Q4".into(), ..Settings::default() };
        assert_eq!(settings.model_config().model, "hf.co/a/b:Q4");
    }
}
