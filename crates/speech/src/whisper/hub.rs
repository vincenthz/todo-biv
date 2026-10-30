//! The Whisper checkpoints the app offers, and downloading them from Hugging Face.

use std::{
    panic::{AssertUnwindSafe, catch_unwind, resume_unwind},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail};
use hf_hub::{
    Cache, Repo, RepoType,
    api::{Progress, sync::ApiBuilder},
};

/// A multilingual Whisper checkpoint on Hugging Face.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Checkpoint {
    /// Short name, used in settings and on the command line.
    pub name: &'static str,
    pub label: &'static str,
    pub repo: &'static str,
    /// Pinned commit, so the weights match the code that was tested with them.
    pub revision: &'static str,
    /// Approximate size of the weights, in MB.
    pub download_mb: u32,
}

/// Best first.
pub const CHECKPOINTS: &[Checkpoint] = &[
    Checkpoint {
        name: "large-v3-turbo",
        label: "Large v3 Turbo",
        repo: "openai/whisper-large-v3-turbo",
        revision: "41f01f3fe87f28c78e2fbf8b568835947dd65ed9",
        download_mb: 1618,
    },
    Checkpoint {
        name: "small",
        label: "Small",
        repo: "openai/whisper-small",
        revision: "973afd24965f72e36ca33b3055d56a652f456b4d",
        download_mb: 967,
    },
];

pub const DEFAULT_CHECKPOINT: &str = "large-v3-turbo";

/// The checkpoint called `name`, or the default one if there is none.
pub fn checkpoint_or_default(name: &str) -> &'static Checkpoint {
    CHECKPOINTS
        .iter()
        .find(|c| c.name == name)
        .or_else(|| CHECKPOINTS.iter().find(|c| c.name == DEFAULT_CHECKPOINT))
        .unwrap_or(&CHECKPOINTS[0])
}

/// Files a checkpoint directory must provide.
pub const FILES: [&str; 4] = ["config.json", "generation_config.json", "tokenizer.json", "model.safetensors"];

#[derive(Clone, Debug)]
pub struct DownloadProgress {
    pub file: String,
    pub downloaded: u64,
    pub total: u64,
}

fn repo(model: &Checkpoint) -> Repo {
    Repo::with_revision(model.repo.into(), RepoType::Model, model.revision.into())
}

/// The Hugging Face hub cache, found as `huggingface_hub` (and biv, for the
/// language model) find it: `$HF_HUB_CACHE`, `$HUGGINGFACE_HUB_CACHE`,
/// `$HF_HOME/hub`, `$XDG_CACHE_HOME/huggingface/hub`, then
/// `~/.cache/huggingface/hub`. (hf-hub itself only reads `$HF_HOME`.)
pub fn cache_dir() -> PathBuf {
    cache_dir_from(|name| std::env::var(name).ok(), std::env::home_dir())
}

fn cache_dir_from(var: impl Fn(&str) -> Option<String>, home: Option<PathBuf>) -> PathBuf {
    let var = |name: &str| var(name).filter(|v| !v.is_empty());
    if let Some(dir) = var("HF_HUB_CACHE").or_else(|| var("HUGGINGFACE_HUB_CACHE")) {
        return PathBuf::from(dir);
    }
    let hf_home = var("HF_HOME").map(PathBuf::from).unwrap_or_else(|| {
        var("XDG_CACHE_HOME").map(PathBuf::from).unwrap_or_else(|| home.unwrap_or_default().join(".cache")).join("huggingface")
    });
    hf_home.join("hub")
}

/// The checkpoint directory if every file of `model` is already downloaded.
pub fn cached(model: &Checkpoint) -> Option<PathBuf> {
    let repo = Cache::new(cache_dir()).repo(repo(model));
    let mut dir = None;
    for name in FILES {
        dir = repo.get(name)?.parent().map(Path::to_path_buf);
    }
    dir
}

/// Fetches (or finds in the Hugging Face cache) every file of `model`, and
/// returns the directory that holds them. (hf-hub's `download` always
/// downloads; only its cache knows what is already there.)
pub fn fetch(model: &Checkpoint, on_progress: &mut dyn FnMut(DownloadProgress)) -> Result<PathBuf> {
    fetch_unless(model, on_progress, &|| false)
}

/// Why [`fetch_unless`] gave up
pub const CANCELLED: &str = "cancelled";

/// [`fetch`], given up (with the error [`CANCELLED`]) as soon as `cancelled`
/// says so, even in the middle of a file; what was downloaded of it is kept
/// for the next try to resume.
pub fn fetch_unless(model: &Checkpoint, on_progress: &mut dyn FnMut(DownloadProgress), cancelled: &dyn Fn() -> bool) -> Result<PathBuf> {
    let repo = repo(model);
    let dir = cache_dir();
    let cache = Cache::new(dir.clone()).repo(repo.clone());
    // from_env for the endpoint ($HF_ENDPOINT) and the token
    let api = ApiBuilder::from_env().with_cache_dir(dir).with_progress(false).build()?.repo(repo.clone());
    let mut dir = None;
    for name in FILES {
        if cancelled() {
            bail!(CANCELLED);
        }
        let path = match cache.get(name) {
            Some(path) => path,
            None => {
                // hf-hub offers no way to stop a download but to unwind out of its progress callback
                let download = catch_unwind(AssertUnwindSafe(|| api.download_with_progress(name, Reporter::new(on_progress, cancelled))));
                match download {
                    Ok(result) => result.with_context(|| format!("downloading {}/{name}", repo.url()))?,
                    Err(payload) if payload.is::<Stop>() => bail!(CANCELLED),
                    Err(payload) => resume_unwind(payload),
                }
            }
        };
        dir = path.parent().map(Path::to_path_buf);
    }
    dir.context("no files to fetch")
}

/// The unwinding payload that stops a download
struct Stop;

/// Forwards hf-hub progress to a callback, at most every 100 ms, and stops
/// the download once `cancelled`.
struct Reporter<'a> {
    callback: &'a mut dyn FnMut(DownloadProgress),
    cancelled: &'a dyn Fn() -> bool,
    progress: DownloadProgress,
    last_emit: Instant,
}

impl<'a> Reporter<'a> {
    fn new(callback: &'a mut dyn FnMut(DownloadProgress), cancelled: &'a dyn Fn() -> bool) -> Self {
        Self {
            callback,
            cancelled,
            progress: DownloadProgress { file: String::new(), downloaded: 0, total: 0 },
            last_emit: Instant::now(),
        }
    }

    fn emit(&mut self) {
        (self.callback)(self.progress.clone());
        self.last_emit = Instant::now();
    }
}

impl Progress for Reporter<'_> {
    fn init(&mut self, size: usize, filename: &str) {
        self.progress = DownloadProgress { file: filename.into(), downloaded: 0, total: size as u64 };
        self.emit();
    }

    fn update(&mut self, size: usize) {
        if (self.cancelled)() {
            // resume_unwind: no panic message, no backtrace
            resume_unwind(Box::new(Stop));
        }
        self.progress.downloaded += size as u64;
        if self.last_emit.elapsed() >= Duration::from_millis(100) {
            self.emit();
        }
    }

    fn finish(&mut self) {
        self.emit();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_the_cache_as_huggingface_hub_does() {
        let home = Some(PathBuf::from("/home/me"));
        let env = |vars: &'static [(&'static str, &'static str)]| move |name: &str| vars.iter().find(|(n, _)| *n == name).map(|(_, v)| v.to_string());
        assert_eq!(cache_dir_from(env(&[]), home.clone()), PathBuf::from("/home/me/.cache/huggingface/hub"));
        assert_eq!(cache_dir_from(env(&[("XDG_CACHE_HOME", "/x")]), home.clone()), PathBuf::from("/x/huggingface/hub"));
        assert_eq!(cache_dir_from(env(&[("XDG_CACHE_HOME", "/x"), ("HF_HOME", "/hf")]), home.clone()), PathBuf::from("/hf/hub"));
        const ALL: &[(&str, &str)] = &[("HF_HOME", "/hf"), ("HUGGINGFACE_HUB_CACHE", "/old"), ("HF_HUB_CACHE", "/ssd/hub")];
        assert_eq!(cache_dir_from(env(ALL), home.clone()), PathBuf::from("/ssd/hub"));
        assert_eq!(cache_dir_from(env(&ALL[..2]), home.clone()), PathBuf::from("/old"));
        assert_eq!(cache_dir_from(env(&[("HF_HUB_CACHE", "")]), home), PathBuf::from("/home/me/.cache/huggingface/hub"));
    }

    #[test]
    fn a_cancelled_fetch_gives_up_before_any_download() {
        let checkpoint = checkpoint_or_default("small");
        let error = fetch_unless(checkpoint, &mut |_| {}, &|| true).expect_err("cancelled");
        assert_eq!(error.to_string(), CANCELLED);
    }

    #[test]
    fn falls_back_to_the_default() {
        assert_eq!(checkpoint_or_default("small").name, "small");
        assert_eq!(checkpoint_or_default("nope").name, DEFAULT_CHECKPOINT);
        assert_eq!(checkpoint_or_default("").name, DEFAULT_CHECKPOINT);
    }
}
