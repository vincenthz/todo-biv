# Todo Biv

A to-do list you talk to. Requests in plain language — typed, or spoken in
microphone mode — become tasks that are classified, prioritized and given due
dates, and the list is reviewed every so often to say what to do next. The
language model ([biv](../biv), Qwen3.8 27B on Metal) and the speech recognition
(Whisper on [candle](https://github.com/huggingface/candle)) run on the Mac;
nothing leaves it.

```sh
cargo run --release                                        # the app (bin todo-biv)
cargo run --release -p assistant --example try             # requests and a review through the model, in the terminal
cargo run --release -p assistant --example try -- -r "buy milk and call the bank tomorrow"
cargo run --release -p assistant --example try -- --compare  # greedy answers from the saved system prompt vs without it
scripts/bundle.sh                                          # target/release/bundle/osx/Todo Biv.app, signed ad hoc
```

## How it works

A request goes to the model with today's date, the categories, the open tasks
and their ids. It answers in JSON with operations — add, update, complete, reopen,
delete — and a one-line reply, which are applied to the list:

| request | becomes |
|---|---|
| remind me to call mum tomorrow, it's important | add "Call mum", Personal, High, due tomorrow |
| buy milk, call the bank and finish the slides by friday | three tasks, each with its own category; the slides due Friday |
| I finished the report | complete the report task |
| move the dentist to friday | update the dentist task's due date |
| the tax return is urgent | raise its priority to Urgent |
| forget about the gym | delete the gym task |

Replies are parsed leniently: JSON in a code fence or surrounded by prose,
enum words in any case, relative dates, and ids that are not in the list
dropped rather than failing the reply. Categories come from a default set
(Work, Personal, Home, Health, Finance, Errands, Learning, Social) or are
invented by the model and kept.

While the model is loading, or if it cannot be loaded, a rule-based reader
handles requests: adding by default, "done/finished <task>" and
"delete/remove/forget <task>" matched by title words ("cancel <task>" only when
it names the whole title; "drop" and "scratch" always add), a category from
keywords, a priority from words like "urgent" or "whenever", and due dates from
"today", "tomorrow", "friday", "next friday" (next week's, or the coming one
once this week's is past), "next week", "in 3 days", "on the 12th" or "Friday 6
November" (the date wins over the weekday).

The review sends the whole list and asks for the open tasks in the order to
take them, with a reason for each, priorities that look wrong (overdue,
blocking others, stale), and a short summary. Priority suggestions are shown
to be accepted or dismissed, never applied silently; a dismissed one is not
shown again until the task's priority or due date changes. A review runs once
the list and the model have both been quiet for about 20 s (a request cancels a
review, but cannot cut short the reading of its prompt, so the next request is
not made to wait), after the list changed and otherwise every 10 minutes (5 to
60 in the settings); a failed one is retried after 2 minutes. Until the first review,
and without the model, the plan is a deterministic ranking: priority, then
due date (overdue and due-soon tasks rise), then age.

Each request is one independent model turn, with thinking off, so it starts
from a short prompt. The system prompts (the instructions and the JSON format)
are constant text, and what changes (the date, the categories, the list) is in
the user message, so the model's state after a system prompt is saved once
(with biv's `Chat::save`) and every request starts from a copy of it
(`Chat::restore`, ~5 ms): only the user message is read. The request's system
prompt (384 tokens) is read right after loading, in about 2.3 s, before the
model is reported ready; the review's (253 tokens) by the first review. Each
saved state takes about 170 MB. With the default model on an M2 Max a request
takes 3 to 7 s: 1 to 2 s to read the 170 to 340 token message (~160 tok/s),
then 1 to 6 s to write the answer (~28 tok/s; a completion is ~30 tokens,
three new tasks ~170). Reading the system prompt with every request, as
before, took 2 to 3 s more (5 to 10 s, 555 to 730 tokens). A review reads its
280 to 350 token message in about 2 s; the rest is the answer, a summary and
one sentence per task to do next (up to 7), at about 0.04 s a token: about
5.5 s with 2 open tasks (a 96-token answer), about 14 s with 7 or more (about
300 tokens). The first review takes 1.5 s more, to save its system prompt, as
did every review before. With temperature 0 the answers are the same text
either way (`try --compare`). If a state cannot be saved or restored, requests
fall back to reading the system prompt each time.
`TODO_BIV_TRACE=1` prints the token counts and timings of each request to
stderr.

In microphone mode the request is recorded with a level meter, resampled to
16 kHz mono, transcribed by Whisper in English, and sent like a typed one.
Whisper is loaded (and its GPU kernels prepared) when the mode is turned on,
in about 9 s; a short request is then transcribed in 0.2 s (`small`) to 0.6 s
(`large-v3-turbo`). Silence, and the phrases Whisper makes up over it, give an
empty transcript, which is not sent. A recording stops by itself after 2
minutes. One made of nothing but zeros is not transcribed: that is what macOS
delivers when the app was denied the microphone, and the app says where to
allow it. Recording waits for the request being answered, and turning
microphone mode off (or opening the settings) discards a recording. Turning it
off during a download or load stops it.

## Workspace

```text
crates/
  core        todo-core: tasks and the operations on them, the JSON store,
              the model prompts and the parsing of its replies, the rule-based
              reader, the deterministic plan. No UI, no model.
  assistant   The biv model on a worker thread: requests in, operations and
              plans out, with partial text while it generates.
  app         todo-biv: the GPUI Component 0.8 desktop app, and its macOS
              bundle (assets/, [package.metadata.bundle]).
```

Dependencies point one way: `app` → `assistant` → `core`, and `app` →
[speech2text](https://github.com/vincenthz/speech2text) (microphone capture
with cpal and English Whisper on candle, Metal on macOS, with checkpoint
downloads from Hugging Face). The core never sees the model or audio; it reads the model's text
and the text Whisper heard.

## Requirements

- macOS 13 or later on Apple silicon (Metal). The model also runs on the CPU,
  about five times slower.
- A checkout of [biv](../biv) next to this one: the workspace uses
  `../biv/engine` (the engine) and `../biv` (`biv_cli`, for GGUF files and the
  Hugging Face cache).
- The language model in the Hugging Face cache (`$HF_HUB_CACHE`,
  `$HUGGINGFACE_HUB_CACHE`, `$HF_HOME/hub`, `$XDG_CACHE_HOME/huggingface/hub`,
  `~/.cache/huggingface/hub`, the first that is set). The default is
  `hf.co/unsloth/Qwen3.8-27B-GGUF:Q4_1` (16 GB, so 32 GB of memory or more); any
  `hf.co/<owner>/<repo>:<quant>` name of a `qwen35` GGUF, or the full path of a
  `.gguf` file (`~/` for the home folder; a relative path would be resolved
  against `/` in the bundled app), can be set instead. biv only reads the cache, it does not
  download:

  ```sh
  hf download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-Q4_1.gguf
  ```

- For microphone mode, a Whisper checkpoint. It is downloaded into the same
  cache the first time it is needed if it is not there:

  | checkpoint | repository | download |
  |---|---|---|
  | `large-v3-turbo` (default) | `openai/whisper-large-v3-turbo` | 1.6 GB |
  | `small` | `openai/whisper-small` | 970 MB |

An app opened from the Finder does not see the shell's environment, so it
uses `~/.cache/huggingface/hub` unless one of those variables is set with
`launchctl setenv` (e.g. `launchctl setenv HF_HUB_CACHE /Volumes/ssd/hub`).

## Keyboard shortcuts

| keys | does |
|---|---|
| ⌘K, ⌘L | focus the request field |
| Return | send the request (or stop recording) |
| ⌘⇧M | start or stop recording (microphone mode) |
| Esc | cancel the request being answered, or discard a recording |
| ⌘Z | undo the last request or deletion, until the list is changed by hand (in the field: undo typing while there is text) |
| ⌘R | review the list now |
| ⌘, | settings |
| ⌘Q | quit |

## Bundle

```sh
cargo install cargo-bundle      # 0.11
scripts/bundle.sh               # target/release/bundle/osx/Todo Biv.app
scripts/bundle.sh --dmg         # and target/release/bundle/dmg/Todo Biv.dmg
```

cargo-bundle resolves the icon and `Info.plist.ext` paths of
`[package.metadata.bundle]` against the current directory, so it has to run
from `crates/app` (`cd crates/app && cargo bundle --release --format osx`);
from the workspace root, `cargo bundle -p todo-biv` builds an app without
icon and without the microphone usage description. `crates/app/assets/Info.plist.ext`
is appended inside the generated `Info.plist`'s `<dict>`, so it holds bare
key/value pairs: `NSMicrophoneUsageDescription` and
`NSSupportsAutomaticGraphicsSwitching` (cargo-bundle writes
`NSHighResolutionCapable` and `LSApplicationCategoryType` itself). The icon is
drawn in `crates/app/assets/icon/icon.svg` and exported to the PNG sizes the
`.icns` is made of.

The script signs the app ad hoc (`codesign --force --deep -s - "Todo
Biv.app"`). The linker's signature of the binary alone does not cover
`Info.plist`; signing the bundle binds it, so macOS shows the usage
description and remembers the answer. No entitlements file is needed:
`com.apple.security.device.audio-input` is only required with the hardened
runtime or the App Sandbox, which an ad hoc build does not use. An ad hoc
signature changes with every build, so macOS may ask again after a rebuild;
`tccutil reset Microphone io.typed.todo-biv` clears the old answer.

`spctl --assess` rejects an ad hoc signature, so a copy that was downloaded
(quarantined) is blocked the first time. On macOS 15 and later, try to open it
once, then choose System Settings > Privacy & Security > Open Anyway (or run
`xattr -dr com.apple.quarantine "Todo Biv.app"`); on macOS 13 and 14,
right-click > Open also works. A locally built one opens directly.

Run with `cargo run`, the app is not a bundle: macOS asks the terminal for
microphone access instead, and attributes it to the terminal.

## Data

The list, with the last review, is one JSON file written atomically after
every change: `~/Library/Application Support/todo-biv/tasks.json` on macOS
(the platform's data directory elsewhere). Each task keeps the request it came
from. The settings (model, CPU, thinking, load at startup, review interval,
microphone mode, Whisper checkpoint, input device, sending transcripts at
once) are in `settings.json` next to it; every field is optional.

A file that cannot be read does not stop the app, and the window says what
happened. A `tasks.json` that is not entirely readable (edited by hand, or
written by another version) is copied to `tasks.json.unreadable-<time>` and
the tasks that can be read are kept; a plan that cannot be read is dropped. In
a `settings.json` that cannot be read, the fields that cannot are reset to
their defaults and the file is copied to `settings.json.bad` first. A list
that cannot be opened at all (permissions) is used in memory only, so it is
not overwritten.

`TODO_BIV_DATA=<dir>` keeps both files in another directory, e.g. to try the
app without touching the real list.

## Tests

```sh
cargo test --workspace                                      # prompts, reply parsing, rules, store, ranking, app state
cargo test -p todo-biv screenshots -- --ignored             # renders the app offscreen to target/screenshots/*.png
```
