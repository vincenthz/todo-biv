//! On-device English speech recognition: microphone capture (cpal) and
//! Whisper on candle (Metal when available). Adapted from thai-learn's
//! `speech` crate, without its language profiles or text-to-speech.

pub mod audio;
pub mod whisper;
