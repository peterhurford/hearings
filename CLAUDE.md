# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single-script CLI (`transcribe_hearing.py`) that turns a US congressional hearing — a YouTube URL, a committee hearing page, or a local video/audio file — into a Markdown report containing an executive summary (written from the AI Policy Network / AIPN perspective), detailed section-by-section notes, and a timestamped speaker-labeled transcript.

External dependencies: `ffmpeg` (silence detection + audio extraction), `yt-dlp` (YouTube download), and the `llm` CLI with `llm-gemini` and `llm-anthropic` plugins. Gemini is used for audio transcription (Claude doesn't accept audio); Claude is used for the summary and notes. API keys are stored via `llm keys set gemini` / `llm keys set claude`, NOT environment variables.

## Running

```bash
python3 transcribe_hearing.py <hearing-page-url>          # most common: scrape page + embedded YouTube
python3 transcribe_hearing.py --video file.mp4 [url]      # local video/audio; URL still useful for metadata
python3 transcribe_hearing.py --youtube-id <id> <url>     # bypass embed scraping
```

Useful flags: `-o file.md`, `--title "..."`, `--keep-audio`, `--transcript-only`, `--no-cache`, `--clear-cache`.

There is no test suite, linter, or build step — this is a single Python file.

## Architecture

The pipeline is six sequential steps in `main()` (transcribe_hearing.py:1145), each cached in `.hearings_cache/<slug>/` so reruns are cheap and interrupts are safe:

1. **Scrape hearing metadata** (`fetch_hearing_info`) — pulls title, date, witnesses, committee roster, and bill numbers from the committee page. This context gets fed into the transcription prompt so Gemini can label speakers correctly.
2. **Acquire audio** — `download_audio` via yt-dlp, OR `extract_audio_from_video` via ffmpeg for local files. Cached as `audio.mp3`.
3. **Silence-aware chunking** (`detect_silence` → `classify_silences` → `split_audio_at_silence`) — ffmpeg's `silencedetect` finds natural breaks; significant silences (≥2 min, recess/dead air) are *excised* and replaced with `[RECESS]` markers in the final transcript via `inject_silence_markers`. Target chunk size ≈30 min with 30s overlap.
4. **Per-chunk transcription** (`transcribe_chunk`) — each chunk gets a context-rich prompt (`build_transcription_prompt`) including the metadata and a running summary of prior chunks (`summarize_chunk`) for speaker-label continuity. Timestamps in each chunk are rewritten to absolute time via `fix_chunk_timestamps`. Each chunk's output is cached as `chunk_N.txt`.
5. **Stitch + quality check** (`stitch_transcripts`, `check_transcript_quality`) — concatenate, then scan for hallucination artifacts (word salad, duplicated blocks, backwards timestamps). On warning, the user deletes the suspect `chunk_N.txt` and reruns; only that chunk gets redone.
6. **LLM analysis** — `EXECUTIVE_SUMMARY_PROMPT` and `DETAILED_NOTES_PROMPT` are run against the stitched transcript via `run_llm_on_text`, then `assemble_report` writes the final Markdown.

### Things worth knowing before editing

- **Models are constants at the top of the file** (`TRANSCRIBE_MODEL`, `SUMMARY_MODEL`, transcribe_hearing.py:26-27). `TRANSCRIBE_MODEL` must support audio input.
- **The executive-summary voice is opinionated.** `EXECUTIVE_SUMMARY_PROMPT` (transcribe_hearing.py:30) is written for AIPN — a centrist AI-safety advocacy org. The transcript and detailed notes are perspective-neutral; only the executive summary is framed. Don't generalize it without being asked.
- **Caching granularity matters.** Each pipeline step writes to its own file in `.hearings_cache/<slug>/`. Adding a step should follow the same pattern (check for cached file → reuse, else compute → write). Deleting a single cache file should cleanly re-trigger just that step.
- **Silence handling is load-bearing.** The chunker splits *at* silences to avoid cutting speakers mid-sentence, and the silence-marker injection produces the `[RECESS]` annotations in the output. If you touch chunking, preserve both behaviors.
- **The hearing-page scraper is committee-site-specific** and brittle by nature — committee sites change their embed format. Failures here should fall back to `--youtube-id` or `--video`, not crash the run.
