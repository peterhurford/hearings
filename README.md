# Hearings

A command-line tool that turns a US congressional hearing (YouTube video or local video/audio file) into a polished Markdown report containing:

- An executive summary written from the perspective of the AI Policy Network (AIPN)
- Detailed, skimmable notes broken into hearing sections
- A full, timestamped, speaker-labeled transcript

It downloads the audio, splits it on natural silences, transcribes each chunk with Google Gemini (Claude doesn't accept audio input), then asks Claude to summarize and annotate the result.

> **First time?** Skip to [Setup](#setup) at the bottom, then come back here.

---

## Running the tool

There are three ways to use it, depending on what you have in hand. **Mode A is by far the most common.**

### Mode A — You have a hearing page URL

Most congressional committee hearing pages embed the YouTube livestream. Just paste the URL:

```bash
python3 transcribe_hearing.py https://foreignaffairs.house.gov/committee-activity/hearings/strengthening-export-control-enforcement
```

The script will:

1. Scrape the page for the title, witnesses, committee roster, date, and any bill numbers mentioned
2. Find the embedded YouTube video and download its audio
3. Split the audio at natural silences (so chunks don't cut speakers off mid-sentence)
4. Transcribe each chunk with Gemini, feeding in the metadata so speakers get labeled correctly
5. Generate an executive summary and detailed notes
6. Write a Markdown file named after the hearing slug (e.g. `strengthening-export-control-enforcement.md`)

Progress is printed continuously. A spinner shows while Gemini is processing a chunk. Expect roughly **5-20 minutes total** for a typical 2-3 hour hearing.

### Mode B — You already have a video or audio file on disk

If you downloaded the hearing video yourself, or recorded it, point the script at the file:

```bash
python3 transcribe_hearing.py --video judiciary042226.mp4 --title "Judiciary Hearing on AI Oversight"
```

`--title` is optional but recommended — it sets the heading at the top of the report.

You can also combine `--video` with a URL so the script still picks up metadata (members, witnesses, bills) from the hearing page:

```bash
python3 transcribe_hearing.py --video hearing.mp4 https://...committee-url...
```

Supported audio extensions (skip video extraction): `.mp3`, `.wav`, `.flac`, `.ogg`, `.m4a`, `.aac`. Anything else is treated as video and converted to MP3.

### Mode C — You know the YouTube video ID but not the embed URL

```bash
python3 transcribe_hearing.py --youtube-id dQw4w9WgXcQ https://...committee-url...
```

### Useful flags

| Flag | What it does |
| --- | --- |
| `-o file.md` / `--output file.md` | Override the output Markdown filename. |
| `--title "..."` | Override the report's title heading. |
| `--keep-audio` | Save a copy of the extracted MP3 alongside the report (`<slug>.mp3`). |
| `--transcript-only` | Skip the executive summary and detailed notes — just produce the timestamped transcript. |
| `--no-cache` | Throw away any cached work for this hearing and start fresh. |
| `--clear-cache` | Delete the cache for this hearing and exit (no transcription run). |

### What you get out

A single Markdown file in the current directory, structured like:

```markdown
# <Hearing Title>

## Executive Summary
... 1-page analyst summary written for AIPN ...

## Detailed Notes
... skimmable bullet notes broken by hearing section ...

## Transcript
**[00:00:12] CHAIR HUIZENGA:** The committee will come to order...
**[00:00:34] WITNESS NAME:** Thank you, Chairman...
*[00:42:10] [RECESS — 7 minutes]*
...
```

Open it in any Markdown viewer (VS Code, Obsidian, Typora, GitHub preview).

### Re-running is cheap — every step is cached

Every expensive step lands in `.hearings_cache/<hearing-slug>/`: the audio, each transcribed chunk, the stitched transcript, and the two generated analyses. Re-running the same command reuses everything it already has.

- **A failed or interrupted run is safe to restart** — it picks up where it left off.
- **If one chunk transcribes badly,** delete just that `chunk_N.txt` from the cache folder and re-run — only that chunk gets redone.
- **To start completely fresh,** use `--no-cache` (or `--clear-cache` to wipe and exit).

### Quality check

After stitching, the script automatically scans for hallucination artifacts (word salad, duplicated blocks, timestamps that go backwards). If it finds anything, it prints a `TRANSCRIPT QUALITY CHECK` warning telling you which timestamps look suspect. To fix:

1. Look in `.hearings_cache/<slug>/` for the chunk file covering that timestamp
2. Delete that chunk file
3. Re-run the same command

### Troubleshooting

**"Could not find a YouTube video on the hearing page."**
The committee site changed its embed format, or the video isn't on YouTube. Pass `--youtube-id <id>` directly, or download the video yourself and use `--video file.mp4`.

**`yt-dlp` errors about a "Post-Live Manifestless" video.**
The hearing just ended and YouTube hasn't finished processing the VOD. Wait 15-30 minutes and try again.

**`llm` errors about an unknown model.**
You haven't installed the Gemini plugin or set your API key — re-run the [Setup](#setup) steps.

**The transcript looks wrong / has obvious hallucinations.**
Delete the suspect chunk file (see Quality check above) and re-run. If problems persist, the audio quality may be poor — try `--no-cache` to start over.

**You want to swap models.**
Edit the `TRANSCRIBE_MODEL` and `SUMMARY_MODEL` constants near the top of `transcribe_hearing.py`. Anything supported by your installed `llm` plugins will work.

### A note on the analyst voice

The executive summary is written from the perspective of the **AI Policy Network (AIPN)** — a centrist AI-safety-focused advocacy organization. If you want a neutral or differently-framed summary, edit the `EXECUTIVE_SUMMARY_PROMPT` string near the top of `transcribe_hearing.py`. The transcript and detailed notes are perspective-neutral.

---

## Setup

You need a Mac or Linux Terminal, Python 3.9+, and `ffmpeg`. On a Mac:

```bash
brew install python ffmpeg
```

Clone the repo and install Python dependencies:

```bash
git clone <this-repo-url> hearings
cd hearings
pip install -r requirements.txt
```

Register two API keys with `llm` — Gemini (for audio transcription) and Anthropic (for summary/notes):

```bash
# Get a free key at https://aistudio.google.com/app/apikey
llm keys set gemini

# Get a key at https://console.anthropic.com/settings/keys
llm keys set claude
```

Confirm everything works:

```bash
llm -m gemini/gemini-3.1-pro-preview "say hello"
llm -m claude-opus-4-7 "say hello"
```
