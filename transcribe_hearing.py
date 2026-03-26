#!/usr/bin/env python3
"""
Transcribe a congressional hearing and generate a full analysis report.

Usage:
    python transcribe_hearing.py <hearing_url>

Example:
    python transcribe_hearing.py https://foreignaffairs.house.gov/committee-activity/hearings/strengthening-export-control-enforcement
"""

import argparse
import itertools
import os
import re
import subprocess
import shutil
import sys
import threading
import time
import requests


TRANSCRIBE_MODEL = "gemini/gemini-3.1-pro-preview"
SUMMARY_MODEL = "gemini/gemini-3.1-pro-preview"


EXECUTIVE_SUMMARY_PROMPT = """\
You are a policy analyst. Given the following transcript of a congressional hearing, produce an executive summary and member assessment.

**CONTEXT:** This analysis is for the AI Policy Network (AIPN), a 501(c)(4) advocacy organization focused on preparing America for AI superintelligence. AIPN's core view is that AI superintelligence poses a literal existential risk. AIPN believes the US must lead on AI, must embrace the positives, and must NOT ban AI prematurely — but most importantly must prepare to ensure catastrophic and existential risks are mitigated. This makes AIPN a centrist voice: opposed both to those who want to ban all data centers (e.g., Sanders) AND to those who want zero rules or full preemption replacing state laws with nothing meaningful (e.g., some industry actors). AIPN operates through direct congressional engagement on AI safety, export controls, military AI reliability, autonomous weapons accountability, catastrophic risk insurance, and related issues. Key issues: AI safety, AI security, frontier AI risk, compute governance, existential risk preparedness, and building relationships with members engaged on these issues.

**OUTPUT FORMAT (use markdown):**

1. **Executive Summary (1 page max)** — High-level takeaways, vibes, and what a policy team should think after this hearing. Be specific, not generic. Include:
   - Key consensus points or disagreements
   - Notable moments, especially anything touching AI safety/x-risk/frontier concerns (even if it wasn't the dominant frame)
   - Industry dynamics and who got criticized/praised
   - Signals to monitor

2. **Member Assessment** — Assess members from AIPN's perspective (pro-safety, pro-preparedness, anti-reckless-acceleration, anti-premature-ban). Who is aligned with AIPN's mission? Who is persuadable? Who is opposed? Note that members concerned about AI risks, safety, or accountability may be natural allies even if they are critical of industry — and members who are purely pro-industry with no interest in safety guardrails are NOT automatic allies. Be direct and actionable.

3. **Legislative Vehicles** — Bills mentioned, sponsors, and status

**GUIDANCE:**
- Optimize for what's actionable, not comprehensive coverage
- If a member says something notable about AI risk/safety/AGI even in passing, flag it prominently — this is the core lane
- Don't bury outliers just because they weren't the dominant frame
- Specific quotes > vague summaries
- Name names

Do NOT include any preamble like "Here is the summary". Do NOT include top-level section headers like "# Executive Summary" or "## Executive Summary" — these are added externally. Use bold text (e.g., **Member Assessment**) for sub-sections within your output. Start directly with the content.

TRANSCRIPT:
{transcript}
"""


DETAILED_NOTES_PROMPT = """\
You are a policy analyst producing detailed notes on a congressional hearing transcript.

**TASK:**

1. First, identify the hearing sections. Congressional hearings typically have: opening statements, witness testimony, and rounds of member questioning. Break these into clear sections with descriptive headings.
2. Briefly intro the hearing overall — what committee/subcommittee, who chaired it, who the witness(es) were, and the hearing's stated purpose.
3. Then, FOR EACH SECTION — *comprehensively* capture *as much discussion as possible* in bullets, maintaining detail and specific examples. Use twice as much detail as you might think.

Use bullets to capture the notes and increase the ability to quickly skim, avoid long paragraphs. Optimize the notes for easy skimming.

**STYLE GUIDE (MUST FOLLOW):**
- When referring to people, mention their full name first and then refer to them using their last name. Special exceptions for people known mononymously, stage names, and where a title is important (e.g., royalty).
- Use 'US' instead of 'U.S.'
- Prefer em dashes to hyphens and en dashes for separating sentences. Put spaces around em dashes, like " — ".

Do NOT include any preamble like "Here are the notes". Do NOT include a top-level section header like "# Detailed Notes" or "## Detailed Notes" — this is added externally. Start directly with the hearing overview content.

TRANSCRIPT:
{transcript}
"""


CACHE_DIR = ".hearings_cache"


def get_cache_dir(slug):
    """Get or create the cache directory for a hearing."""
    cache_dir = os.path.join(CACHE_DIR, slug)
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def clear_cache(slug):
    """Remove all cached data for a hearing."""
    cache_dir = os.path.join(CACHE_DIR, slug)
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"Cleared cache for: {slug}")
    else:
        print(f"No cache found for: {slug}")


def fetch_hearing_info(hearing_url):
    """Fetch the hearing page and extract video ID, title, witnesses, and subcommittee info."""
    print(f"Fetching hearing page: {hearing_url}")
    resp = requests.get(hearing_url, timeout=30)
    resp.raise_for_status()
    html = resp.text

    # Extract title
    title = None
    title_match = re.search(r'<title>([^<]+)</title>', html, re.IGNORECASE)
    if title_match:
        title = title_match.group(1).strip()
        title = re.split(r'\s*[|–—]\s*', title)[0].strip()

    # Extract YouTube video ID
    video_id = None
    for pattern in [
        r'youtube\.com/embed/([a-zA-Z0-9_-]{11})',
        r'youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})',
        r'youtu\.be/([a-zA-Z0-9_-]{11})',
    ]:
        match = re.search(pattern, html)
        if match:
            video_id = match.group(1)
            break

    # Extract witness info from page text using BeautifulSoup-like approach
    # Strip all HTML tags to get clean text
    clean_text = re.sub(r'<[^>]+>', '\n', html)
    clean_text = re.sub(r'\n\s*\n+', '\n', clean_text).strip()

    witnesses = []
    # Parse witness blocks from the HTML directly
    witness_section = re.search(r'field-evo-witnesses.*?field__items(.*?)</div>\s*</div>\s*</div>', html, re.DOTALL)
    if witness_section:
        section_html = witness_section.group(1)
        # Extract content from each <p> tag (witnesses are in <p><strong>Name</strong><br>Title<br>Org</p>)
        witness_blocks = re.findall(r'<p>(.*?)</p>', section_html, re.DOTALL)
        if not witness_blocks:
            # Fallback: just strip all tags from the section
            witness_blocks = [section_html]
        for block in witness_blocks:
            text = re.sub(r'<[^>]+>', ' ', block)
            text = re.sub(r'\s+', ' ', text).strip()
            if text:
                witnesses.append(text)

    # Extract subcommittee from the hearing metadata area (field__item div, not nav)
    subcommittee_name = None
    subcommittee_url = None
    from urllib.parse import urljoin
    sc_content_match = re.search(
        r'class="field__item"[^>]*>\s*<a\s+href="(/subcommittees/[^"]+)"[^>]*>([^<]+)</a>',
        html
    )
    if sc_content_match:
        subcommittee_url = urljoin(hearing_url, sc_content_match.group(1))
        subcommittee_name = sc_content_match.group(2).strip()

    # Fetch subcommittee membership
    members = []
    if subcommittee_url:
        try:
            print(f"Fetching subcommittee roster: {subcommittee_url}")
            sc_resp = requests.get(subcommittee_url, timeout=30)
            sc_resp.raise_for_status()
            sc_html = sc_resp.text

            # Extract clean text from the subcommittee page
            sc_clean = re.sub(r'<[^>]+>', '\n', sc_html)
            sc_clean = re.sub(r'\n\s*\n+', '\n', sc_clean).strip()

            # Find Chair line: "Chair\nName Name\nST ##"
            # or inline: "Chair Name Name ST ##"
            chair_match = re.search(
                r'(?:Subcommittee\s+)?Chair\s+([A-Z][a-zA-Z.\s-]+?)\s+([A-Z]{2})\s+(?:\d{2})?',
                sc_clean
            )
            if chair_match:
                members.append(f"Chair: {re.sub(r' +', ' ', chair_match.group(1).strip())}")

            rm_match = re.search(
                r'Ranking\s+Member\s+([A-Z][a-zA-Z.\s-]+?)\s+([A-Z]{2})\s+(?:\d{2})?',
                sc_clean
            )
            if rm_match:
                members.append(f"Ranking Member: {re.sub(r' +', ' ', rm_match.group(1).strip())}")

            # Parse member names between section headers
            # Republican Members section
            rep_match = re.search(
                r'Republican\s+Members\s+(.*?)(?:Ranking\s+Member|Democratic\s+Members)',
                sc_clean, re.DOTALL
            )
            if rep_match:
                rep_names = re.findall(
                    r'([A-Z][a-zA-Z.\s-]+?)\s+[A-Z]{2}\s+(?:\d{2})?',
                    rep_match.group(1)
                )
                for name in rep_names:
                    clean_name = re.sub(r'\s+', ' ', name).strip()
                    if len(clean_name) > 3 and 'twitter' not in clean_name.lower():
                        members.append(f"Republican: {clean_name}")

            dem_match = re.search(
                r'Democratic\s+Members\s+(.*?)(?:Recent\s+Activity|$)',
                sc_clean, re.DOTALL
            )
            if dem_match:
                dem_names = re.findall(
                    r'([A-Z][a-zA-Z.\s-]+?)\s+[A-Z]{2}\s+(?:\d{2})?',
                    dem_match.group(1)
                )
                for name in dem_names:
                    clean_name = re.sub(r'\s+', ' ', name).strip()
                    if len(clean_name) > 3 and 'twitter' not in clean_name.lower():
                        members.append(f"Democrat: {clean_name}")
        except Exception as e:
            print(f"Warning: Could not fetch subcommittee roster: {e}")

    # Extract hearing date from page metadata
    hearing_date = None
    # Try common date patterns in hearing pages (e.g., "field-evo-hearing-date", datetime attrs, etc.)
    date_match = re.search(r'<time[^>]*datetime="([^"]+)"', html)
    if date_match:
        hearing_date = date_match.group(1).strip()
    if not date_match:
        # Try field-based date
        date_field = re.search(r'field-evo-hearing-date.*?<time[^>]*>([^<]+)</time>', html, re.DOTALL)
        if date_field:
            hearing_date = date_field.group(1).strip()
    if not hearing_date:
        # Try text patterns like "March 5, 2025" or "03/05/2025"
        date_text_match = re.search(
            r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}',
            clean_text
        )
        if date_text_match:
            hearing_date = date_text_match.group(0)

    # Derive congress session from date
    congress = None
    if hearing_date:
        year_match = re.search(r'(\d{4})', hearing_date)
        if year_match:
            year = int(year_match.group(1))
            # Congress number: 119th = 2025-2026, 118th = 2023-2024, etc.
            congress_num = (year - 1789) // 2 + 1
            congress = f"{congress_num}th Congress"

    # Extract bill numbers (e.g., "H.R. 7688", "S. 1234", "HR 7688")
    bills = list(set(re.findall(
        r'(?:H\.?\s*R\.?\s*\d+|S\.?\s*\d+|H\.?\s*Res\.?\s*\d+|S\.?\s*Res\.?\s*\d+)',
        clean_text
    )))

    # Extract committee name from breadcrumbs or page structure
    committee_name = None
    # Try breadcrumb pattern
    committee_match = re.search(
        r'<a[^>]+href="/?"[^>]*>([^<]*(?:Committee|commission)[^<]*)</a>',
        html, re.IGNORECASE
    )
    if committee_match:
        committee_name = committee_match.group(1).strip()
    if not committee_name:
        # Try og:site_name or page title patterns
        og_match = re.search(r'<meta[^>]+property="og:site_name"[^>]+content="([^"]+)"', html)
        if og_match:
            committee_name = og_match.group(1).strip()

    # Extract hearing description/body text
    description = None
    # Try field-evo-body or similar content blocks
    body_match = re.search(r'field-evo-body.*?field__item[^>]*>(.*?)</div>', html, re.DOTALL)
    if body_match:
        desc_text = re.sub(r'<[^>]+>', ' ', body_match.group(1))
        desc_text = re.sub(r'\s+', ' ', desc_text).strip()
        if len(desc_text) > 20:
            description = desc_text
    if not description:
        # Try meta description
        meta_desc = re.search(r'<meta[^>]+name="description"[^>]+content="([^"]+)"', html, re.IGNORECASE)
        if meta_desc:
            desc_text = meta_desc.group(1).strip()
            if len(desc_text) > 20:
                description = desc_text

    hearing_context = {
        "title": title,
        "witnesses": witnesses,
        "subcommittee": subcommittee_name,
        "members": members,
        "date": hearing_date,
        "bills": bills,
        "description": description,
        "committee": committee_name,
        "congress": congress,
    }

    return video_id, title, hearing_context


def download_audio(youtube_id, output_dir):
    """Download audio from YouTube video using yt-dlp."""
    youtube_url = f"https://www.youtube.com/watch?v={youtube_id}"
    output_path = os.path.join(output_dir, "audio.mp3")

    print(f"Downloading audio from: {youtube_url}")
    cmd = [
        "yt-dlp",
        "-x",                      # Extract audio
        "--audio-format", "mp3",   # Convert to mp3
        "--audio-quality", "2",    # Good quality
        "-o", output_path,
        youtube_url,
    ]
    subprocess.run(cmd, check=True)

    if not os.path.exists(output_path):
        # yt-dlp sometimes adds extension
        for f in os.listdir(output_dir):
            if f.startswith("audio") and f.endswith(".mp3"):
                output_path = os.path.join(output_dir, f)
                break

    print(f"Audio saved to: {output_path}")
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Audio file size: {size_mb:.1f} MB")
    return output_path


CHUNK_DURATION_SECS = 1800  # 30 min per chunk (shorter = less hallucination)
CHUNK_OVERLAP_SECS = 30     # 30s overlap to avoid cutting mid-sentence

# Silence-aware splitting constants
SILENCE_NOISE_THRESH_DB = -30     # dB threshold for silence detection
SILENCE_MIN_DURATION_SECS = 10    # Minimum silence to detect
SILENCE_MERGE_GAP_SECS = 30      # Merge silences separated by < this
SIGNIFICANT_SILENCE_SECS = 120    # Silences >= 2 min = recess/dead air (excised)
PREFERRED_SPLIT_SILENCE_SECS = 15 # Silences >= 15s = good split points


def get_audio_duration(audio_path):
    """Get audio duration in seconds using ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
        capture_output=True, text=True
    )
    return float(result.stdout.strip())


def detect_silence(audio_path):
    """Detect silence regions in audio using ffmpeg silencedetect. Returns list of (start_secs, end_secs)."""
    cmd = [
        "ffmpeg", "-i", audio_path,
        "-af", f"silencedetect=noise={SILENCE_NOISE_THRESH_DB}dB:d={SILENCE_MIN_DURATION_SECS}",
        "-f", "null", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    stderr = result.stderr

    silences = []
    current_start = None
    for line in stderr.split("\n"):
        if "silence_start:" in line:
            match = re.search(r'silence_start:\s*([\d.]+)', line)
            if match:
                current_start = float(match.group(1))
        elif "silence_end:" in line and current_start is not None:
            match = re.search(r'silence_end:\s*([\d.]+)', line)
            if match:
                silences.append((current_start, float(match.group(1))))
                current_start = None

    # Handle edge case: file ends during silence (no silence_end)
    if current_start is not None:
        duration = get_audio_duration(audio_path)
        silences.append((current_start, duration))

    return silences


def merge_adjacent_silences(silences, gap_threshold=SILENCE_MERGE_GAP_SECS):
    """Merge silence regions separated by less than gap_threshold seconds."""
    if not silences:
        return []

    merged = [silences[0]]
    for start, end in silences[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end < gap_threshold:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged


def classify_silences(silences):
    """Classify silences into significant (excise) and preferred split points.

    Returns (significant, preferred_splits) where:
    - significant: silences >= SIGNIFICANT_SILENCE_SECS (dead air / recesses to excise)
    - preferred_splits: silences >= PREFERRED_SPLIT_SILENCE_SECS but < SIGNIFICANT_SILENCE_SECS (split points)
    """
    significant = []
    preferred_splits = []
    for start, end in silences:
        duration = end - start
        if duration >= SIGNIFICANT_SILENCE_SECS:
            significant.append((start, end))
        elif duration >= PREFERRED_SPLIT_SILENCE_SECS:
            preferred_splits.append((start, end))
    return significant, preferred_splits


def split_audio_fixed(audio_path, chunk_dir):
    """Split audio into fixed-duration chunks. Fallback when no significant silences found. Returns list of (chunk_path, start_secs)."""
    duration = get_audio_duration(audio_path)
    print(f"Audio duration: {duration:.0f}s ({duration/3600:.1f}h)")

    if duration <= CHUNK_DURATION_SECS + 60:  # Don't bother splitting if only slightly over
        return [(audio_path, 0)]

    chunks = []
    start = 0
    chunk_num = 0
    while start < duration:
        chunk_num += 1
        chunk_path = os.path.join(chunk_dir, f"chunk_{chunk_num}.mp3")
        end = min(start + CHUNK_DURATION_SECS + CHUNK_OVERLAP_SECS, duration)

        print(f"Splitting chunk {chunk_num}: {start:.0f}s - {end:.0f}s")
        cmd = [
            "ffmpeg", "-y", "-i", audio_path,
            "-ss", str(start), "-to", str(end),
            "-acodec", "copy", chunk_path,
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        chunks.append((chunk_path, start))
        start += CHUNK_DURATION_SECS  # Next chunk starts without overlap

    print(f"Split into {len(chunks)} chunks")
    return chunks


def find_best_split_point(segment_start, segment_end, preferred_splits, target_duration=CHUNK_DURATION_SECS):
    """Find the best silence-based split point near target_duration into a segment."""
    target = segment_start + target_duration
    best = None
    best_dist = float("inf")
    for s_start, s_end in preferred_splits:
        midpoint = (s_start + s_end) / 2
        if segment_start < midpoint < segment_end:
            dist = abs(midpoint - target)
            if dist < best_dist:
                best = midpoint
                best_dist = dist
    return best


def split_audio_at_silence(audio_path, chunk_dir):
    """Split audio using silence detection to excise dead air and find natural split points.

    Returns (chunks, silence_markers) where:
    - chunks: list of (chunk_path, start_secs) same format as split_audio_fixed
    - silence_markers: list of (timestamp_secs, duration_secs) for significant silences
    """
    duration = get_audio_duration(audio_path)
    print(f"Audio duration: {duration:.0f}s ({duration/3600:.1f}h)")

    # Detect and process silences
    print("Detecting silence regions...")
    raw_silences = detect_silence(audio_path)
    print(f"  Found {len(raw_silences)} raw silence regions")

    merged_silences = merge_adjacent_silences(raw_silences)
    print(f"  After merging: {len(merged_silences)} silence regions")

    significant, preferred_splits = classify_silences(merged_silences)

    if significant:
        total_dead_air = sum(end - start for start, end in significant)
        print(f"  Significant silences (>= {SIGNIFICANT_SILENCE_SECS}s): {len(significant)} "
              f"totaling {total_dead_air:.0f}s ({total_dead_air/60:.1f} min)")
        for s_start, s_end in significant:
            s_dur = s_end - s_start
            print(f"    {s_start/60:.1f}m - {s_end/60:.1f}m ({s_dur/60:.1f} min)")
    else:
        print("  No significant silences found — falling back to fixed splitting")
        return split_audio_fixed(audio_path, chunk_dir), []

    print(f"  Preferred split points (>= {PREFERRED_SPLIT_SILENCE_SECS}s): {len(preferred_splits)}")

    # Compute speech segments by subtracting significant silences from full duration
    speech_segments = []
    prev_end = 0
    for s_start, s_end in significant:
        if s_start > prev_end:
            speech_segments.append((prev_end, s_start))
        prev_end = s_end
    if prev_end < duration:
        speech_segments.append((prev_end, duration))

    print(f"  Speech segments: {len(speech_segments)}")
    for seg_start, seg_end in speech_segments:
        seg_dur = seg_end - seg_start
        print(f"    {seg_start/60:.1f}m - {seg_end/60:.1f}m ({seg_dur/60:.1f} min)")

    # Build silence markers for transcript injection
    silence_markers = []
    for s_start, s_end in significant:
        silence_markers.append((s_start, s_end - s_start))

    # Extract chunks from speech segments
    chunks = []
    chunk_num = 0
    for seg_start, seg_end in speech_segments:
        seg_duration = seg_end - seg_start

        if seg_duration <= CHUNK_DURATION_SECS + 60:
            # Short enough for one chunk
            chunk_num += 1
            chunk_path = os.path.join(chunk_dir, f"chunk_{chunk_num}.mp3")
            print(f"  Extracting chunk {chunk_num}: {seg_start:.0f}s - {seg_end:.0f}s "
                  f"({seg_duration/60:.1f} min, single segment)")
            cmd = [
                "ffmpeg", "-y", "-i", audio_path,
                "-ss", str(seg_start), "-to", str(seg_end),
                "-acodec", "copy", chunk_path,
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            chunks.append((chunk_path, seg_start))
        else:
            # Split long segment at preferred silence boundaries
            pos = seg_start
            while pos < seg_end:
                remaining = seg_end - pos
                if remaining <= CHUNK_DURATION_SECS + 60:
                    # Remaining fits in one chunk
                    chunk_end = seg_end
                else:
                    # Find a good split point
                    split_point = find_best_split_point(pos, seg_end, preferred_splits)
                    if split_point and 300 <= (split_point - pos) <= CHUNK_DURATION_SECS * 2:  # At least 5 min, at most 2x target
                        chunk_end = split_point
                    else:
                        # Fall back to fixed duration
                        chunk_end = min(pos + CHUNK_DURATION_SECS, seg_end)

                chunk_num += 1
                chunk_path = os.path.join(chunk_dir, f"chunk_{chunk_num}.mp3")
                chunk_dur = chunk_end - pos
                print(f"  Extracting chunk {chunk_num}: {pos:.0f}s - {chunk_end:.0f}s "
                      f"({chunk_dur/60:.1f} min)")
                cmd = [
                    "ffmpeg", "-y", "-i", audio_path,
                    "-ss", str(pos), "-to", str(chunk_end),
                    "-acodec", "copy", chunk_path,
                ]
                subprocess.run(cmd, capture_output=True, check=True)
                chunks.append((chunk_path, pos))
                pos = chunk_end

    print(f"Split into {len(chunks)} chunks (excised {len(significant)} silence regions)")
    return chunks, silence_markers


def inject_silence_markers(transcript, silence_markers):
    """Insert [RECESS - X minutes] markers into transcript at the correct positions.

    Each silence_marker is (timestamp_secs, duration_secs). The marker is inserted
    between the last transcript line before the silence and the first line after it.
    """
    if not silence_markers:
        return transcript

    lines = transcript.strip().split("\n")
    result_lines = []
    marker_idx = 0

    for line in lines:
        # Before adding this line, check if any silence markers belong before it
        ts_match = re.match(r'\[(\d+):(\d+):(\d+)\]', line)
        if ts_match and marker_idx < len(silence_markers):
            h, m, s = int(ts_match.group(1)), int(ts_match.group(2)), int(ts_match.group(3))
            line_time = h * 3600 + m * 60 + s

            while marker_idx < len(silence_markers):
                marker_time, marker_duration = silence_markers[marker_idx]
                if marker_time <= line_time:
                    # Insert this marker before the current line
                    marker_h = int(marker_time // 3600)
                    marker_m = int((marker_time % 3600) // 60)
                    marker_s = int(marker_time % 60)
                    dur_minutes = marker_duration / 60
                    if dur_minutes >= 1:
                        dur_str = f"{dur_minutes:.0f} minutes"
                    else:
                        dur_str = f"{marker_duration:.0f} seconds"
                    result_lines.append(
                        f"[{marker_h:02d}:{marker_m:02d}:{marker_s:02d}] [RECESS — {dur_str}]"
                    )
                    marker_idx += 1
                else:
                    break

        result_lines.append(line)

    # Append any remaining markers at the end
    while marker_idx < len(silence_markers):
        marker_time, marker_duration = silence_markers[marker_idx]
        marker_h = int(marker_time // 3600)
        marker_m = int((marker_time % 3600) // 60)
        marker_s = int(marker_time % 60)
        dur_minutes = marker_duration / 60
        if dur_minutes >= 1:
            dur_str = f"{dur_minutes:.0f} minutes"
        else:
            dur_str = f"{marker_duration:.0f} seconds"
        result_lines.append(
            f"[{marker_h:02d}:{marker_m:02d}:{marker_s:02d}] [RECESS — {dur_str}]"
        )
        marker_idx += 1

    return "\n".join(result_lines)


def build_transcription_prompt(hearing_context, chunk_index=None, total_chunks=None, previous_summary=None):
    """Build the transcription prompt with hearing context and progressive summarization."""
    context_lines = []
    if hearing_context:
        if hearing_context.get("title"):
            context_lines.append(f"Hearing title: {hearing_context['title']}")
        if hearing_context.get("committee"):
            context_lines.append(f"Committee: {hearing_context['committee']}")
        if hearing_context.get("subcommittee"):
            context_lines.append(f"Subcommittee: {hearing_context['subcommittee']}")
        if hearing_context.get("date"):
            context_lines.append(f"Hearing date: {hearing_context['date']}")
        if hearing_context.get("congress"):
            context_lines.append(f"Congress: {hearing_context['congress']}")
        if hearing_context.get("bills"):
            context_lines.append(f"Bills referenced: {', '.join(hearing_context['bills'])}")
        if hearing_context.get("description"):
            context_lines.append(f"Hearing purpose: {hearing_context['description']}")
        if hearing_context.get("witnesses"):
            context_lines.append("Witnesses: " + "; ".join(hearing_context["witnesses"]))
        if hearing_context.get("members"):
            context_lines.append("Committee members who may speak:")
            for m in hearing_context["members"]:
                context_lines.append(f"  - {m}")

    context_block = ""
    if context_lines:
        context_block = (
            "HEARING METADATA (use this to correctly identify speakers and stay grounded):\n"
            + "\n".join(context_lines)
            + "\n\n"
        )

    summary_block = ""
    if previous_summary:
        summary_block = (
            "CONTEXT FROM PREVIOUS CHUNKS (what has happened in the hearing so far):\n"
            + previous_summary.strip()
            + "\n\n"
        )

    chunk_note = ""
    if chunk_index is not None and total_chunks and total_chunks > 1:
        chunk_note = f"This is part {chunk_index} of {total_chunks} of the hearing audio. "

    # Build date/congress grounding string for anti-hallucination rules
    grounding_ref = ""
    if hearing_context:
        congress = hearing_context.get("congress")
        date = hearing_context.get("date")
        if congress and date:
            grounding_ref = f" This hearing is from the {congress} ({date})."
        elif congress:
            grounding_ref = f" This hearing is from the {congress}."
        elif date:
            grounding_ref = f" This hearing is from {date}."

    prompt = (
        f"{context_block}"
        f"{summary_block}"
        f"{chunk_note}"
        "This is audio from a US congressional hearing. Produce a complete, verbatim transcript. "
        "Format: start each speaker turn on a new line as [HH:MM:SS] SPEAKER NAME: text. "
        "Always use HH:MM:SS format for timestamps (e.g., [00:05:30], [01:12:45]). "
        "Use the hearing metadata above to correctly identify speakers. "
        "The Chair listed above is the person who opens the hearing, recognizes other members, "
        "and manages procedural matters — label them by name (e.g., 'BILL HUIZENGA' if the Chair is Bill Huizenga). "
        "The Ranking Member is the senior minority member. The witness testifies and answers questions. "
        "Match voices to the listed members based on context (who is recognized to speak, how they are addressed). "
        "For any speaker you truly cannot identify, use UNKNOWN SPEAKER. "
        "CRITICAL ACCURACY RULES: "
        "1. Be EXACTLY verbatim — transcribe every word as spoken. Do NOT paraphrase, summarize, or rephrase. "
        "If someone says 'I think that what we need to do is to make sure that we are' then write exactly that, "
        "not a cleaned-up version. "
        "2. OMIT all filler words: um, uh, ums, uhs, uh-huh, mm-hmm, yeah (when used as filler). "
        "Do NOT transcribe silence, background noise, or throat-clearing. "
        "3. During rapid exchanges (e.g., Q&A between members and witnesses), pay extra attention. "
        "Transcribe each person's words completely and accurately. Do NOT skip or garble words "
        "when speakers talk quickly or interrupt each other. If you're unsure of a word, use your best guess "
        "rather than skipping it or inserting a wrong word. "
        "4. Transcribe EVERY word of actual testimony and questions — do not abbreviate or truncate. "
        "Combine each speaker's continuous remarks into one paragraph per turn. "
        "ANTI-HALLUCINATION RULES: "
        "5. If you hear silence, dead air, or unintelligible audio, write [INAUDIBLE] or [SILENCE]. "
        "Do NOT fabricate or guess at content that is not clearly audible. "
        "6. Do NOT generate content from your training data. Only transcribe words you actually hear in the audio."
        f"{grounding_ref} Do NOT transcribe content from other hearings or legislative sessions. "
        "7. If audio is unclear or garbled, write [INAUDIBLE - approximately X seconds] rather than guessing. "
        "8. If you find yourself repeating similar phrases or patterns, STOP and re-listen to the audio carefully. "
        "Do NOT include any preamble — start directly with the first speaker turn."
    )
    return prompt


def transcribe_chunk(audio_path, prompt):
    """Transcribe a single audio file. Returns transcript text."""
    cmd = [
        "llm",
        "-m", TRANSCRIBE_MODEL,
        "-a", audio_path,
        prompt,
    ]

    transcript_lines = []
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Show a spinner while waiting for the first output (uploading/processing)
    spinner_stop = threading.Event()
    def spin():
        phases = ["Uploading audio and waiting for model", "Processing audio", "Still processing"]
        spinner_chars = itertools.cycle(["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])
        start = time.time()
        phase_idx = 0
        while not spinner_stop.is_set():
            elapsed = time.time() - start
            if elapsed > 60 and phase_idx < 2:
                phase_idx = 2
            elif elapsed > 15 and phase_idx < 1:
                phase_idx = 1
            char = next(spinner_chars)
            msg = f"\r{char} {phases[phase_idx]}... ({elapsed:.0f}s)"
            sys.stdout.write(msg)
            sys.stdout.flush()
            spinner_stop.wait(0.1)
        # Clear the spinner line
        sys.stdout.write("\r" + " " * 80 + "\r")
        sys.stdout.flush()

    spinner_thread = threading.Thread(target=spin, daemon=True)
    spinner_thread.start()

    first_line = True
    for line in proc.stdout:
        if first_line:
            spinner_stop.set()
            spinner_thread.join()
            first_line = False
        sys.stdout.write(line)
        sys.stdout.flush()
        transcript_lines.append(line)

    if first_line:
        # No output at all — stop spinner
        spinner_stop.set()
        spinner_thread.join()

    proc.wait()

    if proc.returncode != 0:
        print(f"Error from llm: {proc.stderr.read()}", file=sys.stderr)
        sys.exit(1)

    return "".join(transcript_lines)


def fix_chunk_timestamps(transcript, offset_secs):
    """Adjust timestamps in a chunk transcript by adding the chunk's start offset."""
    if offset_secs == 0:
        return transcript

    def adjust_timestamp(match):
        time_str = match.group(1)
        parts = time_str.split(":")
        if len(parts) == 3:
            total_secs = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2]) + offset_secs
        elif len(parts) == 2:
            total_secs = int(parts[0]) * 60 + int(parts[1]) + offset_secs
        else:
            return match.group(0)
        hours = int(total_secs // 3600)
        minutes = int((total_secs % 3600) // 60)
        seconds = int(total_secs % 60)
        return f"[{hours:02d}:{minutes:02d}:{seconds:02d}]"

    return re.sub(r'\[(\d+:\d+:\d+)\]', adjust_timestamp, transcript)


def summarize_chunk(chunk_transcript, cache_dir=None, chunk_index=None):
    """Generate a brief summary of a transcript chunk for progressive context."""
    # Check cache first
    if cache_dir and chunk_index is not None:
        cached = os.path.join(cache_dir, f"chunk_{chunk_index}_summary.txt")
        if os.path.exists(cached):
            with open(cached) as f:
                return f.read()

    prompt = (
        "Summarize this congressional hearing transcript chunk in 3-4 sentences for context. "
        "Focus on: who spoke, what bills/amendments were discussed, key votes or procedural actions, "
        "and where the discussion left off. This summary will be used to provide context for "
        "transcribing the next chunk of audio.\n\n"
        f"TRANSCRIPT CHUNK:\n{chunk_transcript}"
    )

    cmd = [
        "llm",
        "-m", SUMMARY_MODEL,
        prompt,
    ]

    print(f"  Generating chunk summary...")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"  Warning: chunk summary failed: {proc.stderr}", file=sys.stderr)
        return ""

    summary = proc.stdout.strip()

    # Cache the summary
    if cache_dir and chunk_index is not None:
        with open(os.path.join(cache_dir, f"chunk_{chunk_index}_summary.txt"), "w") as f:
            f.write(summary)

    return summary


def stitch_transcripts(all_transcripts):
    """Stitch chunk transcripts together, deduplicating overlap by timestamp."""
    if not all_transcripts:
        return ""

    result = all_transcripts[0]

    for i in range(1, len(all_transcripts)):
        next_chunk = all_transcripts[i]

        # Find the last timestamp in the accumulated result
        prev_timestamps = re.findall(r'\[(\d+):(\d+):(\d+)\]', result)
        if prev_timestamps:
            last_h, last_m, last_s = [int(x) for x in prev_timestamps[-1]]
            last_total = last_h * 3600 + last_m * 60 + last_s

            # Skip lines in next chunk whose timestamps fall within the overlap
            next_lines = next_chunk.strip().split("\n")
            first_new_line = 0
            for j, line in enumerate(next_lines):
                ts_match = re.match(r'\[(\d+):(\d+):(\d+)\]', line)
                if ts_match:
                    h, m, s = int(ts_match.group(1)), int(ts_match.group(2)), int(ts_match.group(3))
                    total = h * 3600 + m * 60 + s
                    if total <= last_total:
                        first_new_line = j + 1
                    else:
                        break

            if first_new_line > 0:
                print(f"  Overlap: skipped {first_new_line} duplicate lines from chunk {i+1}")
                next_chunk = "\n".join(next_lines[first_new_line:])

        result += "\n" + next_chunk

    return result


def check_transcript_quality(transcript):
    """Check transcript for hallucination artifacts. Returns list of warning strings."""
    issues = []
    lines = transcript.strip().split("\n")

    # 1. Word salad detection — high ratio of filler/repeated short words
    filler_words = {
        "wait", "what", "these", "exactly", "any", "our", "those", "look",
        "right", "actually", "all", "the", "out", "an", "this", "they",
    }
    for line in lines:
        words = line.lower().split()
        if len(words) > 80:
            filler_count = sum(1 for w in words if w in filler_words)
            filler_ratio = filler_count / len(words)
            if filler_ratio > 0.4:
                ts_match = re.match(r'\[[\d:]+\]', line)
                ts = ts_match.group(0) if ts_match else "unknown"
                speaker_match = re.match(r'\[[\d:]+\]\s*([^:]+):', line)
                speaker = speaker_match.group(1).strip() if speaker_match else "unknown"
                issues.append(
                    f"  WORD SALAD at {ts} ({speaker}): "
                    f"{filler_ratio:.0%} filler words in {len(words)}-word block"
                )

    # 2. Duplicate block detection — same speaker text appearing twice
    seen_blocks = {}
    for line in lines:
        if len(line) < 200:
            continue
        normalized = re.sub(r'\[[\d:]+\]', '', line).strip()[:300]
        if normalized in seen_blocks:
            ts_match = re.match(r'\[[\d:]+\]', line)
            ts = ts_match.group(0) if ts_match else "unknown"
            issues.append(
                f"  DUPLICATE BLOCK at {ts} (same content as {seen_blocks[normalized]})"
            )
        else:
            ts_match = re.match(r'\[[\d:]+\]', line)
            ts = ts_match.group(0) if ts_match else "unknown"
            seen_blocks[normalized] = ts

    # 3. Timestamp regression — non-monotonic timestamps (sign of bad stitching)
    prev_total = -1
    for line in lines:
        ts_match = re.match(r'\[(\d+):(\d+):(\d+)\]', line)
        if ts_match:
            h, m, s = int(ts_match.group(1)), int(ts_match.group(2)), int(ts_match.group(3))
            total = h * 3600 + m * 60 + s
            if total < prev_total - 120:
                issues.append(
                    f"  TIMESTAMP REGRESSION: [{h:02d}:{m:02d}:{s:02d}] after "
                    f"[{prev_total//3600:02d}:{(prev_total%3600)//60:02d}:{prev_total%60:02d}]"
                )
            prev_total = total

    return issues


def transcribe_audio(audio_path, hearing_context=None, cache_dir=None):
    """Transcribe audio using llm + Gemini, splitting into chunks if needed. Returns transcript text."""
    print("Transcribing audio with Gemini...")

    if hearing_context:
        print(f"Injecting hearing context: {len(hearing_context.get('members', []))} members, "
              f"{len(hearing_context.get('witnesses', []))} witness lines")

    # Check if we need to split
    duration = get_audio_duration(audio_path)

    if duration <= CHUNK_DURATION_SECS + 60:
        # Single chunk — no splitting needed
        if cache_dir:
            cached = os.path.join(cache_dir, "chunk_1.txt")
            if os.path.exists(cached):
                print("Using cached transcription (single chunk)")
                with open(cached) as f:
                    return f.read()

        prompt = build_transcription_prompt(hearing_context)
        transcript = transcribe_chunk(audio_path, prompt)
        print(f"\n\nTranscription complete")

        if cache_dir:
            with open(os.path.join(cache_dir, "chunk_1.txt"), "w") as f:
                f.write(transcript)

        return transcript

    # Split into chunks using silence-aware splitting
    chunk_dir = os.path.join(os.path.dirname(audio_path) or "/tmp", "chunks")
    os.makedirs(chunk_dir, exist_ok=True)
    chunks, silence_markers = split_audio_at_silence(audio_path, chunk_dir)

    all_transcripts = []
    running_summary = ""
    for i, (chunk_path, start_secs) in enumerate(chunks, 1):
        # Check cache for this chunk
        if cache_dir:
            cached = os.path.join(cache_dir, f"chunk_{i}.txt")
            if os.path.exists(cached):
                print(f"\n--- Chunk {i}/{len(chunks)} found in cache, skipping ---")
                with open(cached) as f:
                    chunk_transcript = f.read()
                all_transcripts.append(chunk_transcript)
                # Load cached summary to maintain context chain
                chunk_summary = summarize_chunk(chunk_transcript, cache_dir, i)
                if chunk_summary:
                    running_summary += f"\n[Chunk {i}] {chunk_summary}"
                continue

        print(f"\n--- Transcribing chunk {i}/{len(chunks)} (starts at {start_secs:.0f}s) ---")
        prompt = build_transcription_prompt(
            hearing_context, i, len(chunks),
            previous_summary=running_summary if running_summary else None
        )
        chunk_transcript = transcribe_chunk(chunk_path, prompt)

        # Fix timestamps: offset from chunk-relative to absolute hearing time
        chunk_transcript = fix_chunk_timestamps(chunk_transcript, start_secs)

        all_transcripts.append(chunk_transcript)

        # Save to cache immediately so progress isn't lost (with corrected timestamps)
        if cache_dir:
            with open(os.path.join(cache_dir, f"chunk_{i}.txt"), "w") as f:
                f.write(chunk_transcript)
            print(f"Chunk {i} cached.")

        # Generate summary for progressive context (used by subsequent chunks)
        chunk_summary = summarize_chunk(chunk_transcript, cache_dir, i)
        if chunk_summary:
            running_summary += f"\n[Chunk {i}] {chunk_summary}"

    # Stitch together with timestamp-based overlap deduplication
    transcript = stitch_transcripts(all_transcripts)

    # Inject recess/silence markers into the stitched transcript
    if silence_markers:
        transcript = inject_silence_markers(transcript, silence_markers)
        print(f"Injected {len(silence_markers)} recess marker(s) into transcript")

    print(f"\n\nTranscription complete ({len(chunks)} chunks stitched)")
    return transcript


def run_llm_on_text(prompt_template, transcript, label):
    """Run an LLM prompt on transcript text. Returns response text."""
    print(f"\nGenerating {label}...")

    prompt = prompt_template.format(transcript=transcript)

    cmd = [
        "llm",
        "-m", SUMMARY_MODEL,
        prompt,
    ]

    result_lines = []
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        result_lines.append(line)
    proc.wait()

    if proc.returncode != 0:
        print(f"Error from llm: {proc.stderr.read()}", file=sys.stderr)
        sys.exit(1)

    result = "".join(result_lines)
    print(f"\n{label} complete.")
    return result


def slug_from_url(url):
    """Extract a slug from the hearing URL for the output filename."""
    slug = url.rstrip("/").split("/")[-1]
    slug = re.sub(r'[^a-zA-Z0-9-]', '-', slug)
    return slug


def format_transcript_for_markdown(transcript):
    """Format raw transcript for readable markdown.

    Bolds the [timestamp] SPEAKER: prefix and ensures blank lines between turns.
    """
    lines = transcript.strip().split("\n")
    formatted = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Format recess/silence markers in italic
        if re.match(r'\[[\d:]+\]\s*\[(?:RECESS|SILENCE)', line):
            formatted.append(f"*{line}*")
            continue
        # Match lines like: [00:12:34] SPEAKER NAME: text...
        m = re.match(r'(\[[\d:]+\]\s+[^:]+:\s*)(.*)', line)
        if m:
            prefix = m.group(1).strip()
            body = m.group(2).strip()
            formatted.append(f"**{prefix}** {body}")
        else:
            # Non-speaker lines (rare) — just pass through
            formatted.append(line)
    return "\n\n".join(formatted)


def assemble_report(hearing_title, executive_summary, detailed_notes, transcript, output_path):
    """Assemble the final markdown report."""
    formatted_transcript = format_transcript_for_markdown(transcript)

    parts = []
    parts.append(f"# {hearing_title}\n")
    parts.append("## Executive Summary\n")
    parts.append(executive_summary.strip())
    parts.append("\n\n")
    parts.append("## Detailed Notes\n")
    parts.append(detailed_notes.strip())
    parts.append("\n\n")
    parts.append("## Transcript\n")
    parts.append(formatted_transcript)
    parts.append("\n")

    report = "\n".join(parts)

    with open(output_path, "w") as f:
        f.write(report)

    print(f"\nReport saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Transcribe and analyze a congressional hearing.")
    parser.add_argument("url", help="URL of the hearing page")
    parser.add_argument("-o", "--output", help="Output markdown file path")
    parser.add_argument("--youtube-id", help="Skip scraping and use this YouTube video ID directly")
    parser.add_argument("--title", help="Override hearing title")
    parser.add_argument("--keep-audio", action="store_true", help="Copy the audio file to the current directory")
    parser.add_argument("--transcript-only", action="store_true", help="Only transcribe, skip analysis")
    parser.add_argument("--no-cache", action="store_true",
                        help="Clear cache and start fresh (results are still cached for future runs)")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Clear cached results for this hearing and exit")
    args = parser.parse_args()

    slug = slug_from_url(args.url)

    if args.clear_cache:
        clear_cache(slug)
        return

    cache_dir = get_cache_dir(slug)

    if args.no_cache:
        clear_cache(slug)
        cache_dir = get_cache_dir(slug)
        print("Starting fresh (cache cleared)")

    print(f"Cache directory: {cache_dir}")

    # Step 1: Get YouTube video ID, title, and hearing context
    hearing_context = {}
    if args.youtube_id:
        video_id = args.youtube_id
        hearing_title = args.title or slug.replace("-", " ").title()
        print(f"Using provided YouTube ID: {video_id}")
        # Still try to scrape context even with manual youtube-id
        try:
            _, _, hearing_context = fetch_hearing_info(args.url)
        except Exception:
            pass
    else:
        video_id, scraped_title, hearing_context = fetch_hearing_info(args.url)
        hearing_title = args.title or scraped_title or slug.replace("-", " ").title()
        if not video_id:
            print("Could not find a YouTube video on the hearing page.", file=sys.stderr)
            print("Try providing the YouTube video ID directly with --youtube-id", file=sys.stderr)
            sys.exit(1)
        print(f"Found YouTube video ID: {video_id}")

    print(f"Hearing title: {hearing_title}")

    output_path = args.output or f"{slug}.md"

    # Step 2: Download audio
    cached_audio = os.path.join(cache_dir, "audio.mp3")
    if os.path.exists(cached_audio):
        print(f"Using cached audio ({os.path.getsize(cached_audio) / (1024*1024):.1f} MB)")
        audio_path = cached_audio
    else:
        audio_path = download_audio(video_id, cache_dir)

    # Step 3: Transcribe
    cached_transcript = os.path.join(cache_dir, "transcript.txt")
    if os.path.exists(cached_transcript):
        print("Using cached transcript")
        with open(cached_transcript) as f:
            transcript = f.read()
    else:
        transcript = transcribe_audio(audio_path, hearing_context, cache_dir)
        with open(cached_transcript, "w") as f:
            f.write(transcript)
        print("Full transcript cached.")

    # Step 3b: Quality control
    quality_issues = check_transcript_quality(transcript)
    if quality_issues:
        print(f"\n{'='*60}")
        print(f"TRANSCRIPT QUALITY CHECK: {len(quality_issues)} issue(s) found")
        print(f"{'='*60}")
        for issue in quality_issues:
            print(issue)
        print(f"{'='*60}")
        print("Tip: delete problem chunk files from the cache dir and re-run to re-transcribe them.")
        print(f"Cache dir: {cache_dir}")
        print()
    else:
        print("Quality check passed — no issues detected.")

    if args.keep_audio:
        kept_path = f"{slug}.mp3"
        if not os.path.exists(kept_path):
            shutil.copy2(audio_path, kept_path)
            print(f"Audio copied to: {kept_path}")

    if args.transcript_only:
        with open(output_path, "w") as f:
            f.write(f"# {hearing_title}\n\n## Transcript\n\n{transcript.strip()}\n")
        print(f"\nTranscript saved to: {output_path}")
        print("Done!")
        return

    # Step 4: Generate executive summary
    cached_summary = os.path.join(cache_dir, "executive_summary.txt")
    if os.path.exists(cached_summary):
        print("Using cached executive summary")
        with open(cached_summary) as f:
            executive_summary = f.read()
    else:
        executive_summary = run_llm_on_text(EXECUTIVE_SUMMARY_PROMPT, transcript, "Executive Summary")
        with open(cached_summary, "w") as f:
            f.write(executive_summary)

    # Step 5: Generate detailed notes
    cached_notes = os.path.join(cache_dir, "detailed_notes.txt")
    if os.path.exists(cached_notes):
        print("Using cached detailed notes")
        with open(cached_notes) as f:
            detailed_notes = f.read()
    else:
        detailed_notes = run_llm_on_text(DETAILED_NOTES_PROMPT, transcript, "Detailed Notes")
        with open(cached_notes, "w") as f:
            f.write(detailed_notes)

    # Step 6: Assemble final report
    assemble_report(hearing_title, executive_summary, detailed_notes, transcript, output_path)

    print("Done!")


if __name__ == "__main__":
    main()
