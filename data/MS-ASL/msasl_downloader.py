"""Download MS-ASL clips for the words in vocab.txt.

Each MS-ASL sample is a {start_time, end_time} range inside a longer YouTube
video, not its own hosted file (unlike WLASL). yt-dlp's --download-sections
pulls just that range instead of the whole source video, and
--force-keyframes-at-cuts (needs ffmpeg) re-encodes at the cut points so a
short 2-3s sign clip doesn't balloon out to the nearest keyframe boundary.
"""
import json
import logging
import os
import re
import subprocess
import sys
import time
import random
from pathlib import Path

import imageio_ffmpeg

HERE = Path(__file__).parent
VOCAB_PATH = HERE.parent.parent / "vocab.txt"
SAVETO = HERE.parent / "raw_videos"
YT_DLP = Path(sys.executable).parent / "Scripts" / "yt-dlp.exe"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

# yt-dlp error text for videos that are gone for good -- retrying never helps,
# unlike a bot-check or rate-limit block which can clear on its own.
PERMANENT_FAILURE_MARKERS = (
    'Video unavailable',
    'This video is unavailable',
    'Private video',
    'has been removed',
    'account associated with this video has been terminated',
)
DEAD_YT_IDS_PATH = HERE / 'dead_youtube_ids.txt'

# Retry pacing: back off between passes so a bot-check block gets time to
# clear instead of being hammered every few seconds.
RETRY_BACKOFF_START = 120
RETRY_BACKOFF_MAX = 900
MAX_NO_PROGRESS_PASSES = 10

logging.basicConfig(filename=str(HERE / 'msasl_download_{}.log'.format(int(time.time()))),
                     filemode='w', level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def expand_with_synonyms(vocab: set) -> set:
    """Grow the vocab with MSASL_synonym.json groups: if any word in a group
    is in vocab (e.g. 'father'), pull in the rest too ('dad', 'daddy') so
    clips labeled with the synonym aren't skipped.
    """
    expanded = set(vocab)
    for group in json.loads((HERE / 'MSASL_synonym.json').read_text()):
        if vocab & set(group):
            expanded |= set(group)
    return expanded


def load_filtered_samples():
    vocab = expand_with_synonyms(set(VOCAB_PATH.read_text().splitlines()))
    samples = []
    for fn in ('MSASL_train.json', 'MSASL_val.json', 'MSASL_test.json'):
        for e in json.loads((HERE / fn).read_text()):
            if e['text'] in vocab:
                samples.append(e)
    return samples


def extract_yt_id(url: str) -> str | None:
    m = re.search(r'[?&]v=([\w-]{11})', url) or re.search(r'youtu\.be/([\w-]{11})', url)
    return m.group(1) if m else None


def is_permanent_failure(stderr: str) -> bool:
    return any(marker in stderr for marker in PERMANENT_FAILURE_MARKERS)


def load_dead_yt_ids() -> set:
    if DEAD_YT_IDS_PATH.exists():
        return set(DEAD_YT_IDS_PATH.read_text().splitlines())
    return set()


def mark_dead(yt_id: str) -> None:
    with open(DEAD_YT_IDS_PATH, 'a') as f:
        f.write(yt_id + '\n')


def clip_path(sample: dict) -> Path | None:
    url = sample['url'] if sample['url'].startswith('http') else 'https://' + sample['url']
    yt_id = extract_yt_id(url)
    if not yt_id:
        return None
    return SAVETO / '{}_{}.mp4'.format(yt_id, int(sample['start_time'] * 1000))


def iter_pending(samples, dead_ids: set):
    """Yield samples not already downloaded and not confirmed permanently
    unavailable.
    """
    for sample in samples:
        url = sample['url'] if sample['url'].startswith('http') else 'https://' + sample['url']
        yt_id = extract_yt_id(url)
        if not yt_id or yt_id in dead_ids:
            continue
        if clip_path(sample).exists():
            continue
        yield sample


def download_clip(sample: dict, dead_ids: set) -> bool:
    """Attempt one download. Returns True if the clip is now on disk."""
    url = sample['url'] if sample['url'].startswith('http') else 'https://' + sample['url']
    yt_id = extract_yt_id(url)
    if not yt_id:
        logging.warning('Could not parse YouTube id from {}'.format(url))
        return False

    saveto = clip_path(sample)
    section = '*{}-{}'.format(sample['start_time'], sample['end_time'])
    cmd = [
        str(YT_DLP), url,
        '--download-sections', section,
        '--force-keyframes-at-cuts',
        '-f', 'bv*+ba/b',
        '--merge-output-format', 'mp4',
        '--ffmpeg-location', str(FFMPEG),
        # ponytail: android client skips the JS challenge that triggers
        # YouTube's "sign in to confirm you're not a bot" block. If this
        # client ever gets rate-limited too, --cookies-from-browser is the
        # next fallback.
        '--extractor-args', 'youtube:player_client=android',
        '--quiet', '--no-warnings',
        '-o', str(saveto),
    ]
    logging.info('Requesting {} [{}-{}]'.format(url, sample['start_time'], sample['end_time']))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr[-500:]
        if is_permanent_failure(stderr):
            logging.warning('Permanently unavailable, will not retry - {}: {}'.format(url, stderr))
            dead_ids.add(yt_id)
            mark_dead(yt_id)
        else:
            logging.error('Unsuccessful downloading - {} [{}-{}]: {}'.format(
                url, sample['start_time'], sample['end_time'], stderr))
        return False

    logging.info('Finished {}'.format(saveto.name))
    time.sleep(random.uniform(0.5, 1.5))
    return True


def download_clips_until_done(samples):
    """Keep re-running the download pass -- with backoff between passes --
    until every sample is either downloaded or confirmed permanently
    unavailable, or MAX_NO_PROGRESS_PASSES pass in a row land zero new
    downloads (most likely an active bot-check/rate-limit block; re-run
    later).
    """
    dead_ids = load_dead_yt_ids()
    backoff = RETRY_BACKOFF_START
    no_progress_streak = 0

    while True:
        pending = list(iter_pending(samples, dead_ids))
        if not pending:
            logging.info('All MS-ASL clips downloaded or confirmed unavailable.')
            break

        downloaded = sum(download_clip(sample, dead_ids) for sample in pending)
        pending = list(iter_pending(samples, dead_ids))

        if not pending:
            logging.info('All MS-ASL clips downloaded or confirmed unavailable.')
            break

        if downloaded == 0:
            no_progress_streak += 1
        else:
            no_progress_streak = 0
            backoff = RETRY_BACKOFF_START

        if no_progress_streak >= MAX_NO_PROGRESS_PASSES:
            logging.warning(
                '{} MS-ASL clips still pending after {} passes with no progress '
                '(likely still bot-check/rate-limit blocked). Stopping for now -- '
                're-run this script later to keep going.'.format(len(pending), no_progress_streak))
            break

        logging.info('{} pending after this pass ({} downloaded). Waiting {:.0f}s before retrying.'.format(
            len(pending), downloaded, backoff))
        time.sleep(backoff)
        backoff = min(backoff * 1.5, RETRY_BACKOFF_MAX)


def _self_check():
    assert extract_yt_id('https://www.youtube.com/watch?v=C37R_Ix8-qs') == 'C37R_Ix8-qs'
    assert extract_yt_id('www.youtube.com/watch?v=9FdHlMOnVjg') == '9FdHlMOnVjg'
    assert extract_yt_id('https://youtu.be/dQw4w9WgXcQ') == 'dQw4w9WgXcQ'
    assert extract_yt_id('not a url') is None

    expanded = expand_with_synonyms({'father', 'unrelated_word'})
    assert {'father', 'dad', 'daddy', 'unrelated_word'} <= expanded
    assert 'mother' not in expanded  # only groups touching vocab get pulled in

    assert is_permanent_failure("ERROR: [youtube] xyz: Video unavailable")
    assert not is_permanent_failure("ERROR: [youtube] xyz: Sign in to confirm you're not a bot")

    print('self-check passed: YouTube id extraction handles both URL shapes and missing scheme, '
          'synonym expansion pulls in whole groups, permanent-failure detection works as expected')


if __name__ == '__main__':
    if '--self-test' in sys.argv:
        _self_check()
        sys.exit(0)

    SAVETO.mkdir(exist_ok=True)
    samples = load_filtered_samples()
    logging.info('{} MS-ASL samples match the {}-word vocab'.format(
        len(samples), len(set(VOCAB_PATH.read_text().splitlines()))))

    download_clips_until_done(samples)
