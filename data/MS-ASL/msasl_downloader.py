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
SAVETO = HERE / "raw_videos"
YT_DLP = Path(sys.executable).parent / "Scripts" / "yt-dlp.exe"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

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


def download_clip(sample: dict) -> None:
    url = sample['url'] if sample['url'].startswith('http') else 'https://' + sample['url']
    yt_id = extract_yt_id(url)
    if not yt_id:
        logging.warning('Could not parse YouTube id from {}'.format(url))
        return

    saveto = SAVETO / '{}_{}.mp4'.format(yt_id, int(sample['start_time'] * 1000))
    if saveto.exists():
        logging.info('{} exists at {}'.format(saveto.name, saveto))
        return

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
        logging.error('Unsuccessful downloading - {} [{}-{}]: {}'.format(
            url, sample['start_time'], sample['end_time'], result.stderr[-500:]))
        return

    logging.info('Finished {}'.format(saveto.name))
    time.sleep(random.uniform(0.5, 1.5))


def _self_check():
    assert extract_yt_id('https://www.youtube.com/watch?v=C37R_Ix8-qs') == 'C37R_Ix8-qs'
    assert extract_yt_id('www.youtube.com/watch?v=9FdHlMOnVjg') == '9FdHlMOnVjg'
    assert extract_yt_id('https://youtu.be/dQw4w9WgXcQ') == 'dQw4w9WgXcQ'
    assert extract_yt_id('not a url') is None

    expanded = expand_with_synonyms({'father', 'unrelated_word'})
    assert {'father', 'dad', 'daddy', 'unrelated_word'} <= expanded
    assert 'mother' not in expanded  # only groups touching vocab get pulled in

    print('self-check passed: YouTube id extraction handles both URL shapes and missing scheme, synonym expansion pulls in whole groups')


if __name__ == '__main__':
    if '--self-test' in sys.argv:
        _self_check()
        sys.exit(0)

    SAVETO.mkdir(exist_ok=True)
    samples = load_filtered_samples()
    logging.info('{} MS-ASL samples match the {}-word vocab'.format(
        len(samples), len(set(VOCAB_PATH.read_text().splitlines()))))

    for sample in samples:
        download_clip(sample)
