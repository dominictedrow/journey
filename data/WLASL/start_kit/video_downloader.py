import os
import json
import time
import sys
import subprocess
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import random

import imageio_ffmpeg

HERE = Path(__file__).parent
VOCAB_PATH = HERE.parent.parent.parent / "vocab.txt"
SYNONYM_PATH = HERE.parent.parent / "MS-ASL" / "MSASL_synonym.json"
YT_DLP = Path(sys.executable).parent / "Scripts" / "yt-dlp.exe"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

# Confirmed 0% success across thousands of attempts -- skip outright instead
# of wasting a request per video.
DEAD_HOSTS = {
    'www.signingsavvy.com',
    'www.handspeak.com',
    'www.aslpro.com',
    'aslsignbank.haskins.yale.edu',
}

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

# Retry pacing for the YouTube phase: back off between passes so a bot-check
# block gets time to clear instead of being hammered every few seconds.
RETRY_BACKOFF_START = 120
RETRY_BACKOFF_MAX = 900
MAX_NO_PROGRESS_PASSES = 10

import logging
logging.basicConfig(filename='download_{}.log'.format(int(time.time())), filemode='w', level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def expand_with_synonyms(vocab: set) -> set:
    """Grow the vocab with MSASL_synonym.json groups: if any word in a group
    is in vocab (e.g. 'father'), pull in the rest too ('dad', 'daddy') so
    glosses labeled with the synonym aren't skipped. The synonym groups are
    generic English word groups, not MS-ASL specific, so they apply equally
    to WLASL gloss text.
    """
    expanded = set(vocab)
    for group in json.loads(SYNONYM_PATH.read_text()):
        if vocab & set(group):
            expanded |= set(group)
    return expanded


def load_filtered_content(indexfile='WLASL_v0.3.json'):
    vocab = expand_with_synonyms(set(VOCAB_PATH.read_text().splitlines()))
    content = json.load(open(indexfile))
    return [entry for entry in content if entry['gloss'] in vocab]


def request_video(url, referer=''):
    user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'

    headers = {'User-Agent': user_agent,
               }

    if referer:
        headers['Referer'] = referer

    request = urllib.request.Request(url, None, headers)  # The assembled request

    logging.info('Requesting {}'.format(url))
    response = urllib.request.urlopen(request)
    data = response.read()  # The data you need

    return data


def save_video(data, saveto):
    with open(saveto, 'wb+') as f:
        f.write(data)

    # please be nice to the host - take pauses and avoid spamming
    time.sleep(random.uniform(0.5, 1.5))


def download_youtube(url, dirname, video_id):
    raise NotImplementedError("Urllib cannot deal with YouTube links.")


def download_aslpro(url, dirname, video_id):
    saveto = os.path.join(dirname, '{}.swf'.format(video_id))
    if os.path.exists(saveto):
        logging.info('{} exists at {}'.format(video_id, saveto))
        return

    data = request_video(url, referer='http://www.aslpro.com/cgi-bin/aslpro/aslpro.cgi')
    save_video(data, saveto)


def download_others(url, dirname, video_id):
    saveto = os.path.join(dirname, '{}.mp4'.format(video_id))
    if os.path.exists(saveto):
        logging.info('{} exists at {}'.format(video_id, saveto))
        return

    data = request_video(url)
    save_video(data, saveto)


def select_download_method(url):
    if 'aslpro' in url:
        return download_aslpro
    elif 'youtube' in url or 'youtu.be' in url:
        return download_youtube
    else:
        return download_others


def download_nonyt_videos(content, saveto='raw_videos'):
    if not os.path.exists(saveto):
        os.mkdir(saveto)

    for entry in content:
        gloss = entry['gloss']
        instances = entry['instances']

        for inst in instances:
            video_url = inst['url']
            video_id = inst['video_id']

            logging.info('gloss: {}, video: {}.'.format(gloss, video_id))

            if urlparse(video_url).netloc in DEAD_HOSTS:
                logging.warning('Skipping dead host {}'.format(video_id))
                continue

            download_method = select_download_method(video_url)

            if download_method == download_youtube:
                continue  # handled separately by download_yt_videos, below

            try:
                download_method(video_url, saveto, video_id)
            except Exception as e:
                logging.error('Unsuccessful downloading - video {}: {}'.format(video_id, e))


def is_permanent_failure(stderr: str) -> bool:
    return any(marker in stderr for marker in PERMANENT_FAILURE_MARKERS)


def load_dead_yt_ids() -> set:
    if DEAD_YT_IDS_PATH.exists():
        return set(DEAD_YT_IDS_PATH.read_text().splitlines())
    return set()


def mark_dead(yt_id: str) -> None:
    with open(DEAD_YT_IDS_PATH, 'a') as f:
        f.write(yt_id + '\n')


def iter_yt_targets(content, saveto, dead_ids):
    """Yield (video_url, yt_id) for every YouTube instance not already
    downloaded and not confirmed permanently unavailable.
    """
    for entry in content:
        for inst in entry['instances']:
            video_url = inst['url']
            if 'youtube' not in video_url and 'youtu.be' not in video_url:
                continue

            yt_id = video_url[-11:]
            if yt_id in dead_ids:
                continue
            if os.path.exists(os.path.join(saveto, yt_id + '.mp4')) or os.path.exists(os.path.join(saveto, yt_id + '.mkv')):
                continue

            yield video_url, yt_id


def download_yt_videos(content, saveto='raw_videos', dead_ids=None):
    """One pass over the pending YouTube instances. Returns how many succeeded."""
    if not os.path.exists(saveto):
        os.mkdir(saveto)
    if dead_ids is None:
        dead_ids = load_dead_yt_ids()

    downloaded = 0
    for video_url, yt_id in iter_yt_targets(content, saveto, dead_ids):
        cmd = [
            str(YT_DLP), video_url,
            '-f', 'bv*+ba/b',
            '--merge-output-format', 'mp4',
            '--ffmpeg-location', str(FFMPEG),
            # ponytail: android client skips the JS challenge that triggers
            # YouTube's "sign in to confirm you're not a bot" block. If this
            # client ever gets rate-limited too, --cookies-from-browser is the
            # next fallback.
            '--extractor-args', 'youtube:player_client=android',
            '--quiet', '--no-warnings',
            '-o', os.path.join(saveto, '%(id)s.%(ext)s'),
        ]
        logging.info('Requesting {}'.format(video_url))
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            stderr = result.stderr[-500:]
            if is_permanent_failure(stderr):
                logging.warning('Permanently unavailable, will not retry - {}: {}'.format(video_url, stderr))
                dead_ids.add(yt_id)
                mark_dead(yt_id)
            else:
                logging.error('Unsuccessful downloading - youtube video url {}: {}'.format(video_url, stderr))
            continue

        logging.info('Finish downloading youtube video url {}'.format(video_url))
        downloaded += 1
        # please be nice to the host - take pauses and avoid spamming
        time.sleep(random.uniform(0.5, 1.5))

    return downloaded


def download_yt_videos_until_done(content, saveto='raw_videos'):
    """Keep re-running the YouTube pass -- with backoff between passes -- until
    every instance is either downloaded or confirmed permanently unavailable,
    or MAX_NO_PROGRESS_PASSES pass in a row land zero new downloads (most
    likely an active bot-check block that hasn't cleared yet; re-run later).
    """
    dead_ids = load_dead_yt_ids()
    backoff = RETRY_BACKOFF_START
    no_progress_streak = 0

    while True:
        pending = sum(1 for _ in iter_yt_targets(content, saveto, dead_ids))
        if pending == 0:
            logging.info('All YouTube videos downloaded or confirmed unavailable.')
            break

        downloaded = download_yt_videos(content, saveto, dead_ids)
        pending = sum(1 for _ in iter_yt_targets(content, saveto, dead_ids))

        if pending == 0:
            logging.info('All YouTube videos downloaded or confirmed unavailable.')
            break

        if downloaded == 0:
            no_progress_streak += 1
        else:
            no_progress_streak = 0
            backoff = RETRY_BACKOFF_START

        if no_progress_streak >= MAX_NO_PROGRESS_PASSES:
            logging.warning(
                '{} YouTube videos still pending after {} passes with no progress '
                '(likely still bot-check/rate-limit blocked). Stopping for now -- '
                're-run this script later to keep going.'.format(pending, no_progress_streak))
            break

        logging.info('{} pending after this pass ({} downloaded). Waiting {:.0f}s before retrying.'.format(
            pending, downloaded, backoff))
        time.sleep(backoff)
        backoff = min(backoff * 1.5, RETRY_BACKOFF_MAX)


def _self_check():
    assert select_download_method('http://www.aslpro.com/x.swf') == download_aslpro
    assert select_download_method('https://www.youtube.com/watch?v=x') == download_youtube
    assert select_download_method('https://media.spreadthesign.com/x.mp4') == download_others

    expanded = expand_with_synonyms({'father', 'unrelated_word'})
    assert {'father', 'dad', 'daddy', 'unrelated_word'} <= expanded
    assert 'mother' not in expanded  # only groups touching vocab get pulled in

    assert is_permanent_failure("ERROR: [youtube] xyz: Video unavailable")
    assert not is_permanent_failure("ERROR: [youtube] xyz: Sign in to confirm you're not a bot")

    print('self-check passed: download method routing, synonym expansion, and permanent-failure '
          'detection all work as expected')


if __name__ == '__main__':
    if '--self-test' in sys.argv:
        _self_check()
        sys.exit(0)

    content = load_filtered_content()
    logging.info('{} WLASL glosses match the vocab (incl. synonyms)'.format(len(content)))

    logging.info('Start downloading non-youtube videos.')
    download_nonyt_videos(content)
    logging.info('Non-youtube downloads complete.')

    logging.info('Start downloading youtube videos (retrying until all are done or confirmed unavailable).')
    download_yt_videos_until_done(content)
