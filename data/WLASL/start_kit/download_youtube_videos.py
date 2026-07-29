"""Run just the YouTube half of video_downloader.py as its own process --
useful for resuming the YouTube retry loop later without re-running the
(already complete) non-YouTube download first.
"""
import logging

from video_downloader import download_yt_videos_until_done, load_filtered_content

if __name__ == '__main__':
    logging.info('Start downloading youtube videos.')
    download_yt_videos_until_done(load_filtered_content())
