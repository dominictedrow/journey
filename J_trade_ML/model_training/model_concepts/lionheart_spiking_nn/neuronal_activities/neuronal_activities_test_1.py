import os
import sys
import pickle
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from moviepy.editor import VideoFileClip, concatenate_videoclips, clips_array
import imageio
import subprocess
import gc
import math
from PIL import Image
from math import sqrt, ceil

# Absolute path of the current script
current_script_path = os.path.abspath(__file__)

# Directory of the current script
current_directory = os.path.dirname(current_script_path)

# Parent directory (where new_nn_main_test_25_4_1.py is located)
parent_directory = os.path.dirname(current_directory)

# Add the parent directory to sys.path if not already included
if parent_directory not in sys.path:
    sys.path.append(parent_directory)

print(parent_directory)

# Now attempt to import from lion_heart_hybrid_regression_1.py
# from lion_heart_hybrid_test_1 import SequentialSpikeTrainModel, SpikingNeuronLayer

num_neurons = 1024

def create_3d_visualization_for_layer(spikes, output_dir, layer_name, num_neurons, fps=60, duration_in_seconds=30):
    num_frames = fps * duration_in_seconds
    if len(spikes.shape) == 3:
        spikes_combined = np.mean(spikes, axis=0)
    else:
        spikes_combined = spikes
    
    repeats = np.ceil(num_frames / spikes_combined.shape[0]).astype(int)
    spikes_combined = np.tile(spikes_combined, (repeats, 1))

    fig_width, fig_height = 16, 8
    dpi = 150
    output_file_path = os.path.join(output_dir, f"neuronal_activity_{layer_name}.mp4")
    writer = imageio.get_writer(output_file_path, fps=fps, codec='libx264', quality=7)

    # Define unit locations within boundaries
    max_bound = 10  # Define the maximum boundary for unit locations
    min_bound = -10  # Define the minimum boundary for unit locations
    unit_locations = np.random.uniform(low=min_bound, high=max_bound, size=(num_neurons, 3))

    # Create a single figure outside the loop
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    for frame_number in range(num_frames):
        ax = fig.add_subplot(111, projection='3d')
        frame_idx = frame_number % spikes_combined.shape[0]
        colors = ['red' if spikes_combined[frame_idx, neuron] > 0 else 'black' for neuron in range(spikes_combined.shape[1])]
        
        # Plot units at predefined locations
        for neuron in range(num_neurons):
            ax.scatter(unit_locations[neuron][0], unit_locations[neuron][1], unit_locations[neuron][2], c=colors[neuron], s=65)
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        plt.title(f"Neuronal Activity: {layer_name} (Frame {frame_number+1})")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        writer.append_data(image)

        # Clear the figure instead of closing it
        plt.clf()

    plt.close(fig)
    writer.close()
    gc.collect()

# Load the aggregated spike records
neuronal_activities_dir = "C:/Users/crgon/OneDrive/Desktop/trading/model_testing_text/training_models/neuronal_activities"
epoch_file_name = "spike_records_epoch_1.pkl"
epoch_file_path = os.path.join(neuronal_activities_dir, epoch_file_name)

with open(epoch_file_path, "rb") as f:
    epoch_spike_records = pickle.load(f)

# Generate a visualization for each layer
for layer_name, spikes in epoch_spike_records.items():
    create_3d_visualization_for_layer(spikes, neuronal_activities_dir, layer_name, num_neurons)

def combine_videos(video_files, output_file):
    """
    Combine multiple video files into a single video file using moviepy.
    
    Args:
    video_files (list): A list of paths to the video files to combine.
    output_file (str): The path to the output combined video file.
    """
    clips = [VideoFileClip(video_file) for video_file in video_files]
    final_clip = concatenate_videoclips(clips, method="compose")
    final_clip.write_videofile(output_file, codec="libx264", audio_codec="aac")

# Specify the directory where your videos are located
video_directory = "C:/Users/crgon/OneDrive/Desktop/trading/model_testing_text/training_models/neuronal_activities"

# List all MP4 files in the specified directory
video_files = [os.path.join(video_directory, f) for f in os.listdir(video_directory) if f.endswith('.mp4')]

# Define the path for the output combined video file
output_file = os.path.join(video_directory, "combined_video.mp4")

# Combine the videos
combine_videos(video_files, output_file)

def combine_videos_grid(video_files, output_file):
    """
    Combine multiple video files into a single video frame, with all videos displayed simultaneously.
    The duration is automatically determined based on the first video file.

    Args:
        video_files (list): A list of paths to the video files to combine.
        output_file (str): The path to the output combined video file.
    """
    # Load the first video to determine the common duration
    first_clip = VideoFileClip(video_files[0])
    duration_in_seconds = first_clip.duration

    # Load all video clips and ensure they're limited to the first video's duration
    clips = [VideoFileClip(video_file).subclip(0, duration_in_seconds) for video_file in video_files]

    # Determine grid size based on the number of videos
    num_videos = len(clips)
    grid_size = math.ceil(math.sqrt(num_videos))  # Square grid to hold all videos
    cols = grid_size
    rows = math.ceil(num_videos / grid_size)

    # Resize clips to fit the grid while maintaining aspect ratio
    max_width = min(clip.size[0] for clip in clips) // cols
    max_height = min(clip.size[1] for clip in clips) // rows
    resized_clips = [clip.resize(width=max_width, height=max_height).set_duration(duration_in_seconds) for clip in clips]

    # Arrange clips in a grid
    grid = []
    for r in range(rows):
        row_clips = resized_clips[r*cols:(r+1)*cols]
        while len(row_clips) < cols:
            # Use a blank clip to fill in any missing slots
            blank_clip = VideoFileClip(video_files[0]).subclip(0, 0.01).resize(width=max_width, height=max_height).set_duration(duration_in_seconds).set_opacity(0)
            row_clips.append(blank_clip)
        grid.append(row_clips)

    # Create the final composite video
    final_clip = clips_array(grid)
    final_clip.write_videofile(output_file, codec="libx264", fps=24, audio_codec="aac")

# Specify the directory where your videos are located
video_directory = "C:/Users/crgon/OneDrive/Desktop/trading/model_testing_text/training_models/neuronal_activities"

combined_video_name = "combined_video.mp4"  # Name of the combined video to exclude

# List video files, excluding the combined_video.mp4
video_files = [
    os.path.join(video_directory, f) 
    for f in os.listdir(video_directory) 
    if f.endswith('.mp4') and f != combined_video_name
]

# Define the path for the output combined video file
output_file_all_in_one = os.path.join(video_directory, "all_in_one_combined_video.mp4")

# Combine the videos into a single frame
combine_videos_grid(video_files, output_file_all_in_one)

