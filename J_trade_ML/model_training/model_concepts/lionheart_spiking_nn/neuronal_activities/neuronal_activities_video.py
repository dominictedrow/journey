import os
import sys
import pickle
import numpy as np
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
fps=100 
duration_in_seconds=60

def create_3d_visualization_for_layer(spikes, output_dir, layer_name, fps, duration_in_seconds):
    # Determine total frames for the given duration and fps
    num_frames = fps * duration_in_seconds
    
    if len(spikes.shape) == 3:
        spikes_combined = np.mean(spikes, axis=0)
    else:
        spikes_combined = spikes
    
    # Filter out all-black frames (where all units are inactive)
    active_frame_indices = np.where(np.sum(spikes_combined, axis=1) > 0)[0]
    spikes_combined = spikes_combined[active_frame_indices]
    
    # Calculate the number of repeats needed to fill the duration, prioritizing spike diversity
    repeats = np.ceil(num_frames / len(active_frame_indices)).astype(int)
    spikes_combined = np.tile(spikes_combined, (repeats, 1))[:num_frames]

    fig_width, fig_height = 8, 8
    dpi = 300
    enhanced_font_size = 14
    output_file_path = os.path.join(output_dir, f"neuronal_activity_{layer_name}.mp4")
    writer = imageio.get_writer(output_file_path, fps=fps, codec='libx264', quality=9)

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    for frame_number in range(num_frames if num_frames < spikes_combined.shape[0] else spikes_combined.shape[0]):
        ax = fig.add_subplot(111, projection='3d')
        coords = np.random.rand(3, spikes_combined.shape[1]) * 20 - 10
        colors = ['red' if spikes_combined[frame_number, neuron] > 0 else 'black' for neuron in range(spikes_combined.shape[1])]
        ax.scatter(coords[0], coords[1], coords[2], c=colors, s=75)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        
        modified_layer_name = "Lion_Heart_Selective_Firing_Layer" + layer_name.split("spiking_neuron_layer")[-1]
        plt.title(f"Neuronal Activity: {modified_layer_name} (Frame {frame_number+1})", fontsize=enhanced_font_size, fontweight='bold')

        active_units = np.sum(spikes_combined[frame_number] > 0)
        non_active_units = np.sum(spikes_combined[frame_number] == 0)
        plt.figtext(0.01, 0.01, f"Active Units = {active_units}", color="black", fontsize=enhanced_font_size, fontweight='bold')
        plt.figtext(0.01, 0.05, f"Non-Active Units = {non_active_units}", color="black", fontsize=enhanced_font_size, fontweight='bold')

        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        writer.append_data(image)
        
        plt.clf()

    plt.close(fig)
    writer.close()
    gc.collect()

# Example usage
neuronal_activities_dir = "C:/Users/crgon/OneDrive/Desktop/trading/model_testing_text/training_models/neuronal_activities"
epoch_file_name = "spike_records_epoch_1.pkl"
epoch_file_path = os.path.join(neuronal_activities_dir, epoch_file_name)

with open(epoch_file_path, "rb") as f:
    epoch_spike_records = pickle.load(f)

for layer_name, spikes in epoch_spike_records.items():
    create_3d_visualization_for_layer(spikes, neuronal_activities_dir, layer_name)

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
    The duration and frame rate are automatically determined based on the first video file.

    Args:
        video_files (list): A list of paths to the video files to combine.
        output_file (str): The path to the output combined video file.
    """
    # Load the first video to determine the common duration and frame rate
    first_clip = VideoFileClip(video_files[0])
    duration_in_seconds = first_clip.duration
    fps = first_clip.fps  # Fetch the frame rate of the first video

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
    final_clip.write_videofile(output_file, codec="libx264", fps=fps, audio_codec="aac")

    # Specify the directory where your videos are located
video_directory = "C:/Users/crgon/OneDrive/Desktop/trading/model_testing_text/training_models/neuronal_activities"

# Example usage (assuming you've already defined `video_directory` and other variables as in your original code)
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

