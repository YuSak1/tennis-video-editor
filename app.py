import gradio as gr
import librosa
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip, concatenate_videoclips
from PIL import Image
import os
import io
# import tempfile
# import sys
# import platform
# import pkg_resources

# print("Python version:", sys.version)
# print("Platform:", platform.platform())
# print("Installed packages:")

# for pkg in sorted(pkg_resources.working_set, key=lambda x: x.project_name.lower()):
#     print(f"{pkg.project_name}=={pkg.version}")


# === Helper function: Detect rally segments from audio ===
def detect_rallies_from_audio(video_path, onset_delta=2.0, min_hits=3, padding=0.5):
    """
    onset_delta: max gap between hits to stay in same rally
    min_hits: filter out short noise
    padding: extra seconds before/after rally
    """

    audio_path = video_path.replace('.mp4', '_temp.wav')
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path, verbose=False, logger=None)
    
    y, sr = librosa.load(audio_path, sr=None)
    onset_times = librosa.onset.onset_detect(y=y, sr=sr, units='time', backtrack=True)

    os.remove(audio_path)  # Clean up temp audio

    rallies = []
    if len(onset_times) > 0:
        start_time = onset_times[0]
        hits = [start_time]
        for i in range(1, len(onset_times)):
            if onset_times[i] - onset_times[i-1] <= onset_delta:
                hits.append(onset_times[i])
            else:
                if len(hits) >= min_hits:
                    rallies.append((max(0, hits[0] - padding), hits[-1] + padding))
                hits = [onset_times[i]]
        if len(hits) >= min_hits:
            rallies.append((max(0, hits[0] - padding), hits[-1] + padding))
    
    return rallies, onset_times


# === Plotting audio waveform with hits ===
def generate_audio_plot(rallies, onset_times):
    plt.figure(figsize=(12, 3))
    plt.vlines(onset_times, ymin=0, ymax=1, color='r', alpha=0.6, label='Detected Hits')
    for start, end in rallies:
        plt.axvspan(start, end, color='green', alpha=0.2)
    plt.title("Detected Rally Segments from Audio")
    plt.xlabel("Time (s)")

    plt.gca().axes.get_yaxis().set_visible(False)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    img = Image.open(buf)

    return img


# === Core logic: Process video ===
def process_video(video_file, progress=gr.Progress(track_tqdm=True)):
    # Gradio gives you the path directly
    video_path = video_file

    # Step 1: Detect rally segments
    progress(0.15, desc="Analyzing video...")
    rally_segments, onset_times = detect_rallies_from_audio(video_path)

    if not rally_segments:
        return "No rally segments detected.", None

    # Step 2: Extract and merge clips
    progress(0.4, desc="Extracting rally segments...")
    video = VideoFileClip(video_path)
    clips = []
    for start, end in rally_segments:
        if end <= video.duration:
            clips.append(video.subclip(start, end))

    if not clips:
        return "No valid video segments found.", None

    progress(0.80, desc="Rendering final video...")
    output_path = video_path.replace(".mp4", "_rallies_only.mp4")
    final_clip = concatenate_videoclips(clips)
    final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", verbose=False, logger=None)

    # Step 3: Generate plot
    progress(0.95, desc="Creating waveform plot...")
    audio_plot = generate_audio_plot(rally_segments, onset_times)

    progress(1.0, desc="Done!")
    return "Done! Here's your edited rally-only video:", output_path, audio_plot


# === Gradio Interface ===
iface = gr.Interface(
    fn=process_video,
    inputs=gr.Video(label="Upload Your Tennis Match Video"),
    outputs=[
        gr.Text(), 
        gr.Video(label="Rally-Only Video"),
        gr.Image(label="Audio Waveform with Detected Hits")
    ],
    title="🎾 Tennis Video Editor",
    description="Upload a tennis match video. This app detects rallies and edits the video to show only those moments. <br>It works better with short videos like 10 minutes."

)

iface.launch()
