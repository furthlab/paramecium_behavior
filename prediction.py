import cv2
import copy
import numpy as np
import pandas as pd
import sys
from tracking_kalman.detect_stardist import Detectors
from tracking_kalman.tracking import Tracker

import haiku as hk
import jax
import mediapy as media
import tree

from tapnet import tapir_model
from tapnet.utils import transforms
from tapnet.utils import viz_utils

# @title Load Checkpoint {form-width: "25%"}

checkpoint_path = 'tapnet/checkpoints/causal_tapir_checkpoint.npy' #tapir_checkpoint.npy
ckpt_state = np.load(checkpoint_path, allow_pickle=True).item()
params, state = ckpt_state['params'], ckpt_state['state']

# @title Build Model {form-width: "25%"}

def build_model(frames, query_points):
  """Compute point tracks and occlusions given frames and query points."""
  model = tapir_model.TAPIR()
  outputs = model(
      video=frames,
      is_training=False,
      query_points=query_points,
      query_chunk_size=64,
  )
  return outputs

model = hk.transform_with_state(build_model)
model_apply = jax.jit(model.apply)

# @title Utility Functions {form-width: "25%"}

def preprocess_frames(frames):
  """Preprocess frames to model inputs.

  Args:
    frames: [num_frames, height, width, 3], [0, 255], np.uint8

  Returns:
    frames: [num_frames, height, width, 3], [-1, 1], np.float32
  """
  frames = frames.astype(np.float32)
  frames = frames / 255 * 2 - 1
  return frames


def postprocess_occlusions(occlusions, expected_dist):
  """Postprocess occlusions to boolean visible flag.

  Args:
    occlusions: [num_points, num_frames], [-inf, inf], np.float32
    expected_dist: [num_points, num_frames], [-inf, inf], np.float32

  Returns:
    visibles: [num_points, num_frames], bool
  """
  # visibles = occlusions < 0
  visibles = (1 - jax.nn.sigmoid(occlusions)) * (1 - jax.nn.sigmoid(expected_dist)) > 0.5
  return visibles

def inference(frames, query_points):
    # Preprocess video to match model inputs format
    frames = preprocess_frames(frames)
    num_frames, height, width = frames.shape[0:3]
    query_points = query_points.astype(np.float32)
    frames, query_points = frames[None], query_points[None]  # Add batch dimension
    
    # Model inference
    rng = jax.random.PRNGKey(42)
    outputs, _ = model_apply(params, state, rng, frames, query_points)
    outputs = tree.map_structure(lambda x: np.array(x[0]), outputs)
    tracks, occlusions, expected_dist = outputs['tracks'], outputs['occlusion'], outputs['expected_dist']
    
    # Binarize occlusions
    visibles = postprocess_occlusions(occlusions, expected_dist)
    return tracks, visibles



def sample_random_points(frame_max_idx, height, width, num_points):
  """Sample random points with (time, height, width) order."""
  y = np.random.randint(0, height, (num_points, 1))
  x = np.random.randint(0, width, (num_points, 1))
  t = np.random.randint(0, frame_max_idx + 1, (num_points, 1))
  points = np.concatenate((t, y, x), axis=-1).astype(np.int32)  # [num_points, 3]
  return points

def main(video_path, output_video_path, track_data_path):
    track_data = pd.DataFrame(columns=['ID', 'X', 'Y'])

    cap = cv2.VideoCapture(video_path)
    detector = Detectors()
    tracker = Tracker(10, 30, 8, 120)

    # Variables initialization
    track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                    (0, 255, 255), (255, 0, 255), (255, 127, 255),
                    (127, 0, 255), (127, 0, 127)]
    pause = False

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Variables initialization
    skip_frame_count = 0

    # Create video writer object to save the output
    output_video = cv2.VideoWriter(output_video_path,
                                   cv2.VideoWriter_fourcc(*'mp4v'),
                                   fps, (width, height))

    # Read the first frame
    ret, prev_frame = cap.read()
    centers = detector.Detect(prev_frame)

    tracking_data = []
    # Iterate through the 'centers' array
    for cell in centers:
        x, y = cell[:, 0]  # Extract the x and y coordinates from the array
        tracking_data.append([skip_frame_count, y, x])  # Append the frame number and coordinates as a list

    # Convert the 'tracking_data' list to a numpy array with dtype int32
    tracking = np.array(tracking_data, dtype=np.int32)

    # Infinite loop to process video frames
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            break

        # Make copy of original frame
        orig_frame = copy.copy(frame)

        # Detect and return centroids of the objects in the frame
        centers = detector.Detect(orig_frame)
        
        tracking_data = []
        # Iterate through the 'centers' array
        for cell in centers:
            x, y = cell[:, 0]  # Extract the x and y coordinates from the array
            tracking_data.append([skip_frame_count, y, x])  # Append the frame number and coordinates as a list


        # Convert the 'tracking_data' list to a numpy array with dtype int32
        tracking_tmp = np.array(tracking_data, dtype=np.int32)
        # Check if 'skip_frame_count' is a multiple of 15
        if skip_frame_count % 10 == 0:
            # Append 'tracking_tmp' to 'tracking' every 15th frame
            tracking = np.append(tracking, tracking_tmp, axis=0)

        skip_frame_count += 1

        # If centroids are detected then track them
        if len(centers) > 0:
            # Track object using Kalman Filter
            tracker.Update(centers)

            # For identified object tracks, draw tracking lines
            # Use various colors to indicate different track_id
            for i in range(len(tracker.tracks)):
                if len(tracker.tracks[i].trace) > 1:
                    for j in range(len(tracker.tracks[i].trace) - 1):
                        # Draw trace line
                        x1 = tracker.tracks[i].trace[j][0][0]
                        y1 = tracker.tracks[i].trace[j][1][0]
                        x2 = tracker.tracks[i].trace[j + 1][0][0]
                        y2 = tracker.tracks[i].trace[j + 1][1][0]
                        clr = tracker.tracks[i].track_id % 9
                        cv2.line(orig_frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                 track_colors[clr], 1)

                    track_id = tracker.tracks[i].track_id
                    x = tracker.tracks[i].trace[j + 1][0][0]
                    y = tracker.tracks[i].trace[j + 1][1][0]
                    track_data = pd.concat([track_data, pd.DataFrame({'Time': [skip_frame_count], 'ID': [track_id], 'X': [x], 'Y': [y]})], ignore_index=True)

        # Make copy of original frame
        prev_frame = copy.copy(orig_frame)

        # Write the processed frame to the output video
        output_video.write(orig_frame)

    # Release the video capture and writer objects
    cap.release()
    output_video.release()

    # Count the number of unique track IDs
    unique_ids = track_data['ID'].nunique()

    # Print the count
    print("Number of unique track IDs:", unique_ids)
    print("Frames Per Seconds (fps):", fps)

    # Save track_data as CSV
    track_data.to_csv(track_data_path, index=False)

    # @title Load an Exemplar Video {form-width: "25%"}

    print("Loading video:")

    video = media.read_video(video_path) #'tapnet/examplar_videos/horsejump-high.mp4'

    # @title Predict Sparse Point Tracks {form-width: "25%"}

    resize_height = 320  # @param {type: "integer"}
    resize_width = 320  # @param {type: "integer"}

    height, width = video.shape[1:3]
    frames = media.resize_video(video, (resize_height, resize_width))

    # Scale down the coordinates to a 256x256 image
    scaled_coords = tracking * (1, resize_height / 480, resize_width / 640)

    #query_points = np.insert(query_points, 0, 0, axis=1)

    print("Inference:")

    tracks, visibles = inference(frames, scaled_coords)

    # Visualize sparse point tracks
    print("transform:")

    tracks = transforms.convert_grid_coordinates(tracks, (resize_width, resize_height), (width, height))
    video_viz = viz_utils.paint_point_track(video, tracks, visibles)
    #media.show_video(video_viz, fps=10)

    # Assuming you have a NumPy array called 'frames' with shape (num_frames, height, width, 3) and dtype uint8

    # Define the output video path and filename
    output_path = 'output_video_cells.mp4'

    # Obtain the video dimensions
    num_frames, height, width, _ = video_viz.shape

    print("Saving video:")
    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec (e.g., 'mp4v', 'avc1', 'XVID')
    fps = 10  # Specify the frames per second (FPS)
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Iterate through each frame and write it to the video file
    for i in range(num_frames):
        frame = cv2.cvtColor(video_viz[i], cv2.COLOR_RGB2BGR) 
        video_writer.write(frame)

    # Release the VideoWriter
    video_writer.release()

    print(f"Video saved as {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python prediction.py './video/example.avi' 'out_example.mp4' 'example.csv'")
    else:
        video_path = sys.argv[1]
        output_video_path = sys.argv[2]
        track_data_path = sys.argv[3]
        main(video_path, output_video_path, track_data_path)
