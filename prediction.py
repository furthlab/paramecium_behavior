import cv2
import copy
import numpy as np
import pandas as pd
import sys
from tracking_kalman.detect_stardist import Detectors
from tracking_kalman.tracking import Tracker

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

    # Infinite loop to process video frames
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            break

        # Make copy of original frame
        orig_frame = copy.copy(frame)

        frame = cv2.absdiff(frame, prev_frame)
        # Define the contrast and brightness adjustment parameters
        alpha = 1.8  # Contrast control (1.0 means no change)
        beta = 60   # Brightness control (0-100, with 0 being black)

        # Apply the contrast and brightness adjustment
        adjusted_image = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        skip_frame_count += 1

        # Detect and return centroids of the objects in the frame
        centers = detector.Detect(orig_frame)

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


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python prediction.py './video/example.avi' 'out_example.mp4' 'example.csv'")
    else:
        video_path = sys.argv[1]
        output_video_path = sys.argv[2]
        track_data_path = sys.argv[3]
        main(video_path, output_video_path, track_data_path)
