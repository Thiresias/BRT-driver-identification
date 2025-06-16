import os, argparse, subprocess, re
import numpy as np
import cv2


def parse():
    """
    Load arguments from parser.
    Output:
     - args: parser object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_list', type=str, help='Path of the .txt containing the list of the sequences')
    parser.add_argument('--data_path', type=str, help='Path to the video dataset. (default: data/belkacem)', default=os.path.join('dataset', 'belkacem'))
    parser.add_argument('--save_clips', action='store_true', help="Specify if the clips must be stored")
    parser.add_argument('--output_folder', type=str, default='clips', help="Specify the folder where you want to save the clips. This argument will be ignored if you don't use save_clip.")
    args = parser.parse_args()
    return args


def convert_time_stamp(time):
    """
    Convert time in float format to hh:mm:sec string format.
    Input:
     - time <float>: in seconds.
    Ouput:
     - time <str>: in hh:mm:sec format
    """
    sec = time%60
    min = (time//60)%60
    hour = (time//60)//60
    return f"{hour:02.0f}:{min:02.0f}:{sec:05.2f}"



def get_clip_info(args, clip_name):
    """
    Get timestamps for a single video.
    Input:
     - clip_name: Clip name of the following form: <clip_fullID>_[<time_start> - <time_end>].mp4 (ex: '042_part_10 [447.88 - 467.92].mp4' or '000_part_1 [59.32 - 93.96].mp4')
    Ouput:
     - original_video_name: Name of the original video without the extension (ex: '042' or '000')
     - clip_ID
     - frame_start: Frame index of the beginning of the sequence.
     - frame_end: Frame index of the end of the sequence.
    """
    # From <clip_full_name>_[<time_start> - <time_end>].mp4 to <clip_full_name>
    clip_full_name = clip_name.split(' ')[0]

    # From <clip_ID>_[<time_start> - <time_end>].mp4 to ["[<time_start>"", "<time_end>]"]
    timestamp = re.findall(r'\[.*?\]', clip_name)[0].split(' - ')

    # From ["[<time_start>"", "<time_end>]"] to [<time_start>, <time_end>]
    time_start, time_end = float(timestamp[0][1:]), float(timestamp[1][:-1])

    # From <clip_full_name> to <original_video_name> || From <clip_full_name> to <clip_ID>
    original_video_name, clip_ID = os.path.join(args.data_path, clip_full_name.split('_')[0]+'.mp4'), int(clip_full_name.split('_')[-1])

    return original_video_name, clip_ID, time_start, time_end


def get_frame_idx(vid_path, time_start, time_end):
    """
    Get frames index for a single video.
    Input:
     - vid_path: Path to the video.
     - time_start: Timestamp in second of the beginning of the clip.
     - time_end: Timestamp in second of the end of the clip.
    Ouput:
     - frame_start: Frame index of the beginning of the sequence.
     - frame_end: Frame index of the end of the sequence.
    """
    print(f"Editing video {vid_path}")

    # Read the video and get the fps
    cap = cv2.VideoCapture(vid_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Get the exact starting and ending frame based on the frame rate (fps)
    dt = 1/fps
    frame_start = time_start//dt + 1
    frame_end = time_end//dt + 1

    return frame_start, frame_end


def extract_full_clip(vid_path, frame_start, frame_end):
    """
    Extract clip from original video with exact starting and ending frame. Usefull for generation purposes.
    Input:
     - vid_path: Path to the video.
     - frame_start: Frame index of the beginning of the sequence.
     - frame_end: Frame index of the end of the sequence.
    Output:
     - seq: List of the frames from the clip. (Usefull for dataloader in deeplearning training.)
    """
    # Read the video and get usefull information
    cap = cv2.VideoCapture(vid_path)

    # Go to the beginning of the sequence
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start-1)

    # Run through the frames of the sequence
    current_frame = frame_start
    seq = list()
    while True:
        ret, frame = cap.read()
        seq.append(frame)
        current_frame += 1
        if current_frame == frame_end: # End of sequence
            break
        elif not ret: # Not enough frames to reach the end of the sequence. Should not happen normally.
            print(f"Can't receive frame from video {vid_path} (stream end?). Exiting ...")
            break
    return seq


def extract_fair_sample_of_clip(vid_path, frame_start, frame_end):
    """
    Extract a fair sample of the clip depending of the frame rate of the video. Usefull for detection purposes.
    Input:
     - vid_path: Path to the video.
     - frame_start: Frame index of the beginning of the sequence.
     - frame_end: Frame index of the end of the sequence.
    Output:
     - seq: List of the frames sampled from the clip. (Usefull for dataloader in deeplearning training.)
    """
    # Read the video and get usefull information
    cap = cv2.VideoCapture(vid_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    sample_idx = 5 if np.abs(25-fps)<np.abs(30-fps) else 6 # Get the number of frame until you reach 0.2s of video between 25 and 30 fps videos

    # Go to the beginning of the sequence
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start-1)

    # Run through the frames of the sequence
    current_frame = frame_start
    seq = list()
    count = 0
    while True:
        ret, frame = cap.read()
        if count % sample_idx == 0: # Extract one frame every 0.2s
            seq.append(frame)
        current_frame += 1
        if current_frame == frame_end: # End of sequence
            break
        elif not ret: # Not enough frames to reach the end of the sequence. Should not happen normally.
            print(f"Can't receive frame from video {vid_path} (stream end?). Exiting ...")
            break
    return seq


def save_clip(args, vid_path, time_start, time_end, clip_ID, SKIP_DIALOGUE, SAVE_ALL):
    """
    Save or not the extracted clip.
    INPUT:
     - args
     - vid_path: full path of the original video.
     - time_start: Timestamp in second of the beginning of the clip.
     - time_end: Timestamp in second of the end of the clip.
     - clip_ID: ID of the clip to extract for video from vid_path
     - SKIP_DIALOGUE: Skip question about overwriting video with ffmpeg. (default=False)
     - SAVE_ALL: If SKIP_DIALOGUE is set to True decide if the video must be overwritten or not. (default=False)
    OUTPUT:
     - SKIP_DIALOGUE: Skip question about overwriting video with ffmpeg starting from the next iteration. (default=False)
     - SAVE_ALL: If SKIP_DIALOGUE is set to True decide if the video must be overwritten or not for the following iteration. (default=False)
    """
    # Get the name of the video to save and check if it exists
    vid_file = os.path.basename(vid_path)[:-4]
    [vid_name, _] = os.path.splitext(vid_file)
    if os.path.exists(f"{args.output_folder}/{vid_name}_part_{clip_ID}_[{time_start}_-_{time_end}].mp4"):
        # If the clip already exists, ask if the user wants to overwrite for one/all video/videos.
        ## Ask question
        if not SKIP_DIALOGUE:
            choice = input("Clip already created. Do you wish to overwrite? [y/n/yA/nA] (yes/no/yes all/no all): ")
        ## If you want to overwrite all clips
        if SKIP_DIALOGUE and SAVE_ALL:
            command = f"ffmpeg -y -i {vid_path} -ss {convert_time_stamp(time_start)} -t {convert_time_stamp(time_end-time_start)} {args.output_folder}/{vid_name}_part_{clip_ID}_[{time_start}_-_{time_end}].mp4"
            subprocess.call(command, shell=True)
        ## If you don't want to overwrite all clips
        elif (SKIP_DIALOGUE and (not SAVE_ALL)):
            return SKIP_DIALOGUE, SAVE_ALL
        ## If you said yes, but only for one video
        elif choice in ['y', 'yes', 'yA', 'yes all']:
            if choice in ['yA', 'yes all']:
                SKIP_DIALOGUE = True
                SAVE_ALL = True
            command = f"ffmpeg -y -i {vid_path} -ss {convert_time_stamp(time_start)} -t {convert_time_stamp(time_end-time_start)} {args.output_folder}/{vid_name}_part_{clip_ID}_[{time_start}_-_{time_end}].mp4"
            subprocess.call(command,shell=True)
        ## If you don't want to overwrite all clips, make the change of the variable
        elif choice in ['nA', 'no all']:
            SKIP_DIALOGUE = True
    else:
        # The clip doesn't exist, the clip will be written normally
        command = f"ffmpeg -i {vid_path} -ss {convert_time_stamp(time_start)} -t {convert_time_stamp(time_end-time_start)} {args.output_folder}/{vid_name}_part_{clip_ID}_[{time_start}_-_{time_end}].mp4"
        subprocess.call(command, shell=True)
    return SKIP_DIALOGUE, SAVE_ALL

if __name__ == '__main__':
    SKIP_DIALOGUE = False
    SAVE_ALL = False
    args = parse()
    f = open(args.clip_list)

    for i, clip_name in enumerate(f.readlines()):
        # Get info from the clip to process
        vid_path, clip_ID, time_start, time_end = get_clip_info(args, clip_name)
        # Get the exact starting and the ending frame
        frame_start, frame_end = get_frame_idx(vid_path, time_start, time_end)

        # Video extraction
        sequence = extract_full_clip(vid_path, frame_start, frame_end)

        # Audio extraction
        ## [TO DO]

        # If you wish to save all the clips
        if args.save_clips:
            if not os.path.exists(args.output_folder):
                os.makedirs(args.output_folder)
            SKIP_DIALOGUE, SAVE_ALL = save_clip(args, vid_path, time_start, time_end, clip_ID, SKIP_DIALOGUE, SAVE_ALL)
        