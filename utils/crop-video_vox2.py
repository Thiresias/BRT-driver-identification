import face_alignment
import skimage.io
import numpy
from argparse import ArgumentParser
from skimage import img_as_ubyte
from skimage.transform import resize
from tqdm import tqdm
import os
import imageio
import numpy as np
import warnings
import cv2
import subprocess
warnings.filterwarnings("ignore")

# def extract_bbox(frame, fa):
#     if max(frame.shape[0], frame.shape[1]) > 640:
#         scale_factor =  max(frame.shape[0], frame.shape[1]) / 640.0
#         frame = resize(frame, (int(frame.shape[0] / scale_factor), int(frame.shape[1] / scale_factor)))
#         frame = img_as_ubyte(frame)
#     else:
#         scale_factor = 1
#     frame = frame[..., :3]
#     bboxes = fa.face_detector.detect_from_image(frame[..., ::-1])
#     if len(bboxes) == 0:
#         return []
#     return np.array(bboxes)[:, :-1] * scale_factor

# def list_of_valid_videos(args):
#     """
#     Input:
#        - args.inp: rootdir of the videos
#     Output:
#        - list_of_videos: list of video to crop
#     """
#     rootdir = args.inp
#     list_of_videos = []
#     list_of_vid = os.listdir(f"{rootdir}")
#     for vid in list_of_vid:
#         if vid.endswith('.mp4'):
#             vidID = vid.split('_')[0]
#             list_of_videos.append(os.path.join(f"{rootdir}",vid))
#     return list_of_videos


# def extract_bbox(frame, fa):
#     if max(frame.shape[0], frame.shape[1]) > 640:
#         scale_factor = max(frame.shape[0], frame.shape[1]) / 640.0
#         frame = resize(frame, (int(frame.shape[0] / scale_factor), int(frame.shape[1] / scale_factor)))
#         frame = img_as_ubyte(frame)
#     else:
#         scale_factor = 1
#     frame = frame[..., :3]
#     bboxes = fa.face_detector.detect_from_image(frame[..., ::-1])
#     if len(bboxes) == 0:
#         return []
#     return np.array(bboxes)[:, :-1] * scale_factor

def extract_bbox(frame, fa):
    # Resize frame if larger than 640 on any dimension to reduce computational load
    if max(frame.shape[0], frame.shape[1]) > 640:
        scale_factor = max(frame.shape[0], frame.shape[1]) / 640.0
        frame = resize(frame, (int(frame.shape[0] / scale_factor), int(frame.shape[1] / scale_factor)))
        frame = img_as_ubyte(frame)
    else:
        scale_factor = 1

    # Ensure the frame has only three channels (RGB)
    frame = frame[..., :3]

    # Detect faces in the frame; the face detector expects BGR images
    bboxes = fa.face_detector.detect_from_image(frame[..., ::-1])
    if len(bboxes) == 0:
        return []

    # Calculate the area of each bounding box and select the biggest one
    areas = [(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) for bbox in bboxes]
    max_area_index = np.argmax(areas)

    # Return the biggest bounding box, scaled back to original size
    biggest_bbox = bboxes[max_area_index]
    return np.array(biggest_bbox[:-1]) * scale_factor  # omitting the confidence score and scaling up



def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def join(tube_bbox, bbox):
    xA = min(tube_bbox[0], bbox[0])
    yA = min(tube_bbox[1], bbox[1])
    xB = max(tube_bbox[2], bbox[2])
    yB = max(tube_bbox[3], bbox[3])
    return (xA, yA, xB, yB)


def compute_bbox(args, start, end, fps, tube_bbox, frame_shape, inp, out, image_shape, increase_area=0.1):
    left, top, right, bot = tube_bbox
    width = right - left
    height = bot - top

    #Computing aspect preserving bbox
    width_increase = max(increase_area, ((1 + 2 * increase_area) * height - width) / (2 * width))
    height_increase = max(increase_area, ((1 + 2 * increase_area) * width - height) / (2 * height))

    left = int(left - width_increase * width)
    top = int(top - height_increase * height)
    right = int(right + width_increase * width)
    bot = int(bot + height_increase * height)

    top, bot, left, right = max(0, top), min(bot, frame_shape[0]), max(0, left), min(right, frame_shape[1])
    h, w = bot - top, right - left

    start = start / fps
    end = end / fps
    time = end - start

    scale = f'{image_shape[0]}:{image_shape[1]}'
    filename = os.path.basename(inp)

    return f'ffmpeg -i "{inp}" -ss {start} -t {time} -filter:v "crop={w}:{h}:{left}:{top}, scale={scale}" "{out}__{start:.2f}_{end:.2f}.mp4" -n'


def compute_bbox_trajectories(trajectories, fps, frame_shape, args, path_of_vid, output_path):
    commands = []
    for i, (bbox, tube_bbox, start, end) in enumerate(trajectories):
        if (end - start) > args.min_frames:
            command = compute_bbox(args, start, end, fps, tube_bbox, frame_shape, inp=path_of_vid, out=output_path, image_shape=args.image_shape, increase_area=args.increase)
            commands.append(command)
    return commands


# def process_video(args):
#     device = 'cpu' if args.cpu else 'cuda'
#     fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device=device)
#     video = imageio.get_reader(args.inp)

#     trajectories = []
#     previous_frame = None
#     fps = video.get_meta_data()['fps']
#     commands = []
#     try:
#         for i, frame in tqdm(enumerate(video)):
#             frame_shape = frame.shape
#             bboxes =  extract_bbox(frame, fa)
#             ## For each trajectory check the criterion
#             not_valid_trajectories = []
#             valid_trajectories = []

#             for trajectory in trajectories:
#                 tube_bbox = trajectory[0]
#                 intersection = 0
#                 for bbox in bboxes:
#                     intersection = max(intersection, bb_intersection_over_union(tube_bbox, bbox))
#                 if intersection > args.iou_with_initial:
#                     valid_trajectories.append(trajectory)
#                 else:
#                     not_valid_trajectories.append(trajectory)

#             commands += compute_bbox_trajectories(not_valid_trajectories, fps, frame_shape, args)
#             trajectories = valid_trajectories

#             ## Assign bbox to trajectories, create new trajectories
#             for bbox in bboxes:
#                 intersection = 0
#                 current_trajectory = None
#                 for trajectory in trajectories:
#                     tube_bbox = trajectory[0]
#                     current_intersection = bb_intersection_over_union(tube_bbox, bbox)
#                     if intersection < current_intersection and current_intersection > args.iou_with_initial:
#                         intersection = bb_intersection_over_union(tube_bbox, bbox)
#                         current_trajectory = trajectory

#                 ## Create new trajectory
#                 if current_trajectory is None:
#                     trajectories.append([bbox, bbox, i, i])
#                 else:
#                     current_trajectory[3] = i
#                     current_trajectory[1] = join(current_trajectory[1], bbox)


#     except IndexError as e:
#         raise (e)

#     commands += compute_bbox_trajectories(trajectories, fps, frame_shape, args)
#     return commands


def process_video(args, vid_path, output_path):
    device = 'cpu' if args.cpu else 'cuda'
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device=device)
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        raise ValueError("Error opening video stream or file")

    fps = cap.get(cv2.CAP_PROP_FPS)
    trajectories = []
    previous_frame = None
    commands = []
    frame_index = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_shape = frame.shape
            bbox = extract_bbox(frame, fa)
            if len(bbox) == 0:
                break
            # For each trajectory check the criterion
            not_valid_trajectories = []
            valid_trajectories = []
            for trajectory in trajectories:
                tube_bbox = trajectory[0]
                intersection = 0
                # for bbox in bboxes:
                intersection = max(intersection, bb_intersection_over_union(tube_bbox, bbox))
                if intersection > args.iou_with_initial:
                    valid_trajectories.append(trajectory)
                else:
                    not_valid_trajectories.append(trajectory)

            commands += compute_bbox_trajectories(not_valid_trajectories, fps, frame_shape, args, vid_path, output_path)
            trajectories = valid_trajectories

            ## Assign bbox to trajectories, create new trajectories
            # for bbox in bboxes:
            intersection = 0
            current_trajectory = None
            for trajectory in trajectories:
                tube_bbox = trajectory[0]
                current_intersection = bb_intersection_over_union(tube_bbox, bbox)
                if intersection < current_intersection and current_intersection > args.iou_with_initial:
                    intersection = bb_intersection_over_union(tube_bbox, bbox)
                    current_trajectory = trajectory

            ## Create new trajectory
            if current_trajectory is None:
                trajectories.append([bbox, bbox, frame_index, frame_index])
            else:
                current_trajectory[3] = frame_index
                current_trajectory[1] = join(current_trajectory[1], bbox)

            frame_index += 1

    except Exception as e:
        raise e

    finally:
        cap.release()

    commands += compute_bbox_trajectories(trajectories, fps, frame_shape, args, vid_path, output_path)
    return commands


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--image_shape", default=(256, 256), type=lambda x: tuple(map(int, x.split(','))),
                        help="Image shape")
    parser.add_argument("--increase", default=0.1, type=float, help='Increase bbox by this amount')
    parser.add_argument("--iou_with_initial", type=float, default=0.25, help="The minimal allowed iou with inital bbox")
    parser.add_argument("--inp", required=True, help='Input image or video (Contains videos)')
    parser.add_argument("--out", required=True, help="Output folder (Will contain videos)")
    parser.add_argument("--min_frames", type=int, default=150,  help='Minimum number of frames')
    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")
    parser.add_argument("--train", action="store_true")


    args = parser.parse_args()
    # subset = 'train' if args.train else 'test'
    list_of_IDs = os.listdir(args.inp)
    print(list_of_IDs)
    
    pbar = tqdm(list_of_IDs, leave=True)

    for ID in pbar:
        pbar.set_description(f'Processing {os.path.basename(ID)}')
        id_path = os.path.join(args.inp, ID)
        list_of_vid = os.listdir(id_path)
        for idx, vid_id in enumerate(list_of_vid):
            pbar.set_description(f'Processing {os.path.basename(ID)}, video {idx}/{len(list_of_vid)}.')
            for clipname in os.listdir(os.path.join(id_path, vid_id)):
                video_path = os.path.join(id_path, vid_id, clipname)
                output_path = os.path.join(args.out, f'{ID}__{vid_id}__{clipname[:-4]}')

                if os.path.exists(output_path):
                    continue
                try:
                    commands = process_video(args, video_path, output_path)
                except:
                    print(f"Error encountered for video {output_path}. Skipping...")
                for command in commands:
                        subprocess.call(command, shell=True)
    # for vid_path in pbar:
    #     pbar.set_description(f'Processing {os.path.basename(vid_path)}')
    #     new_path = f'cropped/{subset}/{os.path.basename(vid_path)}'
        
    #     if os.path.exists(f'cropped/{os.path.basename(vid_path)}'):
    #         continue
    #     commands = process_video(args, f"{vid_path}")
    #     for command in commands:
    #         subprocess.call(command,shell=True)

        