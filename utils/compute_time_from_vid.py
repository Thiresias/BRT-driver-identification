import os
import cv2
from argparse import ArgumentParser
import time
from tqdm import tqdm

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--min_duration', type=float, default=8.0)
    args = parser.parse_args()

    rootdir = args.path
    min_duration = args.min_duration
    filenames = os.listdir(rootdir)

    valid_list, valid_duration, nb_valid = [], 0, 0
    non_valid_list, non_valid_duration, nb_non_valid = [], 0, 0
    for filename in tqdm(filenames):
        if filename.endswith('.mp4'):
            filepath = os.path.join(rootdir, filename)

            cap = cv2.VideoCapture(filepath)
            fps = cap.get(cv2.CAP_PROP_FPS)
            nb_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)

            duration = nb_frame/fps

            if duration >= min_duration:
                valid_list.append(filename)
                valid_duration += duration
                nb_valid += 1
            else:
                non_valid_list.append(filename)
                non_valid_duration += duration
                nb_non_valid += 1
    
    print(f"Total of valid duration: {time.strftime('%H:%M:%S', time.gmtime(valid_duration))} over {time.strftime('%H:%M:%S', time.gmtime(valid_duration+non_valid_duration))} ({100*valid_duration/(valid_duration+non_valid_duration):.2f}%).")
    print(f"Total number of valid clips: {nb_valid} over {nb_valid+nb_non_valid} ({100*nb_valid/(nb_valid+nb_non_valid):.2f}%).")

        
