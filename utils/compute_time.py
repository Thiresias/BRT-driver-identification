import os
import time
from tqdm import tqdm
import cv2

isFF = False
def get_time(filename):
    [start, end] = filename.split(' - ')
    start, end = start.split('[')[-1], end.split(']')[0]
    start, end = float(start), float(end)
    duration = end - start
    return duration

def get_time_from_vid(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    nb_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = nb_frame/fps
    return duration

if not isFF:
    path = r'ME-GraphAU/output'
    path_train = os.path.join(path, 'train')
    path_test = os.path.join(path, 'test')

    list_file_train = os.listdir(path_train)
    list_file_test = os.listdir(path_test)

    # Compute train
    duration_train = 0
    pbar = tqdm(list_file_train, leave=False)
    for file in pbar:
        if file.endswith(']_AU.json'):
            duration_train += get_time(file)
    print(f"Duration Train: {time.strftime('%H:%M:%S', time.gmtime(duration_train))}")


    # Compute test
    duration_test = 0
    pbar = tqdm(list_file_test, leave=False)
    for file in pbar:
        if file.endswith(']_AU.json'):
            duration_test += get_time(file)
    print(f"Duration Test: {time.strftime('%H:%M:%S', time.gmtime(duration_test))}")

    print(f"Total duration = {time.strftime('%H:%M:%S', time.gmtime(duration_train + duration_test))}")

else:
    path = r'ME-GraphAU/output'
    path_train = os.path.join(path, 'train')
    path_test = os.path.join(path, 'test')
    FFpath =  r'/medias/db/deepfakes/Faceforensics/original_sequences/youtube/c23/videos'
    CDFpath = r'/medias/db/deepfakes/Celeb-DF-v2/Celeb-real'

    list_file_train = os.listdir(path_train)
    list_file_test = os.listdir(path_test)

    # Compute train
    duration_train = 0
    pbar = tqdm(list_file_train, leave=False)
    for file in pbar:
        if file.endswith('_AU.json'):
            vidID = file.split('_AU.json')[0][-3:]
            FFequivalent = os.path.join(FFpath, vidID+'.mp4')
            if os.path.exists(FFequivalent):
                duration_train += get_time_from_vid(FFequivalent)
    print(f"Duration Train: {time.strftime('%H:%M:%S', time.gmtime(duration_train))}")

    # Compute test
    duration_test = 0
    pbar = tqdm(list_file_test, leave=False)
    for file in pbar:
        if file.endswith('_AU.json'):
            vidID = file.split('_AU.json')[0]
            CDFequivalent = os.path.join(CDFpath, vidID+'.mp4')
            if os.path.exists(CDFequivalent):
                duration_test += get_time_from_vid(CDFequivalent)
    print(f"Duration Test: {time.strftime('%H:%M:%S', time.gmtime(duration_test))}")

    print(f"Total duration = {time.strftime('%H:%M:%S', time.gmtime(duration_train + duration_test))}")