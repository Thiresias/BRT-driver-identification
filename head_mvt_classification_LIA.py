import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from PIL import Image
from torchvision import transforms as trans
import glob
import json
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import cv2
from LIA_encoder.networks.classifier import LSTMClassifier, TransformerClassifier # , BiLSTMClassifier, BiLSTMRecon, LSTMClassifier_wAttention
from sklearn import metrics
from utils.display import display_batch_lia


# from dataloader.headpose import VideoFramesDataset, MvtAnalysis
from LIA_encoder.dataset import Belkacem_CLS_lia_balanced, Belkacem_CLS_vid
from utils.metrics import compute_video_level_AUC, calculate_fnr_fpr

from torch.utils.tensorboard import SummaryWriter
        

def compute_accuracy(label_list, pred_list):
    acc = 0
    for label, score in zip(label_list, pred_list):
        pred = 1 if score>0.5 else 0
        acc += 1 if (label == pred) else 0
    return acc/len(label_list)


def num2ID(num):
    if num < 10:
        return '00'+ str(num)
    if (num >= 10) and (num < 100):
        return '0' + str(num)
    else:
        return str(num)


def drop_if_frames_missing(videos):
    for video in list(videos):
        if len(videos[video]) != 50:
            print(f"Sequence {video} has only {len(videos[video])} frames. Need 50.")
            del videos[video]
    return videos


def drop_other_videos(videos):
    for video in list(videos):
        if 'FF' in videos:
            del videos[video]
    return videos


def group_videos(videos, grp):
    new_videos = defaultdict(list)
    for video in list(videos):
        if 'FF' not in video:
            vid_name, seq_idx = video.split('_')[0], int(video.split('_')[1].split('seq')[-1])
        else:
            vid_name, seq_idx = "_".join(video.split('_')[:2]), int(video.split('_')[2].split('seq')[-1])
        # Compute new seq and frame
        new_seq_idx = seq_idx // grp
        new_seq_ID = num2ID(new_seq_idx)
        new_vid_seq = vid_name + '_seq' + new_seq_ID
        new_videos[new_vid_seq] = new_videos[new_vid_seq] + videos[video]
    return new_videos
    

def plot_time_serie(embeddings):
    time = np.linspace(0, 49, 50)
    plt.figure(figsize=(12, 8))
    # Use a loop to create each subplot if you have multiple plots
    subplots_name = ['Translation 1', 'Translation 2', 'Translation 3', 'Rotation 1', 'Rotation 2', 'Rotation 3']
    for i in range(6):
        ax = plt.subplot(2, 3, i+1)
        ax.plot(time, embeddings[0, :, i])

        # Set titles for each subplot
        ax.set_title(f'{subplots_name[i]}', fontsize=14)

        # Set larger labels if necessary
        ax.set_xlabel('frame', fontsize=12)
        ax.set_ylabel('amplitude', fontsize=12)

        # Set common amplitude scale
        ax.set_ylim(-1, 1)


    # Adjust the layout so the titles and labels do not overlap
    plt.tight_layout()
    plt.savefig(f"output/{name}/mvts_{videoID[0]}.png")

def create_experiment_folder(path):
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        os.makedirs(os.path.join(path, 'res'))
        os.makedirs(os.path.join(path, 'ckpt'))
        os.makedirs(os.path.join(path, 'tensorboard_output'))
        print(f"Experiment folder {path} has been created successfully")
    else:
        print(f"Experiment folder {path} already exists. Will overwrite some files")

def write_spec(name, classifier, lr, optimizer, data, batch_size):
    with open(os.path.join('output/', name,'summary.txt'), 'a') as f1:
        print(f"Training: {name}", file=f1)
        print(f"Classifier architecture: {classifier}", file=f1)
        print(f"lr = {lr}", file=f1)
        print(f"optimizer = {optimizer}", file=f1)
        print(f"data = {data}", file=f1)
        print(f"batch_size = {batch_size}", file=f1)  


def filter_train_test(videos, test_file, mode):
    f_test = open(test_file, 'r')
    list_test_other = list()
    list_test_POI = list()
    for line in f_test.readlines():
        if 'FFreal' in line:
            list_test_other.append(line[:-1])
        else:
            list_test_POI.append(line[:-1])

    if mode == 'train':
        for video in list(videos):
            # Because of not so brilliant implementation idea, POI video ID is like '000', '001' and other are like 'FFreal_000', 'FFreal_001'
            # The condition below is first looking if for a real video we have a pattern of a real video and then for a POI video, do we have a pattern of a POI video.
            if (('FFreal' in video) and (any(pat in video for pat in list_test_other))) or (('FFreal' not in video) and (any(pat in video for pat in list_test_POI))):
                del videos[video]
    if mode == 'test':
        for video in list(videos):
            # Because of not so brilliant implementation idea, POI video ID is like '000', '001' and other are like 'FFreal_000', 'FFreal_001'
            # The condition below is first looking if for a real video we have a pattern of a real video and then for a POI video, do we have a pattern of a POI video.
            if (('FFreal' in video) and (any(pat in video for pat in list_test_other))) or (('FFreal' not in video) and (any(pat in video for pat in list_test_POI))):
                continue
            else:
                del videos[video]
    return videos


def lr_lambda(epoch):
    if epoch < 10:
        return float(epoch) / 10
    else:
        return 1.0
    

if __name__=="__main__":
    # Set seed for reproducibility
    # torch.manual_seed(0)
    # np.random.seed(0)
    device = 'cuda'
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", default=False, help="Set training mode", action="store_true")
    parser.add_argument("--ckpt", default="", help="add ckpt path (optional)")
    parser.add_argument("--data_pos", default="../Deep3DFaceRecon_pytorch/checkpoints/face_model/results/epoch_20_000000/", help="Root path of extracted face frames")
    parser.add_argument("--data_neg", help="Root path of extracted face frames")
    parser.add_argument("--epochs", default=101, type=int, help="Number of epochs for training")
    parser.add_argument("--name", help="Name of the training")
    parser.add_argument("--train_mode", default='supervised', help="supervised or self-supervised")
    args = parser.parse_args()
    training = args.training
    ckpt = args.ckpt
    data_pos = args.data_pos
    data_neg = args.data_neg

    ds = 'cls_LIA'
    name = args.name
    epochs = args.epochs
    train_mode = args.train_mode
    if (name is None):
        raise NameError("Add value for --name. For example:         python pattern_analysis.py --name analysis1")
    if (train_mode != 'supervised') and (train_mode != 'self-supervised'):
        raise NameError("Wrong input for for train_mode. Should be 'supervised' or 'self-supervised'.")
    
    # Load data
    print("Load data")
    # data_root_HP = os.path.join(data_root_AU, '../../deep-head-pose/output')
    video_Dataset = Belkacem_CLS_vid(data_pos, data_neg, seq_length=8, train=True, obama=True)
    video_Dataset_test = Belkacem_CLS_vid(data_pos, data_neg, seq_length=8, train=False, obama=True)
    batch_size = 4
    loader = DataLoader(video_Dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    loader_test = DataLoader(video_Dataset_test, batch_size=batch_size, shuffle=True, num_workers=16)


    # Load model
    print("Load model")
    # model = TransformerClassifier(embedding_size=20, output_size=2, hidden_size=1024, batch_size=batch_size, seq_length=40).to(device) # ID leakage
    model = TransformerClassifier(embedding_size=512, output_size=2, hidden_size=1024, batch_size=batch_size, seq_length=40, nhead=8).to(device) # no ID leakage
    # model = LSTMClassifier(embedding_size=512, output_size=2, hidden_size=1024, batch_size=batch_size, seq_length=40).to(device)
    for param in model.parameters():
        if param.dim() > 1 and param.requires_grad: # Apply Xavier initialization only to parameters that are weights
            torch.nn.init.xavier_uniform_(param)
    loss_func = list()
    if ckpt:
        checkpoint = torch.load(ckpt)
        model.load_state_dict(checkpoint['model_state_dict'])
        loss_func = checkpoint['loss']

    # Training hyperparameters
    lr = 1e-4
    if training:
        output_folder = os.path.join('output/', name)
        create_experiment_folder(output_folder)
        # Live Tensorboard summary
        writer = SummaryWriter(os.path.join('output/', name, 'tensorboard_output'))
        # Progess bar
        pbar1 = tqdm(range(epochs), unit='epoch', desc='LSTM classifier training', leave=True)

        # Loss
        criterion = torch.nn.CrossEntropyLoss()
        # Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        if ckpt:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Scheduler
        warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs-10)
        scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[10])

        # Write spec
        write_spec(name, model, lr, optimizer, ds, batch_size)

        for epoch in pbar1:
            if ckpt:
                if checkpoint['epoch'] >= epoch:
                    scheduler.step()
                    continue
            pbar2 = tqdm(iter(loader), desc='Face mvt analysis/ Train', unit='batch', leave=False)
            # Score and Label list init
            score_list, label_list, vid_name = np.array([]), np.array([]), np.array([])
            avg_loss = 0
            cnt = 0
            # print("Dataset size: ", video_Dataset.__len__())
            for frames, label, videoID, _ in pbar2:
                optimizer.zero_grad()
                input_dim, label_dim = frames.shape, label.shape
                # (b_s, K, time, height, width, channel) --> (b_s x K, time, height, width, channel)
                frames = frames.reshape(input_dim[0]*input_dim[1], input_dim[2], input_dim[3], input_dim[4], input_dim[5]).to(device)
                label = label.reshape(label_dim[0]*label_dim[1]).to(device)

                # Plot some images
                canvas = display_batch_lia(frames)
                if epoch == 0:
                    writer.add_image('Current batch', canvas, 0, dataformats='NCHW')

                # Inference
                score, lia_hidden_rpz = model((frames/255-0.5)*2)


                loss = criterion(score, label)
                loss.backward()
                optimizer.step()
                avg_loss += loss.item()
                writer.add_scalar("Loss/train", loss.item(), global_step=epoch*video_Dataset.__len__()+cnt)
                cnt += 1

                # Store epoch score and label
                # print("argmax item shape : ", np.argmax(score.squeeze().detach().cpu().numpy(), axis=0))
                # print(f"score:{[np.argmax(score.squeeze().detach().cpu().numpy(), axis=0)]} and true label {[label.squeeze().detach().cpu().numpy()]}")
                score_list = np.concatenate((score_list, np.argmax(score.squeeze().detach().cpu().numpy(), axis=1)), axis=0)
                label_list = np.concatenate((label_list, label.squeeze().detach().cpu().numpy()), axis=0)
                videoID_np = np.array(videoID)
                vid_name = np.concatenate((vid_name, videoID_np.T.reshape(-1)))
                

            scheduler.step()    
            loss_func.append(avg_loss)
            # Compute metric
            # print("Label_list: ", label_list.dtype)
            # print("Score_list: ", score_list.dtype)
            # tpr, fpr, thresh = metrics.roc_curve(label_list, score_list)
            # auc = metrics.auc(fpr, tpr)
            
            overall_acc_train = compute_accuracy(label_list, score_list)
            auc = compute_video_level_AUC(label_list, score_list, vid_name)
            fnr_train, fpr_train = calculate_fnr_fpr(score_list, label_list)
            # Tensorboard
            writer.add_scalar("Epoch Loss/train", avg_loss/video_Dataset.__len__(), epoch)
            writer.add_scalar("ACC/train", overall_acc_train, epoch)

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_func,
                }, f"output/{name}/ckpt/latest.pth")
            if epoch%1 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_func,
                    }, f"output/{name}/ckpt/epoch_{epoch}.pth")

            # Test
            pbar_test = tqdm(iter(loader_test), desc='Face mvt analysis/ Test', unit='batch', leave=False)
            score_list, label_list = np.array([]), np.array([])
            if epoch%1 == 0:
                # Test
                score_list_test, label_list_test, vid_name_test = np.array([]), np.array([]), np.array([])
                pbar_test = tqdm(iter(loader_test), desc='Face mvt analysis', unit='batch', leave=False)
                with torch.no_grad():
                    for frames, label, videoID, weights in pbar_test:
                        input_dim, label_dim = frames.shape, label.shape
                        # Put data on GPU
                        frames = frames.reshape(input_dim[0]*input_dim[1], input_dim[2], input_dim[3], input_dim[4], input_dim[5]).to(device)
                        label = label.reshape(label_dim[0]*label_dim[1]).to(device)
                        
                        # Plot some images
                        canvas = display_batch_lia(frames)
                        # writer.add_image('Current batch', canvas, 0, dataformats='NCHW')

                        # Inference
                        # print("Embedding size: ", embeddings.shape)
                        score, _ = model((frames/255-0.5)*2)
                        
                        # Store epoch score and label
                        score_list_test = np.concatenate((score_list_test, np.argmax(score.squeeze().detach().cpu().numpy(), axis=1)), axis=0)
                        label_list_test = np.concatenate((label_list_test, label.squeeze().detach().cpu().numpy()), axis=0)
                        videoID_np = np.array(videoID)
                        vid_name_test = np.concatenate((vid_name_test, videoID_np.T.reshape(-1)))

                        del score
                # Compute test metric
                overall_acc_test = compute_accuracy(label_list_test, score_list_test)
                auc_test = compute_video_level_AUC(label_list_test, score_list_test, vid_name_test)
                fnr_test, fpr_test = calculate_fnr_fpr(score_list_test, label_list_test)

                writer.add_scalars('ACC', {
                    'Train': overall_acc_train,
                    'Test': overall_acc_test
                }, epoch)
                writer.add_scalars('AUC', {
                    'Train': auc,
                    'Test': auc_test
                }, epoch)
                writer.add_scalars('FNR-FPR', {
                    'FNR-Train': fnr_train,
                    'FNR-Test': fnr_test,
                    'FPR-Train': fpr_train,
                    'FPR-Test': fpr_test
                }, epoch)
                score_list, label_list, vid_name = np.array([]), np.array([]), np.array([])

        writer.flush()
    if not training:
        model.eval()
        writer = SummaryWriter(os.path.join('output/', name, 'tensorboard_output'))
        acc_batch = list()
        auc_batch = list()
        K=1
        pbar_kfold = tqdm(range(K), desc='K-Fold testing')
        score_list_test, label_list_test, vid_name_test, hidden_list = np.array([]), np.array([]), np.array([]), []
        for N_test in pbar_kfold:
            # Test
            pbar_test = tqdm(iter(loader_test), desc='Face mvt analysis', unit='batch', leave=False)
            with torch.no_grad():
                for frames, label, videoID, weights in pbar_test:
                    input_dim, label_dim = frames.shape, label.shape
                    # Put data on GPU
                    frames = frames.reshape(input_dim[0]*input_dim[1], input_dim[2], input_dim[3], input_dim[4], input_dim[5]).to(device)
                    # embeddings = embeddings.to(device)
                    label = label.reshape(label.shape[0]*label.shape[1]).to(device).unsqueeze(1)
                    
                    canvas = display_batch_lia(frames)
                    # writer.add_image('Current batch', canvas, 0, dataformats='NCHW')
                    # Inference
                    # print("Embedding size: ", embeddings.shape)
                    score, _ = model((frames/255-0.5)*2)


                    # Store epoch score and label
                    numpy_score = torch.nn.functional.softmax(score.view(score.shape[0]//10, 10, 2).mean(dim=1), dim=1)[:, 1].detach().cpu().numpy().reshape(score.shape[0]//10, 1).squeeze(-1)
                    numpy_label = label.view(label.shape[0]//10, 10, 1).to(torch.float32).mean(dim=1).detach().cpu().squeeze(-1).numpy()
                    score_list_test = np.concatenate((score_list_test, numpy_score), axis=0)
                    label_list_test = np.concatenate((label_list_test, numpy_label), axis=0)
                    
                    videoID_np = np.array(list(videoID[0]))
                    vid_name_test = np.concatenate((vid_name_test, videoID_np.T.reshape(-1)))
            # Compute test metric
            overall_acc_test = compute_accuracy(label_list_test, score_list_test)
            auc_test = compute_video_level_AUC(label_list_test, score_list_test, vid_name_test)
            # tpr, fpr, thresh = metrics.roc_curve(label_list_test, score_list_test)
            # auc_test = metrics.auc(fpr, tpr)
            fnr_test, fpr_test = calculate_fnr_fpr(score_list_test, label_list_test)
            
            # print(f"Accuracy {overall_acc_test}")
            # print(f"AUC {1-auc_test}")
            # print(f"FNR {fnr_test} | FPR {fpr_test}")
            acc_batch.append(overall_acc_test)
            auc_batch.append(auc_test)
            pbar_kfold.set_description(f'K-Fold testing | ACC: {overall_acc_test:.3f} | AUC: {auc_test:.3f} | FNR: {fnr_test:.3f} | FPR: {fpr_test:.3f}')
        auc_test = compute_video_level_AUC(label_list_test, score_list_test, vid_name_test, disp=True)
        fnr_test, fpr_test = calculate_fnr_fpr(score_list_test, label_list_test)
        print(f"K-Fold (K={K}) | ACC: {overall_acc_test:.3f} | AUC: {np.mean(auc_batch):.3f}")
        print(f"FNR= {fnr_test:.3f}, FPR={fpr_test:.3f}")
        
        print(auc_test)
        res_dict = defaultdict()
        res_dict["pred"] = score_list_test.tolist()
        res_dict["label"] = label_list_test.tolist()
        res_dict["vid_name"] = vid_name_test.tolist()
        with open(f"output/{name}/res_on_test_epoch_{args.ckpt.split('_')[-1][:-4]}.json", 'w') as f:
            json.dump(res_dict, f)
