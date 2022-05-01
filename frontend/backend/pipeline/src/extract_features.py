import cv2
import numpy as np
import torch
import os

from helpers import init_helper, vsumm_helper, bbox_helper, video_helper
from modules.model_zoo import get_model

from ae2d.ae2d_class import AE2D

twoD_Output = {}

def get_features(name):
    def hook(model, input, output):
        twoD_Output[name] = output.detach()
    return hook

def extract_features_for_video(args,videoFilePath,prints=True,returnFeats=False,ae2dPath=None):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # load model
        if (prints):
            print('Loading DSNet model ...')
        model = get_model(args.model, **vars(args))
        model = model.eval().to(device)
        # For non-cuda runs
        # device = torch.device('cpu')
        # model = model.eval().to(device)
        state_dict = torch.load(args.ckpt_path,
                                map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict)

        if (returnFeats):
            seq = videoFilePath

        else:
            # load video
            if (prints):
                print('Preprocessing source video ...')
            video_proc = video_helper.VideoPreprocessor(args.sample_rate)
            # Picking some frames from the video
            n_frames, seq, cps, nfps, picks = video_proc.run(videoFilePath)
            seq_len = len(seq)
            # print(seq_len)

        if (prints):
            print('Extracting Features ...')
        # 1024D Avg Features
        with torch.no_grad():
            # Input to model
            seq_torch = torch.from_numpy(seq).unsqueeze(0).to(device)

            # v + w for all frames
            # model() will just forward eventually
            features = model(seq_torch)
            # Convert from Tensor on GPU to numpy
            features = features.cpu().numpy()
            # Assuming that averaging the features is fine
            # [0] since Tensor is of shape (1,N,1024)
            feature = np.mean(features[0],axis=0).reshape(1,1024)

        # 1024D -> 2D
        if (args.featureFormat == 'avg2D'):

            model = AE2D(input_shape=1024).to(device)

            model.encoder_output_layer.register_forward_hook(get_features('feats'))

            model = model.eval().to(device)
            # For non-cuda runs
            # device = torch.device('cpu')
            # model = model.eval().to(device)
            ae2dLocation = "pipeline/src/ae2d/pretrainedAE2D.pt"
            if (ae2dPath is not None):
                ae2dLocation = ae2dPath

            state_dict = torch.load(ae2dLocation,
                                    map_location=lambda storage, loc: storage)
            model.load_state_dict(state_dict)

            with torch.no_grad():

                feature = torch.from_numpy(feature).unsqueeze(0).to(device)
                # print(feature.shape)
                feature = model(feature)
                feature = twoD_Output['feats'].cpu().numpy()
                feature = feature.reshape((2,))
                # print(feature)

        if (prints):
            print("Feature Shape: " + str(feature.shape))
        # print(feature.dtype)

        # Skips saving features
        if (returnFeats):
            return feature

        video_name = videoFilePath.split('/')[-1]
        feat_file = '%s_%s_feat_%s.npy' % (args.model, video_name, args.featureFormat)
        saveTo = os.path.join(args.save_path, feat_file)
        np.save(saveTo, feature)
        if (prints):
            print("Features saved to: " + str(saveTo))

def main():
    args = init_helper.get_arguments()

    print("[ARGS] Treating save_path as a directory...")

    if (args.featureFormat == 'avg1024'):
        print('[ARGS] Generating 1024D Average Features')
    elif (args.featureFormat == 'avg2D'):
        print('[ARGS] Generating 2D Average Features')

    if (args.batchExtract):
        print("[ARGS] Treating source as csv of video file paths...")

        video_list_file = args.source
        f = open(video_list_file, 'r')
        data_list = f.readlines()

        for vid, vline in enumerate(data_list):
            video_path = vline.split()[0]
            # print(video_path)
            video_name = video_path.split('/')[-1]

            # Call
            extract_features_for_video(args,video_path,prints=False)

            if ( (vid+1) % 5 == 0):
                print('%04d/%04d are done' % (vid + 1, len(data_list)))


    elif (not args.batchExtract):
        extract_features_for_video(args,args.source)

    print("Done.")

    # Skipped since we only want features

    # print('Writing summary video ...')
    #
    # # load original video
    # cap = cv2.VideoCapture(args.source)
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps = cap.get(cv2.CAP_PROP_FPS)
    #
    # # create summary video writer
    # fourcc = cv2.VideoWriter_fourcc(*'a\0\0\0') # https://stackoverflow.com/questions/29703059/how-to-export-video-as-mp4-using-opencv (Works on Windows)
    # out = cv2.VideoWriter(args.save_path, fourcc, fps, (width, height))
    #
    # frame_idx = 0
    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #
    #     if pred_summ[frame_idx]:
    #         out.write(frame)
    #
    #     frame_idx += 1
    #
    # out.release()
    # cap.release()


if __name__ == '__main__':
    main()
