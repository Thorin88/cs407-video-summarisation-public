
import cv2
import numpy as np
import torch
import pickle

from helpers import init_helper, vsumm_helper, bbox_helper, video_helper
from modules.model_zoo import get_model
from basic_NN.basic_NN import BasicRNN
from basic_GRU.basic_GRU import BasicGRU

from extract_features import extract_features_for_video

import sys

# Works out whether the weights loaded are for an RNN or GRU
def model_base_to_use(state_dict):

    is_rnn = any(list(map(lambda s : "rnn" in s, list(state_dict.keys()))))
    if is_rnn:
        return BasicRNN
    else:
        return BasicGRU

def main():
    args = init_helper.get_arguments()

    print(args, file=sys.stderr)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Running in mode:",args.pipeline_mode, file=sys.stderr)

    print("Looking for models in:",args.ckpt_path, file=sys.stderr)

    # load model
    if args.pipeline_mode == "single":

        if (args.model == "basic"):

            state_dict = torch.load(args.ckpt_path,
                                    map_location=lambda storage, loc: storage)

            basic_base = model_base_to_use(state_dict)
            model = basic_base(input_size=1024,output_size=1,hidden_dim=128,n_layers=3)
            print("Basic Model Selected.")

        model = model.eval().to(device)
        model.load_state_dict(state_dict)
        print(model, file=sys.stderr)

    # If using multiple models, need to load the clustering object created during
    # training and the models for each cluster class.
    elif args.pipeline_mode == "multi":

        str_path = str(args.ckpt_path)

        # Get the number before the ".pt"
        split_to_use = str_path.split(".pt")[0][-1]

        models = dict()
        # KMeans alwys returns two cluster classes
        classes = [0,1]
        for c in classes:
            model_path = str_path.split(".yml")[0]+"_multi_"+str(c)+str_path.split(".yml")[1]

            if (args.model == "basic"):

                state_dict = torch.load(model_path,
                                        map_location=lambda storage, loc: storage)

                basic_base = model_base_to_use(state_dict)
                model = basic_base(input_size=1024,output_size=1,hidden_dim=128,n_layers=3)
                print("Basic Model Selected.")

            model = model.eval().to(device)
            model.load_state_dict(state_dict)
            print("Model loaded from:",model_path,file=sys.stderr)

            models[c] = model

        print(models, file=sys.stderr)

        # Load clusterer
        clusterer_path = str_path.split(".yml")[0]+"_clusterer"+str_path.split(".yml")[1]

        with open(clusterer_path, "rb") as f:
            km = pickle.load(f)

        print(km, file=sys.stderr)


    # load video
    print('Preprocessing source video ...')
    video_proc = video_helper.VideoPreprocessor(args.sample_rate)
    n_frames, seq, cps, nfps, picks = video_proc.run(args.source)
    seq_len = len(seq)

    print('Predicting summary ...')
    with torch.no_grad():
        seq_torch = torch.from_numpy(seq).unsqueeze(0).to(device)

        # Extracting features for the video to work out its class, if using the
        # multi model pipeline.
        if args.pipeline_mode == "multi":

            parserInFunc = init_helper.get_parser()
            # Arguements required by extract_featuresfor_video()
            pathToVFT = "./backend/"
            toPass = ["feat-anchor-free",
                  "--ckpt-path",str(pathToVFT)+"pipeline/pretrained_models/pretrain_af_basic/checkpoint/tvsum.yml.0.pt",
                  "--source","",
                  "--save-path","",
                  "--nms-thresh","0.4",
                  "--sample-rate","1",
                  "--featureFormat","avg2D"]
            argsInFunc = parserInFunc.parse_args(toPass)
            # Auto encoder to use
            ae2dPath = "./backend/pipeline/src/ae2d/pretrainedAE2D.pt"
            video_features = extract_features_for_video(argsInFunc,seq,prints=False,returnFeats=True,ae2dPath=ae2dPath)

            video_class = km.predict([video_features])[0]
            print("Video was of class:",video_class,file=sys.stderr)

            # Selecting the model trained on the class of video this video
            # was found to belong to.
            model = models[video_class]

        # Predicted Frame Importance Scores
        output = model.predict(seq_torch)
        # Converting to a keyshot summary
        pred_summ = vsumm_helper.get_keyshot_summ(
            output, cps, n_frames, nfps, picks)

    print('Writing summary video ...')

    # load original video
    cap = cv2.VideoCapture(args.source)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # DCS
    print('Using OpenCV version ', cv2.__version__)
    # fourcc = cv2.VideoWriter_fourcc(*'VP08')

    # DCS has issues with certain codecs not being supported, however .webm works.
    codec = 'VP08'
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(args.save_path, fourcc, fps, (width, height), True)

    print("Writing to " + str(args.save_path) + "...")

    frame_idx = 0
    framesWritten = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if pred_summ[frame_idx]:
            out.write(frame)
            framesWritten += 1

        frame_idx += 1

    out.release()
    cap.release()

    print("Done. (" + str(framesWritten) + " frames written)", file=sys.stderr)

if __name__ == '__main__':
    main()
