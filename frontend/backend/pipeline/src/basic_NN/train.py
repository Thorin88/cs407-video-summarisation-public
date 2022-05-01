import logging

import torch

from basic_NN.basic_NN import BasicRNN
from anchor_free.losses import calc_ctr_loss, calc_cls_loss, calc_loc_loss
# Using a different evaluate function
from my_evaluate import evaluate
from helpers import data_helper, vsumm_helper, video_helper

from extract_features import extract_features_for_video
from helpers import init_helper

# Clustering + Classification
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

#instantiate logger
logger = logging.getLogger()

#function to train Neural network
def train(args, split, save_path):
    #set pytorch to use the GPU if one is available, and the cpu if one is not
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # instantiate RNN with necessary dimensions, taking 1024 input Features
    # and returning a single value.
    model = BasicRNN(input_size=1024,output_size=1,hidden_dim=128,n_layers=3)
    model = model.to(device)

    # set model to training mode
    model.train()

    # get list of trainable parameters in th model
    parameters = [p for p in model.parameters() if p.requires_grad]
    # instantiate optimiser
    optimizer = torch.optim.Adam(parameters, lr=5e-5,
                                 weight_decay=5e-5)

    # set value for early stopping
    max_val_fscore = -1

    #store training split in data loader and shuffle data
    train_set = data_helper.VideoDataset(split['train_keys'])
    train_loader = data_helper.DataLoader(train_set, shuffle=True)
    #store validation split in data loader and shuffle data
    val_set = data_helper.VideoDataset(split['test_keys'])
    val_loader = data_helper.DataLoader(val_set, shuffle=False)

    #################### CLUSTERING PREP ####################
    # python3 train.py basic --model-dir ../models/ab_basic --splits ../splits/tvsum_aug.yml

    # instantiate argument passer store arguments for later funcitons
    parserInFunc = init_helper.get_parser()
    # Use a sample rate of 1, since seq is already downsampled, so taking every
    # frame of a 15 sample rate result.
    pathToVFT = "../../"
    # args including unneeded args
    # toPass = ["feat-anchor-free","--ckpt-path",str(pathToVFT)+"pipeline/pretrained_models/pretrain_af_basic/checkpoint/tvsum.yml.0.pt","--source","./../../2FPS_Videos/2FPS_TVSum/video/2_fps_EE-bNr36nyA.mp4","--save-path","./dsnet_feature_extraction/features/TVSum","--nms-thresh","0.4","--sample-rate","1","featureFormat","avg2D"]
    toPass = ["feat-anchor-free","--ckpt-path",str(pathToVFT)+"pipeline/pretrained_models/pretrain_af_basic/checkpoint/tvsum.yml.0.pt","--source","","--save-path","","--nms-thresh","0.4","--sample-rate","1","--featureFormat","avg2D"]
    argsInFunc = parserInFunc.parse_args(toPass)

    #path to autoencoder
    ae2dPath = "ae2d/pretrainedAE2D.pt"

    # So intended to be called from backend
    # python3 pipeline/src/extract_features.py

    #lists to store video keys and features
    collectedTrainKeys = []
    collectedTrainFeatures = []
    #extract 2D features for each video in train loader
    for mappedKey, seq, _, _, _, _, _, _ in train_loader:
        collectedTrainFeatures.append( extract_features_for_video(argsInFunc,seq,prints=False,returnFeats=True,ae2dPath=ae2dPath) )
        collectedTrainKeys.append(mappedKey)

    # np.savetxt("testFeatures",collectedTrainFeatures)

    print("Number of Training Features to cluster for this split: " + str(len(collectedTrainFeatures)))
    print("Feature Shape:",collectedTrainFeatures[0].shape)
    print("Number of Training Keys:",len(collectedTrainKeys))


    # DBSCAN
    # db = DBSCAN(eps=0.15, min_samples=5).fit(collectedTrainFeatures)
    # labels = db.labels_
    # n_clusters = len(set(labels)) - (1 if -1 in labels_db else 0)

    # KMeans clustering to categorise videos
    KM_model = KMeans(n_clusters=2)
    km = KM_model.fit(collectedTrainFeatures)
    labels = km.labels_
    n_clusters = KM_model.n_clusters

    print("Labels:",labels)
    print("Number of clusters:",n_clusters)

    # Create mapping from collectedTrainKeys to cluster classes
    trainMapping = dict()
    for i, key in enumerate(collectedTrainKeys):
        trainMapping[key] = labels[i]

    print("Train Label Mapping:",trainMapping)

    #lists to store video keys and features
    collectedValKeys = []
    collectedValFeatures = []
    #extract 2D features for each video in val loader
    for mappedKey, seq, _, _, _, _, _, _ in val_loader:
        collectedValFeatures.append( extract_features_for_video(argsInFunc,seq,prints=False,returnFeats=True,ae2dPath=ae2dPath) )
        collectedValKeys.append(mappedKey)

    print("Number of Validation Features to cluster for this split: " + str(len(collectedValFeatures)))
    print("Number of Validation Keys:",len(collectedValKeys))



    # allocate videos in val loader to relevant cluster
    val_labels = km.predict(collectedValFeatures)
    print("Validation Labels:",val_labels)

    # Create mapping from collectedValKeys to cluster classes
    valMapping = dict()
    for i, key in enumerate(collectedValKeys):
        valMapping[key] = val_labels[i]

    print("Validation Label Mapping:",valMapping)

    # Call feature extraction
    # Can cluster on training data
    # Predict clusters of val_loader stuff

    #################### END CLUSTERING PREP ####################

    # Can also cluster, then classifiy each time we have an example. Avoids
    # mapping but more expensive.

    #for each epoch
    for epoch in range(args.max_epoch):
        #set model to training mode
        model.train()
        #keep track of average loss
        stats = data_helper.AverageMeter('loss')

        #for each video in train loader
        for mappingKey, seq, gtscore, change_points, n_frames, nfps, picks, _ in train_loader:

            #get target keyshot summary (not used during training)
            #only used to check if a valid summary can be generated from a given videos ground truth
            keyshot_summ = vsumm_helper.get_keyshot_summ(
                gtscore, change_points, n_frames, nfps, picks)
            target = vsumm_helper.downsample_summ(keyshot_summ)

            #skip video if target summary could not be generated
            if not target.any():
                continue

            #convert seq (preprocessed video frames) to tensor on relevant device
            seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)

            #run current video through model
            output = model(seq)
            #store predicted output
            preds = output[0]
            # (N,1) tensor

            #convert ground truth to tensor on relevant device
            gtscore = torch.tensor(gtscore, dtype=torch.float32).to(device).view((gtscore.shape[0],1))


            # Update 1
            # calculare MSE loss for the current video
            mse_loss = torch.nn.MSELoss()
            loss = mse_loss(preds,gtscore)


            # Update weights
            optimizer.zero_grad()
            # backpropagate loss and update weights accordingly
            loss.backward()
            optimizer.step()
            #update stats with most recent loss
            stats.update(loss=loss.item())

        # Using our own eval to evaluate model performance
        val_fscore, _ = evaluate(model, val_loader, args.nms_thresh, device)

        # Save best preforming model over the training epochs
        if max_val_fscore < val_fscore:
            max_val_fscore = val_fscore
            torch.save(model.state_dict(), str(save_path))

        # torch.save(model.state_dict(), str(save_path))
        #log current epoch
        logger.info(f'Epoch: {epoch}/{args.max_epoch} '
                    f'Loss: {stats.loss:.4f} '
                    f'F-score current/max: {val_fscore:.4f}/{max_val_fscore:.4f}')

    #return best fscore found during training
    return max_val_fscore
