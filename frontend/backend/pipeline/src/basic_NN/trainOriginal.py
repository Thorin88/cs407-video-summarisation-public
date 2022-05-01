import logging

import torch

from basic_NN.basic_NN import BasicRNN
from anchor_free.losses import calc_ctr_loss, calc_cls_loss, calc_loc_loss
# Using a different evaluate function
from my_evaluate import evaluate
from helpers import data_helper, vsumm_helper, video_helper

from extract_features import extract_features_for_video
from helpers import init_helper

logger = logging.getLogger()


def train(args, split, save_path):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Need to create mulitple of these
    model = BasicRNN(input_size=1024,output_size=1,hidden_dim=128,n_layers=3)
    model = model.to(device)

    model.train()

    parameters = [p for p in model.parameters() if p.requires_grad]
    # Just used the default ones in DSNet
    optimizer = torch.optim.Adam(parameters, lr=5e-5,
                                 weight_decay=5e-5)


    max_val_fscore = -1

    # Gets different training scores to original train.py because we use the
    # first random loader, which is what the original would have started with.
    train_set = data_helper.VideoDataset(split['train_keys'])
    train_loader = data_helper.DataLoader(train_set, shuffle=True)

    val_set = data_helper.VideoDataset(split['test_keys'])
    val_loader = data_helper.DataLoader(val_set, shuffle=False)

    #################### CLUSTERING PREP ####################
    # python3 train.py basic --model-dir ../models/ab_basic --splits ../splits/tvsum_aug.yml

    parserInFunc = init_helper.get_parser()
    # Use a sample rate of 1, since seq is already downsampled, so taking every
    # frame of a 15 sample rate result.
    pathToVFT = "../../"
    # args including unneeded args
    # toPass = ["feat-anchor-free","--ckpt-path",str(pathToVFT)+"pipeline/pretrained_models/pretrain_af_basic/checkpoint/tvsum.yml.0.pt","--source","./../../2FPS_Videos/2FPS_TVSum/video/2_fps_EE-bNr36nyA.mp4","--save-path","./dsnet_feature_extraction/features/TVSum","--nms-thresh","0.4","--sample-rate","1","featureFormat","avg2D"]
    toPass = ["feat-anchor-free","--ckpt-path",str(pathToVFT)+"pipeline/pretrained_models/pretrain_af_basic/checkpoint/tvsum.yml.0.pt","--source","","--save-path","","--nms-thresh","0.4","--sample-rate","1","--featureFormat","avg2D"]
    argsInFunc = parserInFunc.parse_args(toPass)

    # So intended to be called from videoFeaturesTest
    # python3 pipeline/src/extract_features.py

    # print(argsInFunc.model)
    # print(argsInFunc.sample_rate)
    collectedTrainKeys = []
    collectedTrainFeatures = []
    for mappedKey, seq, _, _, _, _, _, _ in train_loader:
        collectedTrainFeatures.append( extract_features_for_video(argsInFunc,seq,prints=False,returnFeats=True,ae2dPath="ae2d/pretrainedAE2D.pt") )
        collectedTrainKeys.append(mappedKey)

    print("Number of Training Features to cluster for this split: " + str(len(collectedTrainFeatures)))
    print("Number of Training Keys:",len(collectedTrainKeys))

    # TODO - Cluster
    # Create mapping from collectedTrainKeys to cluster classes

    collectedValKeys = []
    collectedValFeatures = []
    for mappedKey, seq, _, _, _, _, _, _ in val_loader:
        collectedValFeatures.append( extract_features_for_video(argsInFunc,seq,prints=False,returnFeats=True,ae2dPath="ae2d/pretrainedAE2D.pt") )
        collectedValKeys.append(mappedKey)

    print("Number of Validation Features to cluster for this split: " + str(len(collectedValFeatures)))
    print("Number of Validation Keys:",len(collectedValKeys))

    # TODO - Cluster
    # Create mapping from collectedValKeys to cluster classes

    # Call feature extraction
    # Can cluster on training data
    # Predict clusters of val_loader stuff

    #################### END CLUSTERING PREP ####################

    # Can also cluster, then classifiy each time we have an example. Avoids
    # mapping but more expensive.

    for epoch in range(args.max_epoch):
        model.train()
        stats = data_helper.AverageMeter('loss')

        for mappingKey, seq, gtscore, change_points, n_frames, nfps, picks, _ in train_loader:

            # TODO - Extract Video's cluster ID/class using mappingKey

            # print(mappingKey,n_frames)

            # Debug
            # video_proc = video_helper.VideoPreprocessor(args.sample_rate)
            # cps2, nfps2, picks2 = video_proc.kts(n_frames, seq)
            # print(cps2, nfps2, picks2)
            #
            # return

            keyshot_summ = vsumm_helper.get_keyshot_summ(
                gtscore, change_points, n_frames, nfps, picks)
            target = vsumm_helper.downsample_summ(keyshot_summ)

            if not target.any():
                continue

            seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)

            gt = target

            # print(seq.shape)
            # TODO - Need to put the seq into the model assigned to the video's cluster
            output = model(seq)
            preds = output[0]
            # (N,1) tensor

            # preds = torch.tensor(preds, dtype=torch.float32).to(device).view((preds.shape[0],1))
            gtscore = torch.tensor(gtscore, dtype=torch.float32).to(device).view((gtscore.shape[0],1))
            # (N,1) tensor

            # gt = torch.tensor(gt, dtype=torch.float32).to(device)
            # print(gt.shape)

            # sure
            # Changing this to a better one probably makes sense
            # cls_loss = calc_cls_loss(preds, gt, args.cls_loss)
            # loss = cls_loss

            # Update 1
            mse_loss = torch.nn.MSELoss()
            loss = mse_loss(preds,gtscore)

            # Update 2
            # Since my output layer is sigmoid
            # Cannot get this to work properly
            # m = torch.nn.Sigmoid()
            # # criterion = torch.nn.BCELoss()
            # criterion = torch.nn.BCEWithLogitsLoss()
            # target = torch.tensor(target, dtype=torch.float32).to(device).view((target.shape[0],))
            # # loss = criterion(preds,target)
            #
            # preds = preds.squeeze().cpu().detach().numpy()
            # pred_summ = vsumm_helper.downsample_summ(vsumm_helper.get_keyshot_summ(
            #     preds, change_points, n_frames, nfps, picks))
            # pred_summ = torch.tensor(pred_summ, dtype=torch.float32).to(device).view((pred_summ.shape[0],))
            # loss = criterion(pred_summ,target)
            # loss = torch.tensor(loss, requires_grad = True)

            # criterion = nn.CrossEntropyLoss()
            # loss = criterion(output, target)

            # Update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            stats.update(loss=loss.item())

        # Using our own eval
        # TODO - Pass mulitple models and average them in some way
        val_fscore, _ = evaluate(model, val_loader, args.nms_thresh, device)

        # Save best preforming model over the training epochs
        if max_val_fscore < val_fscore:
            max_val_fscore = val_fscore
            torch.save(model.state_dict(), str(save_path))

        # torch.save(model.state_dict(), str(save_path))

        logger.info(f'Epoch: {epoch}/{args.max_epoch} '
                    f'Loss: {stats.loss:.4f} '
                    f'F-score current/max: {val_fscore:.4f}/{max_val_fscore:.4f}')

    return max_val_fscore
