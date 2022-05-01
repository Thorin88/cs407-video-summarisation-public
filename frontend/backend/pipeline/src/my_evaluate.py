import logging
from pathlib import Path

import numpy as np
import torch

from helpers import init_helper, data_helper, vsumm_helper, bbox_helper
from modules.model_zoo import get_model

#instantiate logger
logger = logging.getLogger()

# function to evaluate model performance using keyshot sumamry instead of raw ground truth
def evaluate(model, val_loader, nms_thresh, device):
    #set model to eval mode to prevent weight from being updated
    model.eval()
    #keep track of fscore and diversity
    stats = data_helper.AverageMeter('fscore', 'diversity')

    #with tensor gradients dissabled (no training):
    with torch.no_grad():
        #for each video in validation set
        for test_key, seq, _, cps, n_frames, nfps, picks, user_summary in val_loader:

            #get number of frames in preprocessed video
            seq_len = len(seq)
            # store preprocessed frames in tensor on relevant device
            seq_torch = torch.from_numpy(seq).unsqueeze(0).to(device)

            # Predicted Frame Importance Scores
            output = model.predict(seq_torch)
            # generate keyshot summary based upon predicted frame level improatance scores
            pred_summ = vsumm_helper.get_keyshot_summ(
                output, cps, n_frames, nfps, picks)

            # set evaluation metric to average if the video is from the TVSum dataset,
            # and max otherwise (following existing benchmarks)
            eval_metric = 'avg' if 'tvsum' in test_key else 'max'
            # calculate fscore of predicted summary compared to user generated ground truth
            fscore = vsumm_helper.get_summ_f1score(
                pred_summ, user_summary, eval_metric)


            #downsample predicted summary
            pred_summ = vsumm_helper.downsample_summ(pred_summ)

            # compute diversity
            diversity = vsumm_helper.get_summ_diversity(pred_summ, seq)
            stats.update(fscore=fscore, diversity=diversity)

    # return fubak average fscore and diversity
    return stats.fscore, stats.diversity


def main():
    #intialise object to store arguments
    args = init_helper.get_arguments()

    # initialise logger
    init_helper.init_logger(args.model_dir, args.log_file)
    #initialiase random seed
    init_helper.set_random_seed(args.seed)

    #print function arguments
    logger.info(vars(args))

    #create model
    model = get_model(args.model, **vars(args))
    # pass model to relevant device and set to eval mode to prevent weights from
    #being updated
    model = model.eval().to(args.device)

    #for each validation setting
    for split_path in args.splits:
        #get path to data for current validation setting
        split_path = Path(split_path)
        #load current data split
        splits = data_helper.load_yaml(split_path)
        #keep track of average  fscore and diversity
        stats = data_helper.AverageMeter('fscore', 'diversity')

        #for each fold in cross validation
        for split_idx, split in enumerate(splits):
            # get location of model trained for this evaluation setting on current data split
            ckpt_path = data_helper.get_ckpt_path(args.model_dir, split_path, split_idx)
            # load weights of model trained for this evaluation setting on current data split
            state_dict = torch.load(str(ckpt_path),
                                    map_location=lambda storage, loc: storage)
            # set model weights to pre trained values
            model.load_state_dict(state_dict)

            #store validation set in data loader
            val_set = data_helper.VideoDataset(split['test_keys'])
            val_loader = data_helper.DataLoader(val_set, shuffle=False)

            #call evaluation function and store fscore and diversity
            fscore, diversity = evaluate(model, val_loader, args.nms_thresh, args.device)
            #update running average fscore and diversity
            stats.update(fscore=fscore, diversity=diversity)
            #log fscore and diversity averages for current fold
            logger.info(f'{split_path.stem} split {split_idx}: diversity: '
                        f'{diversity:.4f}, F-score: {fscore:.4f}')
        #log fscore and diversity averages for current setting
        logger.info(f'{split_path.stem}: diversity: {stats.diversity:.4f}, '
                    f'F-score: {stats.fscore:.4f}')


if __name__ == '__main__':
    main()
