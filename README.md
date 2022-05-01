# CS407 Project Repository

Please see the install guide, INSTALLATION_GUIDE.md, and the user guide, USER_GUIDE.md. The install guide will help you set up the code when you first clone the repository. The user guide details in what ways you can use our code, and how to do so.

This file will indicate the general layout of the repository, where to find key files, and details about code we based our own off.

### `frontend/` ###

Contains our code for the web-based UI. `static/`, `templates/` and `main.py` make up the code for the web-based UI.

The `2FPS_Videos` directory contains 2FPS versions of the original SumMe and TVSum Datasets. These videos are **not** the datasets our models train and evaluate on, we use a preprocessed dataset. These videos are included to offer the option of demonstrating raw, batch feature extraction using our code, as well as a bank of videos that the user could test our inference code on. When using the UI, please either use your own videos, or use the example videos in `frontend/backend/pipeline/custom_data/`. The videos in `2FPS_Videos/` **should not** be given as input to our UI as they are downsampled to 2FPS. This is explained more in the "Using the UI" section of `USER_GUIDE.md`.

### `frontend/backend/` ###

Contains the code for our models, training, evaluation, feature extraction and visualisation.

`clustering.ipynb` is a small notebook which has been left included to visualise some example clustering with KMeans and DBSCAN. It demonstrates how we could have used DBSCAN inductively, along with why we chose to use KMeans over DBSCAN.

`dsnet_feature_extraction/` contains features which have been precomputed using our pipelines. The feature extraction commands we discuss in the user guide currently save to these locations, unless the save path parameters are altered.

### `frontend/backend/pipeline/` ###

Is the core directory for our model related code. This directory was modified
and adapted from the DSNet repository (https://github.com/li-plus/DSNet). This area of the repository is where most of our core code resides.

`frontend/backend/pipeline/custom_data/` is a directory that contains some example raw videos. These files are from DSNet, we use these videos in our demonstrations of our software.

`frontend/backend/pipeline/datasets/` is a directory that will exist once the set of instructions in `INSTALLATION_GUIDE.md` are completed. This directory contains the public, preprocessed SumMe, TVSum, OVP and YouTube datasets that we will be using for our evaluation benchmarking.

`frontend/backend/pipeline/models/` contains our pretrained models. Out of the box, this directory contains a pretrained single model and multi model pipeline (trained on under the SumMe Conical Setting, split 1). Other models were not uploaded to the Github to save space, but they can be trained/retrained using `frontend/backend/pipeline/src/train_notebook.ipynb` (how to use this file is covered in `USER_GUIDE.md`). Models trained using our code end up in this directory.

`frontend/backend/pipeline/pretrained_models/` contains a pretrained DSNet Anchor Free model. This directory starts empty, but this model is downloaded when following the instructions in `INSTALLATION_GUIDE.md`. This model is used in our feature extraction process, which is used exclusively by our clustering component.

`frontend/backend/pipeline/splits/` contains data about training and validation splits, which the base training pipeline we adapted from DSNet can interpret. This data has been used work evaluating using this benchmark, such as DSNet, VASNET (https://github.com/ok1zjf/VASNet) and the papers before them.

`frontend/backend/pipeline/LICENSE` is the license from the original DSNet repository, which needs to be included in code which uses/modifies DSNet's code.

### `frontend/backend/pipeline/src/` ###

This directory contains our model definitions as well as our training, evaluation and visualisation code.

Each model is defined in its own directory in this directory, for example `basic_NN` contains code for our RNN based model. The directories `anchor_based` and `anchor_free` are from DSNet, for their models. `basic_NN`, `basic_GRU` and `feat_dsnet` our models we created. In each of our model directories, we have the following files:

- [model_name].py: The class definition of the model
- train.py: Not to be confused with src/train.py. This file contains training code specific to this model, which the driver training functions will use, such as src/train.py or in train_notebook.ipynb.

`extract_features.py` is code we have written for video feature extraction. This code is invoked by our training functions and our inference pipelines. As mentioned in `USER_GUIDE.md`, this file can also be run via the command line, and supports various options regarding the dimension of features to return, as well as processing videos in batch.

`train_notebook.ipynb` is where our models are trained, evaluated and visualised, including representative frame visualisations. More details on this file are in `USER_GUIDE.md` and then file itself. Compared to using the command line, this file was created to make the process of training more interactive, and users can experiment with the same parameters we experimented with. `example_frames` and `example_visualisations` are filled with the results of visualisations when the code in this notebook is run.

`ae2d` contains model definitions and training code for our auto encoder. Pretrained auto encoders and precomputed features are also stored in this directory.

`my_infer.py` is the file that is invoked when video summaries are to be generated using our models. The web-based UI essentially creates a more visual interface to use this file, however `USER_GUIDE.md` also details how to use this file via the command line.

#### External Code ####

It is worth noting which files in this section are originally from DSNet, as well as which we been written/adapted for our models.

The main DSNet files we adapted our own code from are `infer.py`, `evaluate.py` and `train.py`. `infer.py` is specifically for DSNet models and was written by DSNet. `my_infer.py` is for our models. `evaluate.py` is specifically for DSNet models and was written by DSNet. `my_evaluate.py` is for our models. We chose to make separate files for `my_infer.py` and `my_evaluate.py`, since our model's pipeline is very different to DSNet's when it comes to inference and evaluation. We leave the DSNet version of these files in the repository to support training/evaluation of DSNet models if the user desired it, but these files can be removed from the repository without any loss of function of our code.

`train.py` is slightly different, as we were able to adapt it for our models without needing to make a completely separate version. However, as `USER_GUIDE.md` mentions, `train.py` was depreciated for use with our models in favour of `train_notebook.ipynb`. `my_evaluate.py` is also depreciated for the same reason, with our evaluation code being redefined in `train_notebook.ipynb`. This was because we wanted to move towards a more visual and interactive training process. `train.py` has been left included in case the user wanted to tune/train a DSNet model, or train our single model pipeline via the command line, with the required details on how to do this specified in `USER_GUIDE.md`. `modules`, originally from DSNet, is used when training DSNet models so has been left included to support this but is not used by our models.

`kts` and `helpers` contain important code that is specific to this benchmark's training and evaluation pipeline. These have been written over the years by a number of works, but we obtained them from DSNet's repository. `kts` contains code for the KTS algorithm. `helpers` contains code for a number of things, such as video preprocessing, summary downsampling and video dataset loaders. `helpers` also contains a very useful file `helpers/init_helper.py`, which manages command line arguments given to our files. We modified `helpers/init_helper.py` to support our own new command line arguments.

`make_dataset.py`, `make_shots.py`, `make_split.py` are also utilities that have been developed for this benchmark which we also obtained from the DSNet repository. We do not use them, however these can be used to create new datasets from a user's own video dataset. Instructions on how to use these files can be found in the DSNet repository, but are not needed for our code to function.

## Acknowledgements

We thank DSNet (https://github.com/li-plus/DSNet) for their updated training and evaluation pipeline, which we used as the base of our code and adapted upon. We would also like to thank them for their links to the preprocessed public datasets and their pretrained models.

We would also like to thank the below open-source repositories for their contributions to DSNet's original code, and hence our code base.

+ Thank [KTS](https://github.com/pathak22/videoseg/tree/master/lib/kts) for the effective shot generation algorithm.
+ Thank [DR-DSN](https://github.com/KaiyangZhou/pytorch-vsumm-reinforce) for the pre-processed public datasets.
+ Thank [VASNet](https://github.com/ok1zjf/VASNet) for the training and evaluation pipeline.

The 2FPS versions of the original SumMe and TVSum Datasets found in `2FPS_Videos` were computed from the datasets found at the following links:

+ [SumMe] https://gyglim.github.io/me/vsum/index.html
+ [TVSum] http://people.csail.mit.edu/yalesong/tvsum/ (https://github.com/yalesong/tvsum)
