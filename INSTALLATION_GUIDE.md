# Installation Guide

Since this is a Machine Learning project based in python, the installation process is fairly light-weight.

This guide is for DCS machines. We strongly recommend that this code is run on a DCS machine or DCS remote node.

If you are using the code from our Tabula submission, then you will be able to skip the "Downloading the Datasets" and "Downloading the DSNet Pretrained Models" sections of this guide. If you are cloning the code from this repository, you will need to complete all parts of this guide.

## First Time Setup

First, start by cloning the repository to your system. You can skip this step if you are using our Tabula submission code, since you already have the code.

1) Run `git clone https://github.com/Thorin88/cs407-video-summarisation-public.git` in the directory you want the code to be downloaded to.

For the next sections, it is important that you ensure you complete the steps that involve running `module load cs342-python`. This ensures we do not end up with version conflicts for the packages we will be installing. It is also important to note our use of `python3`, not `python`, when invoking commands involving python.

### Installing Flask

First we start by installing Flask. This is slightly more fiddly than usual due to version conflicts with some of the DCS Libraries, however the instructions below should allow you to install the correct version. Since we use `module load cs342-python` later on, pip doesn't seem to install the right version of `flask` considering the modules now loaded. Therefore we need to install a `flask` version which is runnable based on the current versions of all the requirements from the now loaded modules.

1) In a terminal, run `module load cs342-python`.

2) Run `python3 -m pip show flask`. If `flask` is not installed, with this command showing no output or showing warnings, skip to the next step. Else, if the version shown is anything above 1.1.4, you'll need to downgrade or uninstall flask. You can uninstall `flask` using `python3 -m pip uninstall flask.`

3) Run `python3 -m pip install flask==1.1.4`, which should install Flask

### Installing Other Packages

1) In a terminal, run `module load cs342-python`.

The python packages this command puts on PATH should cover most of the packages this project needs to run. However, there are some additional packages you may need if they are not already installed in your standard DCS environment.

2) Run `python3 -m pip install ortools`. This should install a module called `ortools` if this module is not already installed.

Next we want to check if torch and torchvision are installed. If you already know these are installed, you can skip to step to the "Downloading the Datasets" section.

3) Run `python3 -m pip show torchvision`. If you do not see any output or if you see a warning about this module not being found, then continue to step 4. Otherwise, all the required python packages are now installed, and you can move onto the next section.

4) Run `python3 -m pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html`. This should install torch and torchvision.

This should catch everything that `module load cs342-python` does not already load for us, however if when running any code you see errors relating to modules not being found, for example `ImportError: No module named moduleName`, then you can use `python3 -m pip install moduleName` to install the missing module.

### Downloading the Datasets

Our code requires a public dataset to be downloaded before it is able to function. We use the download links that DSNet provide.

1) Navigate to the `frontend/backend/pipeline` directory.

2) Run `mkdir datasets`.

3) Run `cd datasets`.

4) Run `wget https://www.dropbox.com/s/tdknvkpz1jp6iuz/dsnet_datasets.zip`.

If that link did not work, then try the following links to download the zip file:

+ (Baidu Cloud) Link: https://pan.baidu.com/s/1LUK2aZzLvgNwbK07BUAQRQ Extraction Code: x09b
+ (Google Drive) https://drive.google.com/file/d/11ulsvk1MZI7iDqymw9cfL7csAYS0cDYH/view?usp=sharing

5) Run `unzip dsnet_datasets.zip`

6) Run `rm dsnet_datasets.zip`

7) Run `ls`

You should see that the `datasets/` directory now contains 4 dataset files. The file structure from `pipeline/` should now look like this:

```
pipeline
└── datasets/
    ├── eccv16_dataset_ovp_google_pool5.h5
    ├── eccv16_dataset_summe_google_pool5.h5
    ├── eccv16_dataset_tvsum_google_pool5.h5
    ├── eccv16_dataset_youtube_google_pool5.h5
    └── readme.txt
```

### Downloading the DSNet Pretrained Models

For feature extraction, our code uses a pretrained DSNet model. Our single model pipeline will still function without this model, but this model is required for our multi model pipeline.

1) Navigate to the `frontend/backend/pipeline/pretrained_models` directory.

2) Run `wget https://www.dropbox.com/s/2hjngmb0f97nxj0/pretrain_af_basic.zip`

3) Run `unzip pretrain_af_basic.zip`

4) Run `rm pretrain_af_basic.zip`

`frontend/backend/pipeline/pretrained_models` should now contain a directory called `pretrain_af_basic`.

### First Time Setup Complete

You should now be able to begin running the code. You will not need to complete these steps before running the code the next time you load it up.

### Help/Debugging

If you receive any errors regarding `torch` or `torchvision`, including when running the code, then there may be issues with version compatibility. To try and resolve this, first use `python3 -m pip uninstall torch` and `python3 -m pip uninstall torchvision`. Then run `python3 -m pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html`. This command installs a specific version of these libraries which we know to work on DCS.
