# C2FPL

# 
[**A Coarse-to-Fine Pseudo-Labeling (C2FPL) Framework for Unsupervised Video Anomaly Detection (WACV 2024)**](https://arxiv.org/pdf/2310.17650.pdf)





<p align="center">
<img src="imgs/wacv2024.png" width="1050">
</p>



## Training

### Setup

**Please download the concatenated extracted I3d features for XD-Violence and UCF-Crime dataset from links below:**



The following files need to be adapted in order to run the code on your own machine:
- Change the file paths to the downloaded features above in `concatenated/concat_UCF.npy` and `concatenated/Concat_test_10.npy`.
- Feel free to change the hyperparameters in `option.py`

### Train and test 
After the setup, simply run the following commands: 
```shell
sh train.sh
sh test.sh
```