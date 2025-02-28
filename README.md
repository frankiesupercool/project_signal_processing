# Master Project - Signal Processing

This repo contains a reimplementation of the paper 
"Reading to Listen at the Cocktail Party: Multi-Modal Speech Separation." by Rahimi et al. 
The project focuses on separating speech using both audio and visual cues, following the methods outlined in the 
original paper.


## Setup

To setup the conda environment for this project use the [environment.yml](environment.yml) by running: \
`conda env create -f environment.yml`

### Data

We trained with the *LRS3* data set. 
Noise was added either by different speakers or by interfering noise computed from the 
*DNS Challenge at ICASSP 2023*

Exact expected folder structure can be obtained from the [config.py](config.py) file

- *LRS3* data is expected to be in `data/pretrain`, `data/test`, `data/test` folders, exact folder struc
- Noise data can be downloaded via [this script](download_scripts/download-noise-data.sh)

Video preprocessing followed the same approach as *Lipreading Using Temporal Convolutional Networks* as mentioned by 
Rahimi et al.
- If `gdown` is installed [this download script](download_scripts/download-lrw-pth.py) can be used

- Optionally it can be downloaded directly from  
  `https://drive.google.com/file/d/1TGFG0dW5M3rBErgU8i0N7M1ys9YMIvgm/view` to the *video_encoding* folder


## Best Checkpoint

We provide our best checkpoint in the checkpoint folder. 
Due to its large size we uploaded it via Git LFS.

To download our best checkpoint checkout this repository, install `git lfs install` and pull from LFS `git lfs pull`.
For usage the downloaded checkpoint must either be put to the expected folder from [config.py](config.py) or the path in
config.py must be changed to the checkpoint location.

## References

Original paper *Reading to listen at the cocktail party: Multi-modal speech separation*:
```
@inproceedings{rahimi2022reading,
  title={Reading to listen at the cocktail party: Multi-modal speech separation},
  author={Rahimi, Akam and Afouras, Triantafyllos and Zisserman, Andrew},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10493--10502},
  year={2022}
}
```
LRS3 Data set used for training:
```
@article{LRS3,
  author       = {Triantafyllos Afouras and
                  Joon Son Chung and
                  Andrew Zisserman},
  title        = {{LRS3-TED:} a large-scale dataset for visual speech recognition},
  journal      = {CoRR},
  volume       = {abs/1809.00496},
  year         = {2018},
  url          = {http://arxiv.org/abs/1809.00496},
  eprinttype    = {arXiv},
  eprint       = {1809.00496},
  timestamp    = {Fri, 05 Oct 2018 11:34:52 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-1809-00496.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

Noise data set origin *Icassp 2023 deep noise suppression challenge*:
```
@article{dubey2024icassp,
  title={Icassp 2023 deep noise suppression challenge},
  author={Dubey, Harishchandra and Aazami, Ashkan and Gopal, Vishak and Naderi, Babak and Braun, Sebastian and Cutler, Ross and Ju, Alex and Zohourian, Mehdi and Tang, Min and Golestaneh, Mehrsa and others},
  journal={IEEE Open Journal of Signal Processing},
  year={2024},
  publisher={IEEE}
}
```

Lipreading paper which used for video preprocessing *Lipreading Using Temporal Convolutional Networks*:
```
 cite:@INPROCEEDINGS{martinez2020lipreading,
   author={Martinez, Brais and Ma, Pingchuan and Petridis, Stavros and Pantic, Maja},
   booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
   title={Lipreading Using Temporal Convolutional Networks},
   year={2020},
   pages={6319-6323},
   doi={10.1109/ICASSP40776.2020.9053841}
 }
```
