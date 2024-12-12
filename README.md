# project_signal_processing

## Setup

To setup this environment follow the following steps:

- Load your data into the `data/pretrain`, `data/test`, `data/test` folders
- Add the `lrw_resnet18_dctcn_video_boundary.pth` file to the video_encoding folder
- start the environment by running

`conda env create -f environment.yml`

- then start the necessary file


## Development

- When developing and adding a new dependency like torch run the following command to ensure that the dependency is 
in the `environment.yml`

`conda env export > environment.yml`

## Questions

- Length of the seminar paper presentation?
  - Answer: approx. 40min
- Presentation length of project?
  - Answer: No presentation - just lab report
- Length of lab report?
  - Answer: content conference paper but style basic report

## ssh sppc7 database access

- `ssh 0name@sppc7.informatik.uni-hamburg.de`

## Notes 

- preprocessing in dataset init function
- get item would be called everytime (each access, each epoch etc.)


- init should include preprocessing (Pytorch Lightning prepare_data for static one time operation)
- encoding both
- both modalities to handle in one dataset - else the audio would be loaded twice as the labels are coming from the audio

- cropping, noise, encoding not in dataset -> memory issue


- what we do: cropping separate/before from (noise? and) encoding, encoding in dataset:
- 1. Crop and create separate dataset (LRS3_cropped)
- 2. Load LRS3_cropped and apply everytime noise and different speakers and encoding
- 3. train

### Noise
- either speech from another speaker in the speaker separation experiments
- or a noise audio clip, simulating background noise for the speech enhancement experiments

## Datasets

### Noise: DNS dataset 
- https://www.microsoft.com/en-us/research/academic-program/deep-noise-suppression-challenge-interspeech-2021/links/
- https://github.com/microsoft/DNS-Challenge/tree/master
- https://www.kaggle.com/datasets/muhmagdy/dns-2021-noise

### LSR3



### further notes
- landmarks needed? https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python?hl=de
- SNR - signal to noise https://resources.pcb.cadence.com/blog/2020-what-is-signal-to-noise-ratio-and-how-to-calculate-it