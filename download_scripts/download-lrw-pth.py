import gdown

# script to download lrw video encoding model (Lipreading using Temporal Convolutional Networks)
#
# or download directly from https://drive.google.com/file/d/1TGFG0dW5M3rBErgU8i0N7M1ys9YMIvgm/view
# save to destination path (video_encoding)
#
# Citation of paper (of file origin)
# cite:@INPROCEEDINGS{martinez2020lipreading,
#   author={Martinez, Brais and Ma, Pingchuan and Petridis, Stavros and Pantic, Maja},
#   booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
#   title={Lipreading Using Temporal Convolutional Networks},
#   year={2020},
#   pages={6319-6323},
#   doi={10.1109/ICASSP40776.2020.9053841}
# }

destination_path = '../video_encoding/lrw_resnet18_dctcn_video_boundary.pth'
url = f'https://drive.google.com/uc?export=download&id=1TGFG0dW5M3rBErgU8i0N7M1ys9YMIvgm'
gdown.download(url, destination_path, quiet=False)

print(f"LRW model downloaded to {destination_path}")