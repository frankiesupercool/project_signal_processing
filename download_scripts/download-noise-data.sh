#!/usr/bin/bash

# --- run with screen ---
# script copied and adjust from DNS Challenge to download a subset of their archive to be used as noise
# downloads ~ 5GB of noise data
# Citation of origin
# @article{dubey2024icassp,
#  title={Icassp 2023 deep noise suppression challenge},
#  author={Dubey, Harishchandra and Aazami, Ashkan and Gopal, Vishak and Naderi, Babak and Braun, Sebastian and Cutler, Ross and Ju, Alex and Zohourian, Mehdi and Tang, Min and Golestaneh, Mehrsa and others},
#  journal={IEEE Open Journal of Signal Processing},
#  year={2024},
#  publisher={IEEE}
# }

# ***** 5th DNS Challenge at ICASSP 2023*****

BLOB_NAMES=(
    noise_fullband/datasets_fullband.noise_fullband.audioset_000.tar.bz2
)

AZURE_URL="https://dnschallengepublic.blob.core.windows.net/dns5archive/V5_training_dataset"

# download to public server space, adjust for local download with matching dir in config
OUTPUT_PATH="./../../../../../data/datasets/"

mkdir -p $OUTPUT_PATH/denoiser_subset

for BLOB in ${BLOB_NAMES[@]}
do
    URL="$AZURE_URL/$BLOB"
    echo "Download: $BLOB"

    curl "$URL" | tar -C "$OUTPUT_PATH/denoiser_subset" -f - -x -j
done
