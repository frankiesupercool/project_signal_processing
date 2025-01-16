#!/usr/bin/bash

# RUN WITHIN SCREEN!!

# ***** 5th DNS Challenge at ICASSP 2023*****
# Noise data which is used in both tracks
# Also download the impulse response data

# All compressed noises files are ~39 GB
# -------------------------------------------------------------
# -------------------------------------------------------------
# The directory structure of the unpacked data is:
# +-- noise_fullband

BLOB_NAMES=(
    noise_fullband/datasets_fullband.noise_fullband.audioset_000.tar.bz2
)

###############################################################

AZURE_URL="https://dnschallengepublic.blob.core.windows.net/dns5archive/V5_training_dataset"

OUTPUT_PATH="./../../../../data/datasets/"

mkdir -p $OUTPUT_PATH/denoiser_subset

for BLOB in ${BLOB_NAMES[@]}
do
    URL="$AZURE_URL/$BLOB"
    echo "Download: $BLOB"

    # DRY RUN: print HTTP response and Content-Length
    # WITHOUT downloading the files
    # curl -s -I "$URL" | head -n 2

    # Actually download the files: UNCOMMENT when ready to download
    # curl "$URL" -o "$OUTPUT_PATH/$BLOB"

    # Same as above, but using wget
    # wget "$URL" -O "$OUTPUT_PATH/$BLOB"

    # Same, + unpack files on the fly
    curl "$URL" | tar -C "$OUTPUT_PATH/denoiser_subset" -f - -x -j
done
