#!/usr/bin/env bash
# reset_hand_camera.sh
# Copy raw hand‑camera frames from the original Pi0 dataset
# into the corresponding episode inside the *_valid* dataset.

set -euo pipefail

LABEL="failure"           # success | failure
DATE="2025-04-17"         # top‑level date folder
TASK="data_vert_bookshelf_valid"


SRC_ROOT="~/projects/aawr/backup_data/offline_data/shelf_cabinet/${TASK}/${LABEL}/${DATE}"
DST_ROOT="~/projects/aawr/data/offline_data/shelf_cabinet/${TASK}/${LABEL}/${DATE}"

echo "SRC: ${SRC_ROOT}"
echo "DST: ${DST_ROOT}"
echo "-----------------------------------------------------------"

for SRC_EP in "${SRC_ROOT}"/*; do
  [[ -d "${SRC_EP}" ]] || continue            # skip non‑dirs
  EP_ID=$(basename "${SRC_EP}")

  SRC_HAND="${SRC_EP}/recordings/frames/hand_camera"
  DST_HAND="${DST_ROOT}/${EP_ID}/recordings/frames/hand_camera"

  # only copy if the destination episode already exists
  if [[ -d "${DST_HAND}" ]]; then
    echo "Resetting episode ${EP_ID}"
    rm -rf "${DST_HAND}"
    mkdir -p "$(dirname "${DST_HAND}")"
    cp -r "${SRC_HAND}" "${DST_HAND}"
  else
    echo "Skipping ${EP_ID} - destination episode not found."
  fi
done

echo "Done."
