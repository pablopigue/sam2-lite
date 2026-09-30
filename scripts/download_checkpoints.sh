#!/usr/bin/env bash
# Download the SAM 2.1 Hiera-tiny teacher checkpoint into checkpoints/.
# URL taken from facebookresearch/sam2 checkpoints/download_ckpts.sh.
# Idempotent: skips the download if the file already exists, and always verifies the checksum.
set -euo pipefail

URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
DEST_DIR="checkpoints"
DEST="${DEST_DIR}/sam2.1_hiera_tiny.pt"
# Meta does not publish checksums; this one was computed on the first download.
SHA256="7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69"

mkdir -p "${DEST_DIR}"

if [[ -f "${DEST}" ]]; then
    echo "Already present: ${DEST}"
else
    echo "Downloading ${URL}"
    # Download to a temporary name so an interrupted download never looks complete.
    wget -c -O "${DEST}.part" "${URL}"
    mv "${DEST}.part" "${DEST}"
fi

actual="$(sha256sum "${DEST}" | cut -d' ' -f1)"
if [[ -z "${SHA256}" ]]; then
    echo "No checksum pinned yet. SHA-256 of ${DEST}: ${actual}"
elif [[ "${actual}" != "${SHA256}" ]]; then
    echo "Checksum mismatch for ${DEST}: expected ${SHA256}, got ${actual}" >&2
    exit 1
else
    echo "Checksum OK: ${DEST}"
fi
