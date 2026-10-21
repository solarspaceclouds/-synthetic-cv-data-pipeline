#!/usr/bin/env bash
# Download public source datasets into data/<name>/raw/ and extract into data/<name>/.
# Resumable (curl -C -). Only datasets with direct, no-login links; see docs for Cityscapes / ACDC.
#   tools/download_datasets.sh lvis mot17 mot20 visdrone ade20k openimages_val crowdhuman
set -euo pipefail
cd "$(dirname "$0")/.."

fetch() {  # fetch <dataset> <url> [<filename>]
  local d="data/$1/raw"; mkdir -p "$d"
  local f="$d/${3:-$(basename "${2%%\?*}")}"
  echo "[$1] $2"
  # a stalled connection (<10 kB/s for 60 s) is dropped and resumed; give up after 20 attempts
  for i in $(seq 20); do
    curl -fsSL --speed-limit 10000 --speed-time 60 -C - -o "$f" "$2" && return 0
    [ $? -eq 33 ] && return 0   # 33 = range not satisfiable: already complete
    echo "[$1] retry $i"; sleep 5
  done
  echo "[$1] FAILED $2" | tee -a data/_logs/failed.txt; return 0
}
extract() {  # extract <dataset>: unzip/untar every archive in raw/ into data/<dataset>/
  local d="data/$1"
  for a in "$d"/raw/*; do
    case "$a" in
      *.zip) unzip -q -n "$a" -d "$d" || echo "[$1] extract failed: $a" ;;
      *.tar.gz|*.tgz) tar -xzf "$a" -C "$d" || echo "[$1] extract failed: $a" ;;
      *.tar) tar -xf "$a" -C "$d" || echo "[$1] extract failed: $a" ;;
    esac
  done
  echo "[$1] extracted"
}

for ds in "$@"; do
  case "$ds" in
    lvis)  # annotations only; images are COCO 2017 (downloaded on demand by synthgen)
      fetch lvis https://dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip
      fetch lvis https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip ;;
    mot17) fetch mot17 https://motchallenge.net/data/MOT17.zip ;;
    mot20) fetch mot20 https://motchallenge.net/data/MOT20.zip ;;
    visdrone)
      for s in train val test-dev; do
        fetch visdrone "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-$s.zip"
      done ;;
    ade20k)  # SceneParsing challenge release (semantic) + instance masks
      fetch ade20k http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
      fetch ade20k http://sceneparsing.csail.mit.edu/data/ChallengeData2017/annotations_instance.tar ;;
    openimages_val)  # V7 validation split: 41,620 images + instance segmentation masks
      fetch openimages_val https://open-images-dataset.s3.amazonaws.com/tar/validation.tar.gz
      fetch openimages_val https://storage.googleapis.com/openimages/v5/validation-annotations-object-segmentation.csv
      fetch openimages_val https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv
      for h in 0 1 2 3 4 5 6 7 8 9 a b c d e f; do
        fetch openimages_val "https://storage.googleapis.com/openimages/v5/validation-masks/validation-masks-$h.zip"
      done ;;
    crowdhuman)  # official upload by the first author (CC BY-NC 4.0)
      for f in CrowdHuman_train01.zip CrowdHuman_train02.zip CrowdHuman_train03.zip CrowdHuman_val.zip \
               annotation_train.odgt annotation_val.odgt; do
        fetch crowdhuman "https://huggingface.co/datasets/sshao0516/CrowdHuman/resolve/main/$f"
      done ;;
    vistas)  # Mapillary Vistas: needs an account; put the signed URLs from the download page in a file
      [ -f "${VISTAS_URLS:-}" ] || { echo "set VISTAS_URLS=<file with one signed URL per line>"; exit 2; }
      while read -r u; do
        [ -n "$u" ] && fetch vistas "$u" "$(basename "${u%%\?*}")"
      done < "$VISTAS_URLS" ;;
    *) echo "unknown dataset: $ds"; exit 2 ;;
  esac
  # Open Images masks unzip into their own folder; everything else extracts at the dataset root
  if [ "$ds" = openimages_val ]; then
    mkdir -p data/openimages_val/masks
    for z in data/openimages_val/raw/validation-masks-*.zip; do unzip -q -n "$z" -d data/openimages_val/masks || echo "extract failed: $z"; done
    tar -xzf data/openimages_val/raw/validation.tar.gz -C data/openimages_val || echo "extract failed: validation.tar.gz"
    echo "[$ds] extracted"
  else
    extract "$ds"
  fi
done
