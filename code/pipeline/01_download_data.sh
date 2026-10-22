#!/usr/bin/env bash
set -euo pipefail
PROJECT=/data/workplace/wyx/measurement_intensity_ssl_20261015
AWS="$PROJECT/env/conda/bin/aws"
mkdir -p "$PROJECT/data/raw/cinc2019" "$PROJECT/data/raw/cinc2012" "$PROJECT/data/raw/mimic_iv_demo" "$PROJECT/logs/downloads" "$PROJECT/data/manifests"
sync_tree() {
  local s3_uri="$1"; local dest="$2"
  echo "[$(date -Is)] START $s3_uri -> $dest"
  "$AWS" s3 sync --no-sign-request --only-show-errors "$s3_uri" "$dest"
  echo "[$(date -Is)] DONE $s3_uri"
}
sync_tree 's3://physionet-open/challenge-2019/1.0.0/' "$PROJECT/data/raw/cinc2019"
sync_tree 's3://physionet-open/challenge-2012/1.0.0/' "$PROJECT/data/raw/cinc2012"
sync_tree 's3://physionet-open/mimic-iv-demo/2.2/' "$PROJECT/data/raw/mimic_iv_demo"
MANIFEST="$PROJECT/data/manifests/stage1_raw_files.tsv"
printf 'dataset\tpath\tbytes\tsha256\n' > "$MANIFEST"
for pair in cinc2019:$PROJECT/data/raw/cinc2019 cinc2012:$PROJECT/data/raw/cinc2012 mimic_iv_demo:$PROJECT/data/raw/mimic_iv_demo; do
  dataset="${pair%%:*}"; root="${pair#*:}"
  find "$root" -type f -print0 | sort -z | while IFS= read -r -d '' f; do
    bytes=$(stat -c '%s' "$f")
    sha=$(sha256sum "$f" | awk '{print $1}')
    printf '%s\t%s\t%s\t%s\n' "$dataset" "$f" "$bytes" "$sha" >> "$MANIFEST"
  done
done
printf '[%s] manifest=%s files=%s bytes=%s\n' "$(date -Is)" "$MANIFEST" "$(tail -n +2 "$MANIFEST" | wc -l)" "$(tail -n +2 "$MANIFEST" | awk -F '\t' '{s+=$3} END{print s+0}')"
