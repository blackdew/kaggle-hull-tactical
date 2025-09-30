#!/usr/bin/env bash
set -euo pipefail

COMPETITION_SLUG_DEFAULT="hull-tactical-market-prediction"
COMPETITION_SLUG="${KAGGLE_COMPETITION:-$COMPETITION_SLUG_DEFAULT}"
MESSAGE=""
FILE=""

usage() {
  cat <<USAGE
Usage:
  bash scripts/submit.sh [-f PATH] [-m MESSAGE] [-c SLUG]

Options:
  -f, --file PATH       제출할 파일(.csv/.parquet). 미지정 시 submissions/에서 최신 파일 자동 선택
  -m, --message MSG     제출 메시지(미지정 시 UTC 타임스탬프 기본값)
  -c, --competition SLUG  대회 슬러그(기본: ${COMPETITION_SLUG_DEFAULT})
  -h, --help            도움말

환경 변수:
  KAGGLE_COMPETITION    기본 슬러그 오버라이드(옵션)
  KAGGLE_CONFIG_DIR     Kaggle API 키 디렉터리(옵션)

예시:
  bash scripts/submit.sh -f submissions/submission.parquet -m "first try"
  bash scripts/submit.sh --message "auto"  # submissions/의 최신 파일 자동 선택
USAGE
}

# parse args (long + short)
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -f|--file)
      FILE="$2"; shift 2;;
    -m|--message)
      MESSAGE="$2"; shift 2;;
    -c|--competition)
      COMPETITION_SLUG="$2"; shift 2;;
    -h|--help)
      usage; exit 0;;
    --) shift; break;;
    -*) echo "Unknown option: $1" >&2; usage; exit 2;;
    *) ARGS+=("$1"); shift;;
  esac
done

# Ensure kaggle CLI exists
if ! command -v kaggle >/dev/null 2>&1; then
  echo "Error: 'kaggle' CLI not found. Install via 'pip install kaggle' or ensure PATH." >&2
  exit 1
fi

# Configure API key: prefer ~/.kaggle, fallback to project ./.kaggle
if [[ ! -f "$HOME/.kaggle/kaggle.json" && -f ".kaggle/kaggle.json" ]]; then
  export KAGGLE_CONFIG_DIR="$(pwd)/.kaggle"
fi

# Pick latest file from submissions/ if not provided
if [[ -z "${FILE}" ]]; then
  if [[ -d submissions ]]; then
    # shellcheck disable=SC2012
    FILE=$(ls -t submissions/*.{csv,parquet} 2>/dev/null | head -n1 || true)
  fi
fi

if [[ -z "${FILE}" ]]; then
  echo "Error: 제출 파일을 찾을 수 없습니다. -f 옵션으로 지정하거나 submissions/에 파일을 두세요." >&2
  exit 1
fi

if [[ ! -f "$FILE" ]]; then
  echo "Error: 파일이 존재하지 않습니다: $FILE" >&2
  exit 1
fi

ext="${FILE##*.}"
case "$ext" in
  csv|parquet) : ;; 
  *) echo "Warning: 확장자가 csv/parquet가 아닙니다: .$ext (계속 진행)";;
esac

if [[ -z "$MESSAGE" ]]; then
  MESSAGE="auto submit $(date -u +"%F %T UTC")"
fi

echo "Competition : $COMPETITION_SLUG"
echo "File        : $FILE"
echo "Message     : $MESSAGE"
echo "Submitting..."

kaggle competitions submit -c "$COMPETITION_SLUG" -f "$FILE" -m "$MESSAGE"

echo "Done. Check Kaggle submissions page for status."

