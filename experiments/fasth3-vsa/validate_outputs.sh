#!/usr/bin/env bash
# ffprobe + full decode validation, per the R9700 v1 benchmark standard.
# Checks geometry, codecs, streams, clean decode, non-silent audio, sha256.
set -uo pipefail
OUT="${1:-/ai/lab/experiments/fasth3-vsa/outputs/fasth3}"
fail=0

for f in "$OUT"/*.mp4; do
  [ -e "$f" ] || continue
  echo "======================================================================"
  echo "FILE: $(basename "$f")   ($(du -h "$f" | cut -f1))"

  ffprobe -v error -select_streams v:0 -show_entries \
    stream=width,height,r_frame_rate,nb_frames,codec_name,pix_fmt \
    -of default=nw=1 "$f"
  dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$f")
  echo "duration=$dur"

  if ffprobe -v error -select_streams a:0 -show_entries \
      stream=codec_name,sample_rate,channels -of default=nw=1 "$f" 2>/dev/null | grep -q codec_name; then
    ffprobe -v error -select_streams a:0 -show_entries \
      stream=codec_name,sample_rate,channels,duration -of default=nw=1 "$f"
    # non-silent check: mean_volume of -91 dB is digital silence
    mv=$(ffmpeg -v info -i "$f" -af volumedetect -f null - 2>&1 | grep -o "mean_volume: [-0-9.]* dB" | tail -1)
    echo "audio_${mv:-mean_volume: UNKNOWN}"
    case "$mv" in *"-91"*|*"-inf"*|"") echo "  !! AUDIO MAY BE SILENT"; fail=1;; esac
  else
    echo "  !! NO AUDIO STREAM"; fail=1
  fi

  # full decode to null: any error here means corruption/truncation
  err=$(ffmpeg -v error -i "$f" -f null - 2>&1 | head -5)
  if [ -n "$err" ]; then
    echo "  !! DECODE ERRORS:"; echo "$err" | sed 's/^/     /'; fail=1
  else
    echo "decode=clean"
  fi

  echo "sha256=$(sha256sum "$f" | cut -d' ' -f1)"
done

echo "======================================================================"
[ "$fail" -eq 0 ] && echo "VALIDATION: ALL PASS" || echo "VALIDATION: FAILURES PRESENT"
exit $fail
