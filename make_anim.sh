#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'USAGE'
Usage:
  make_anim.sh -d DIR [options]

Required:
  -d, --dir DIR          Directory containing images

Common options:
  -p, --pattern GLOB     Input glob relative to DIR (default: surface_speed.*.png)
  -o, --outbase NAME     Output base name (default: auto from pattern or 'movie')
  -f, --fps N            Playback FPS (default: 30)
  -q, --quality N        Quality knob:
                           0 = pixel-perfect lossless (FFV1/MKV, no rescale)
                           >0 = compressed (CRF; higher means more compression)
  -s, --scale MODE       Output canvas for compressed path (default: native):
                         native       = keep original size, PAD to even dims (no resample)
                         480p         = 854x480
                         720p         = 1280x720
                         1080p        = 1920x1080
                         1440p|2k     = 2560x1440
                         4k           = 3840x2160
                         4k_dci       = 4096x2160
                         45|4_5k      = 4608x2592
                         5k           = 5120x2880
                         5k2k|ultrawide = 5120x2160 (21:9)
                         6k           = 5760x3240
                         6k_dci       = 6144x3160
                         7k           = 7168x4032
                         8k           = 7680x4320
                         8k_dci       = 8192x4320
                         10k21x9      = 10240x4320 (21:9 wall)
                         12k          = 11520x6480
                         12k_dci      = 12288x6480
                         16k          = 15360x8640

Advanced:
  --pixfmt FMT           Lossless pixel format (default: rgb24; use 'rgba' if inputs have alpha)
  --codec CODEC          hevc (default) | av1
  --tenbit               Use 10-bit for compressed path (yuv420p10le; hevc profile=main10)
  --preset P             Encoder preset (hevc: ultrafast..placebo; default: slow)
  --ext EXT              Force container extension for compressed (default: mp4 for hevc, mkv for av1)
  -h, --help             Show help

Examples:
  # 1) Pixel-IDENTICAL (no rescale), odd sizes OK
  make_anim.sh -d /data/frames -q 0 -f 24

  # 2) 5K HEVC, high quality
  make_anim.sh -d /data/frames -q 16 -s 5k -f 24

  # 3) 8K AV1, 10-bit
  make_anim.sh -d /data/frames -q 18 -s 8k --codec av1 --tenbit -f 24
USAGE
}

# ---------- defaults ----------
DIR=""
INPATTERN="surface_speed.*.png"
OUTBASE=""
FPS=30
QUALITY=0
SCALE="native"
LOSSLESS_PIXFMT="rgb24"
CODEC="hevc"
TENBIT=0
PRESET="slow"
EXT=""   # auto by codec: hevc->mp4, av1->mkv

# ---------- parse args ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -d|--dir)       DIR="$2"; shift 2 ;;
    -p|--pattern)   INPATTERN="$2"; shift 2 ;;
    -o|--outbase)   OUTBASE="$2"; shift 2 ;;
    -f|--fps)       FPS="$2"; shift 2 ;;
    -q|--quality)   QUALITY="$2"; shift 2 ;;
    -s|--scale)     SCALE="$2"; shift 2 ;;
    --pixfmt)       LOSSLESS_PIXFMT="$2"; shift 2 ;;
    --codec)        CODEC="$2"; shift 2 ;;
    --tenbit)       TENBIT=1; shift 1 ;;
    --preset)       PRESET="$2"; shift 2 ;;
    --ext)          EXT="$2"; shift 2 ;;
    -h|--help)      usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

[[ -z "$DIR" ]] && { echo "Error: --dir is required." >&2; usage; exit 1; }
[[ -d "$DIR" ]] || { echo "Error: Directory not found: $DIR" >&2; exit 1; }
command -v ffmpeg >/dev/null 2>&1 || { echo "Error: ffmpeg not found in PATH." >&2; exit 1; }

# Ensure frames exist
if ! compgen -G "$DIR/$INPATTERN" >/dev/null; then
  echo "No images matched: $DIR/$INPATTERN" >&2
  exit 1
fi

# Derive OUTBASE if not provided
if [[ -z "$OUTBASE" ]]; then
  b="$(basename "$INPATTERN")"
  OUTBASE="${b%%.*}"; OUTBASE="${OUTBASE%\**}"
  [[ -z "$OUTBASE" ]] && OUTBASE="movie"
fi

# ---------- helpers ----------
# Build scale/pad filter & tag for compressed path
build_vf_and_tag() {
  local mode="$1"
  case "$mode" in
    native|pad|native_pad)
      VF="pad=ceil(iw/2)*2:ceil(ih/2)*2"
      TAG="native"
      ;;
    480p)  VF="scale=854:480:flags=lanczos:force_original_aspect_ratio=decrease,pad=854:480:(ow-iw)/2:(oh-ih)/2"; TAG="480p" ;;
    720p)  VF="scale=1280:720:flags=lanczos:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2"; TAG="720p" ;;
    1080p) VF="scale=1920:1080:flags=lanczos:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2"; TAG="1080p" ;;
    1440p|2k) VF="scale=2560:1440:flags=lanczos:force_original_aspect_ratio=decrease,pad=2560:1440:(ow-iw)/2:(oh-ih)/2"; TAG="1440p" ;;
    4k)    VF="scale=3840:2160:flags=lanczos:force_original_aspect_ratio=decrease,pad=3840:2160:(ow-iw)/2:(oh-ih)/2"; TAG="4k" ;;
    4k_dci) VF="scale=4096:2160:flags=lanczos:force_original_aspect_ratio=decrease,pad=4096:2160:(ow-iw)/2:(oh-ih)/2"; TAG="4k_dci" ;;
    45|4_5k) VF="scale=4608:2592:flags=lanczos:force_original_aspect_ratio=decrease,pad=4608:2592:(ow-iw)/2:(oh-ih)/2"; TAG="4_5k" ;;
    5k)    VF="scale=5120:2880:flags=lanczos:force_original_aspect_ratio=decrease,pad=5120:2880:(ow-iw)/2:(oh-ih)/2"; TAG="5k" ;;
    5k2k|ultrawide) VF="scale=5120:2160:flags=lanczos:force_original_aspect_ratio=decrease,pad=5120:2160:(ow-iw)/2:(oh-ih)/2"; TAG="5k2k" ;;
    6k)    VF="scale=5760:3240:flags=lanczos:force_original_aspect_ratio=decrease,pad=5760:3240:(ow-iw)/2:(oh-ih)/2"; TAG="6k" ;;
    6k_dci) VF="scale=6144:3160:flags=lanczos:force_original_aspect_ratio=decrease,pad=6144:3160:(ow-iw)/2:(oh-ih)/2"; TAG="6k_dci" ;;
    7k)    VF="scale=7168:4032:flags=lanczos:force_original_aspect_ratio=decrease,pad=7168:4032:(ow-iw)/2:(oh-ih)/2"; TAG="7k" ;;
    8k)    VF="scale=7680:4320:flags=lanczos:force_original_aspect_ratio=decrease,pad=7680:4320:(ow-iw)/2:(oh-ih)/2"; TAG="8k" ;;
    8k_dci) VF="scale=8192:4320:flags=lanczos:force_original_aspect_ratio=decrease,pad=8192:4320:(ow-iw)/2:(oh-ih)/2"; TAG="8k_dci" ;;
    10k21x9) VF="scale=10240:4320:flags=lanczos:force_original_aspect_ratio=decrease,pad=10240:4320:(ow-iw)/2:(oh-ih)/2"; TAG="10k21x9" ;;
    12k)   VF="scale=11520:6480:flags=lanczos:force_original_aspect_ratio=decrease,pad=11520:6480:(ow-iw)/2:(oh-ih)/2"; TAG="12k" ;;
    12k_dci) VF="scale=12288:6480:flags=lanczos:force_original_aspect_ratio=decrease,pad=12288:6480:(ow-iw)/2:(oh-ih)/2"; TAG="12k_dci" ;;
    16k)   VF="scale=15360:8640:flags=lanczos:force_original_aspect_ratio=decrease,pad=15360:8640:(ow-iw)/2:(oh-ih)/2"; TAG="16k" ;;
    *)
      echo "Unknown --scale '$mode'." >&2
      exit 1
      ;;
  esac
}

# ---------- do work ----------
OUTDIR="$DIR"

if [[ "$QUALITY" == "0" ]]; then
  # =================== LOSSLESS PATH (pixel-identical) ===================
  # FFV1 supports odd dimensions; preserves source exactly.
  OUT="$OUTDIR/${OUTBASE}_lossless.mkv"
  # ===== LOSSLESS (FFV1) =====
  ffmpeg -hide_banner -y -stats \
    -framerate "$FPS" \
    -pattern_type glob -i "$DIR/$INPATTERN" \
    -c:v libx264rgb -preset slow -qp 0 \
    "$OUTDIR/${OUTBASE}_x264rgb_lossless.mkv"
  echo "✅ Wrote: $OUT"
  exit 0
fi

# =================== COMPRESSED PATH ===================
build_vf_and_tag "$SCALE"

# Choose codec-specific settings
VID_PIXFMT="yuv420p"
EXT_DEFAULT="mp4"
ENC_OPTS=()

if [[ "$TENBIT" -eq 1 ]]; then
  VID_PIXFMT="yuv420p10le"
fi

case "$CODEC" in
  hevc|h265)
    CODEC_LIB="libx265"
    if [[ "$TENBIT" -eq 1 ]]; then
      ENC_OPTS=(-x265-params "profile=main10:keyint=$FPS:min-keyint=$FPS:scenecut=0:aq-mode=3")
    else
      ENC_OPTS=(-x265-params "keyint=$FPS:min-keyint=$FPS:scenecut=0:aq-mode=3")
    fi
    EXT_DEFAULT="mp4"
    ;;
  av1)
    CODEC_LIB="libaom-av1"
    # CRF scale differs but we keep the same knob for simplicity.
    # -b:v 0 enables CQ mode; adjust cpu-used for speed/quality tradeoff.
    ENC_OPTS=(-cpu-used 4 -row-mt 1 -tile-columns 2 -b:v 0)
    # For 10-bit av1, yuv420p10le is fine; many players prefer MKV/MP4 (MKV safer).
    EXT_DEFAULT="mkv"
    ;;
  *)
    echo "Unsupported --codec '$CODEC' (use hevc or av1)." >&2
    exit 1
    ;;
esac

OUT_EXT="${EXT:-$EXT_DEFAULT}"
OUT="$OUTDIR/${OUTBASE}_crf${QUALITY}_${TAG}.${OUT_EXT}"

ffmpeg -hide_banner -y -stats \
  -framerate "$FPS" \
  -pattern_type glob -i "$DIR/$INPATTERN" \
  -vf "$VF" \
  -c:v "$CODEC_LIB" -preset "$PRESET" -pix_fmt "$VID_PIXFMT" \
  "${ENC_OPTS[@]}" \
  -crf "$QUALITY" \
  $( [[ "$OUT_EXT" == "mp4" ]] && echo -movflags +faststart ) \
  "$OUT"

echo "✅ Wrote: $OUT"

