#!/usr/bin/env python3
"""Turn generated tracks into one long, seamless meditation video/audio.

Needs only ffmpeg + python (fine on an old CPU-only PC).

Steps
 1. crossfade your chosen tracks into one bed
 2. make that bed loop seamlessly (end fades into start)
 3. normalise loudness (-16 LUFS, YouTube-friendly for calm content)
 4. optionally layer a voice/chant track (looped, quieter)
 5. repeat to the target length and, if you give --bg, mux with a looping
    background video or still image

Examples
  python scripts/make_loop.py --music out/bowls_drone/*.flac --duration 3h \
      --bg footage/forest.mp4 --out final/bowls_3h.mp4
  python scripts/make_loop.py --music out/deep_focus_pad/a.flac --voice out/om_vocal_drone/b.flac \
      --voice-db -9 --duration 2h --bg footage/lake.jpg --out final/om_2h.mp4
  python scripts/make_loop.py --music a.flac --duration 1h --out final/bed_1h.m4a   # audio only
"""
import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def run(cmd):
    print("+", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True)


def probe_len(path) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)], text=True)
    return float(out.strip())


def parse_duration(s: str) -> float:
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([hms]?)\s*", s)
    if not m:
        sys.exit(f"bad --duration {s!r}; use e.g. 3h, 90m, 30s")
    v, u = float(m.group(1)), m.group(2) or "s"
    return v * {"h": 3600, "m": 60, "s": 1}[u]


def crossfade_chain(files, xf, out):
    """Join files with equal-power crossfades into one wav."""
    if len(files) == 1:
        run(["ffmpeg", "-y", "-v", "error", "-i", files[0], "-ar", "48000", "-ac", "2", out])
        return
    cmd = ["ffmpeg", "-y", "-v", "error"]
    for f in files:
        cmd += ["-i", f]
    parts, prev = [], "[0:a]"
    for i in range(1, len(files)):
        lab = f"[x{i}]"
        parts.append(f"{prev}[{i}:a]acrossfade=d={xf}:c1=qsin:c2=qsin{lab}")
        prev = lab
    cmd += ["-filter_complex", ";".join(parts), "-map", prev, "-ar", "48000", "-ac", "2", out]
    run(cmd)


def make_seamless(src, xf, out):
    """Fade the last xf seconds into the first xf seconds so loop point is inaudible."""
    L = probe_len(src)
    if L < xf * 3:
        sys.exit(f"track too short ({L:.1f}s) for a {xf}s loop crossfade; lower --crossfade")
    # the same file is opened three times; splitting one input with asplit stalls ffmpeg
    fg = (
        f"[0:a]atrim=0:{xf},asetpts=PTS-STARTPTS,afade=t=in:d={xf}:curve=qsin[head];"
        f"[1:a]atrim={L - xf}:{L},asetpts=PTS-STARTPTS,afade=t=out:d={xf}:curve=qsin[tail];"
        f"[head][tail]amix=inputs=2:duration=longest:normalize=0[mix];"
        f"[2:a]atrim={xf}:{L - xf},asetpts=PTS-STARTPTS[body];"
        f"[mix][body]concat=n=2:v=0:a=1[o]"
    )
    run(["ffmpeg", "-y", "-v", "error", "-i", src, "-i", src, "-i", src,
         "-filter_complex", fg, "-map", "[o]", out])


def loudnorm(src, out, lufs=-16):
    run(["ffmpeg", "-y", "-v", "error", "-i", src,
         "-af", f"loudnorm=I={lufs}:TP=-1.5:LRA=7", "-ar", "48000", out])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--music", nargs="+", required=True, help="audio files for the bed (crossfaded in order)")
    ap.add_argument("--voice", help="optional chant/voice track to layer on top (looped)")
    ap.add_argument("--voice-db", type=float, default=-8.0, help="voice level relative to bed (default -8 dB)")
    ap.add_argument("--duration", required=True, help="final length: 3h, 90m, 30s")
    ap.add_argument("--crossfade", type=float, default=12.0, help="crossfade seconds (default 12)")
    ap.add_argument("--bg", help="background video (.mp4/.mov/.webm) or image (.jpg/.png)")
    ap.add_argument("--reencode", action="store_true",
                    help="re-encode a video bg instead of stream-copy (use if copy gives glitches)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found")
    total = parse_duration(args.duration)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        bed = td / "bed.wav"
        crossfade_chain(args.music, args.crossfade, bed)
        loop = td / "bed_loop.wav"
        make_seamless(bed, args.crossfade, loop)
        norm = td / "bed_norm.wav"
        loudnorm(loop, norm)

        voice_norm = None
        if args.voice:
            v_loop = td / "voice_loop.wav"
            # short xfade for the voice loop point
            make_seamless(args.voice, min(args.crossfade, 6), v_loop)
            voice_norm = td / "voice_norm.wav"
            loudnorm(v_loop, voice_norm, lufs=-16)

        cmd = ["ffmpeg", "-y", "-v", "error", "-stats"]
        cmd += ["-stream_loop", "-1", "-i", norm]
        idx = 1
        if voice_norm:
            cmd += ["-stream_loop", "-1", "-i", voice_norm]
            idx = 2
        if args.bg:
            bg = Path(args.bg)
            if bg.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                cmd += ["-loop", "1", "-framerate", "2", "-i", bg]
                vcodec = ["-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p",
                          "-r", "2", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2"]
            else:
                cmd += ["-stream_loop", "-1", "-i", bg]
                vcodec = (["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p"]
                          if args.reencode else ["-c:v", "copy"])
        if voice_norm:
            cmd += ["-filter_complex",
                    f"[1:a]volume={args.voice_db}dB[v];[0:a][v]amix=inputs=2:duration=first:normalize=0[a]",
                    "-map", "[a]"]
        else:
            cmd += ["-map", "0:a"]
        if args.bg:
            cmd += ["-map", f"{idx}:v"] + vcodec
        else:
            cmd += ["-vn"]
        acodec = ["-c:a", "aac", "-b:a", "192k"] if out.suffix.lower() in {".mp4", ".m4a", ".mov"} else []
        cmd += acodec + ["-t", f"{total}", out]
        run(cmd)
    print(f"\nDone: {out}  ({total / 3600:.2f} h)")


if __name__ == "__main__":
    main()
