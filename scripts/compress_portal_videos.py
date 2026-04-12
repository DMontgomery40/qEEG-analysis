#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VIDEO_EXTENSIONS = {".mp4"}
DEFAULT_VIDEO_BITRATE_KBPS = 2200
DEFAULT_AUDIO_BITRATE_KBPS = 128
DEFAULT_MAX_AVERAGE_BITRATE_MBPS = 2.8
DEFAULT_MIN_SIZE_MB = 0.0


@dataclass
class VideoStats:
    path: str
    size_bytes: int
    duration_seconds: float | None
    average_bitrate_mbps: float | None
    has_audio: bool


def _run_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=True,
        text=True,
        capture_output=True,
    )


def _average_bitrate_mbps(size_bytes: int, duration_seconds: float | None) -> float | None:
    if duration_seconds is None or duration_seconds <= 0:
        return None
    return round((float(size_bytes) * 8.0) / float(duration_seconds) / 1_000_000.0, 3)


def _probe_video(path: Path) -> VideoStats:
    completed = _run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
    )
    payload = json.loads(completed.stdout or "{}")
    format_info = payload.get("format") if isinstance(payload.get("format"), dict) else {}
    streams = payload.get("streams") if isinstance(payload.get("streams"), list) else []

    try:
        size_bytes = int(format_info.get("size") or path.stat().st_size)
    except (TypeError, ValueError):
        size_bytes = path.stat().st_size

    raw_duration = format_info.get("duration")
    try:
        duration_seconds = float(raw_duration) if raw_duration not in (None, "", "N/A") else None
    except (TypeError, ValueError):
        duration_seconds = None

    has_audio = any(
        isinstance(stream, dict) and str(stream.get("codec_type") or "").strip().lower() == "audio"
        for stream in streams
    )
    return VideoStats(
        path=str(path),
        size_bytes=size_bytes,
        duration_seconds=duration_seconds,
        average_bitrate_mbps=_average_bitrate_mbps(size_bytes, duration_seconds),
        has_audio=has_audio,
    )


def _relative_display(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _unique_backup_path(path: Path) -> Path:
    if not path.exists():
        return path
    counter = 2
    while True:
        candidate = path.with_name(f"{path.stem}__{counter}{path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def _should_compress(stats: VideoStats, args: argparse.Namespace) -> bool:
    if args.force:
        return True
    if stats.size_bytes <= 0:
        return False
    size_mb = stats.size_bytes / (1024.0 * 1024.0)
    bitrate = stats.average_bitrate_mbps
    return size_mb >= args.min_size_mb and (bitrate is None or bitrate > args.max_average_bitrate_mbps)


def _compress_video(path: Path, stats: VideoStats, args: argparse.Namespace) -> tuple[Path, VideoStats]:
    suffix = path.suffix or ".mp4"
    with tempfile.NamedTemporaryFile(
        prefix=f"{path.stem}.compressing.",
        suffix=suffix,
        dir=path.parent,
        delete=False,
    ) as tmp_file:
        temp_output = Path(tmp_file.name)

    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-c:v",
            "libx264",
            "-preset",
            args.preset,
            "-b:v",
            f"{args.video_bitrate_kbps}k",
            "-maxrate",
            f"{int(round(args.video_bitrate_kbps * 1.25))}k",
            "-bufsize",
            f"{int(round(args.video_bitrate_kbps * 2.5))}k",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ]
        if stats.has_audio:
            cmd.extend(
                [
                    "-c:a",
                    "aac",
                    "-b:a",
                    f"{args.audio_bitrate_kbps}k",
                    "-ar",
                    "48000",
                    "-ac",
                    "2",
                ]
            )
        else:
            cmd.extend(["-an"])
        cmd.append(str(temp_output))
        _run_command(cmd)
        return temp_output, _probe_video(temp_output)
    except Exception:
        temp_output.unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compress portal patient videos in place, with optional backups and a JSON report.",
    )
    parser.add_argument(
        "root",
        nargs="?",
        default="data/portal_patients",
        help="Root folder containing published portal patient videos.",
    )
    parser.add_argument(
        "--backup-dir",
        default="data/portal_patients_precompress_backups",
        help="Where to copy originals before replacement. Set to '' to disable backups.",
    )
    parser.add_argument(
        "--report-json",
        default="",
        help="Optional JSON report path.",
    )
    parser.add_argument(
        "--video-bitrate-kbps",
        type=int,
        default=DEFAULT_VIDEO_BITRATE_KBPS,
        help="Target video bitrate for compressed outputs.",
    )
    parser.add_argument(
        "--audio-bitrate-kbps",
        type=int,
        default=DEFAULT_AUDIO_BITRATE_KBPS,
        help="Target audio bitrate for compressed outputs.",
    )
    parser.add_argument(
        "--max-average-bitrate-mbps",
        type=float,
        default=DEFAULT_MAX_AVERAGE_BITRATE_MBPS,
        help="Only compress files above this average bitrate unless --force is set.",
    )
    parser.add_argument(
        "--min-size-mb",
        type=float,
        default=DEFAULT_MIN_SIZE_MB,
        help="Only compress files at or above this size unless --force is set.",
    )
    parser.add_argument(
        "--preset",
        default="medium",
        help="ffmpeg x264 preset to use for the compression pass.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Compress every matching video even if it is already within the target bitrate budget.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect and report without writing changes.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Missing root folder: {root}")
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise SystemExit("ffmpeg and ffprobe must both be available on PATH.")

    backup_dir = Path(args.backup_dir).expanduser().resolve() if str(args.backup_dir).strip() else None
    if backup_dir is not None and not args.dry_run:
        backup_dir.mkdir(parents=True, exist_ok=True)

    report: list[dict[str, Any]] = []
    total_original_bytes = 0
    total_final_bytes = 0
    compressed_count = 0
    skipped_count = 0

    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    print(f"Scanning {len(files)} video(s) under {root}")

    for path in files:
        original = _probe_video(path)
        total_original_bytes += original.size_bytes
        item: dict[str, Any] = {
            "path": _relative_display(root, path),
            "status": "skipped",
            "original": asdict(original),
            "final": asdict(original),
            "backup_path": None,
            "error": None,
        }

        if not _should_compress(original, args):
            skipped_count += 1
            total_final_bytes += original.size_bytes
            item["status"] = "within_budget"
            report.append(item)
            print(
                f"skip {item['path']}: "
                f"{original.size_bytes / (1024.0 * 1024.0):.1f} MB at "
                f"{original.average_bitrate_mbps or 0:.2f} Mbps"
            )
            continue

        print(
            f"compress {item['path']}: "
            f"{original.size_bytes / (1024.0 * 1024.0):.1f} MB at "
            f"{original.average_bitrate_mbps or 0:.2f} Mbps"
        )

        if args.dry_run:
            compressed_count += 1
            total_final_bytes += original.size_bytes
            item["status"] = "would_compress"
            report.append(item)
            continue

        try:
            temp_output, final_stats = _compress_video(path, original, args)
            item["final"] = asdict(final_stats)

            if final_stats.size_bytes >= original.size_bytes:
                item["status"] = "kept_original_not_smaller"
                temp_output.unlink(missing_ok=True)
                skipped_count += 1
                total_final_bytes += original.size_bytes
                report.append(item)
                print(f"keep {item['path']}: compressed output was not smaller")
                continue

            if backup_dir is not None:
                backup_path = _unique_backup_path(backup_dir / path.relative_to(root))
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), str(backup_path))
                item["backup_path"] = str(backup_path)
            else:
                path.unlink()

            shutil.move(str(temp_output), str(path))
            final_replaced = _probe_video(path)
            item["final"] = asdict(final_replaced)
            item["status"] = "compressed"
            compressed_count += 1
            total_final_bytes += final_replaced.size_bytes
            saved_mb = (original.size_bytes - final_replaced.size_bytes) / (1024.0 * 1024.0)
            print(
                f"done {item['path']}: "
                f"{original.size_bytes / (1024.0 * 1024.0):.1f} MB -> "
                f"{final_replaced.size_bytes / (1024.0 * 1024.0):.1f} MB "
                f"(saved {saved_mb:.1f} MB)"
            )
        except Exception as exc:
            item["status"] = "error"
            item["error"] = str(exc)
            skipped_count += 1
            total_final_bytes += original.size_bytes
            print(f"error {item['path']}: {exc}")
        report.append(item)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "compressed_count": compressed_count,
        "skipped_count": skipped_count,
        "total_original_bytes": total_original_bytes,
        "total_final_bytes": total_final_bytes,
        "total_saved_bytes": total_original_bytes - total_final_bytes,
        "settings": {
            "video_bitrate_kbps": args.video_bitrate_kbps,
            "audio_bitrate_kbps": args.audio_bitrate_kbps,
            "max_average_bitrate_mbps": args.max_average_bitrate_mbps,
            "min_size_mb": args.min_size_mb,
            "preset": args.preset,
            "force": args.force,
            "dry_run": args.dry_run,
            "backup_dir": str(backup_dir) if backup_dir is not None else None,
        },
        "files": report,
    }

    if args.report_json:
        report_path = Path(args.report_json).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"wrote report: {report_path}")

    saved_mb = (total_original_bytes - total_final_bytes) / (1024.0 * 1024.0)
    print(
        f"summary: compressed={compressed_count} skipped={skipped_count} "
        f"saved={saved_mb:.1f} MB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
