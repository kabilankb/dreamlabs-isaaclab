"""Convert a LeRobot v3.0 dataset to v2.1, the format Dream's uploader takes.

    ~/dream-env/bin/python -m dreamlabs_isaaclab.convert SRC_V3 DST_V21 [--episodes N]

v3 packs many episodes into one parquet / mp4 file; v2.1 wants one parquet and one mp4 per episode:

    meta/{info.json, episodes.jsonl, tasks.jsonl}
    data/chunk-NNN/episode_NNNNNN.parquet
    videos/chunk-NNN/<video key>/episode_NNNNNN.mp4

SRC_V3 is only read. Each episode's clip is cut out of the packed mp4 by timestamp and re-encoded as H.264
(yuv420p), so the cut is frame accurate; the frame count of every clip is checked against the episode length.
Needs pyarrow (installed with dreamlabs), PyAV (installed with dreamlabs) and an ffmpeg binary on PATH.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import av
import pyarrow as pa
import pyarrow.parquet as pq

CHUNKS_SIZE = 1000


def frames_in(path: Path) -> int:
    with av.open(str(path)) as container:
        return sum(1 for _ in container.decode(video=0))


def cut(src: Path, dst: Path, start: float, end: float, fps: int, crf: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-ss", f"{start:.6f}", "-to", f"{end:.6f}",
         "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf), "-g", str(fps), "-r", str(fps), str(dst)],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    parser.add_argument("--episodes", type=int, default=None, help="convert only the first N episodes (a trial run)")
    parser.add_argument("--crf", type=int, default=18)
    args = parser.parse_args()
    src, dst = args.src.expanduser(), args.dst.expanduser()
    if dst.exists() and any(dst.iterdir()):
        sys.exit(f"{dst} exists and is not empty; refusing to write into it")

    info = json.loads((src / "meta/info.json").read_text())
    if not info["codebase_version"].startswith("v3"):
        sys.exit(f"{src} is {info['codebase_version']}, not v3.x")
    fps = int(info["fps"])
    video_keys = [k for k, f in info["features"].items() if f["dtype"] == "video"]
    episodes = []
    for part in sorted((src / "meta/episodes").glob("chunk-*/file-*.parquet")):
        episodes += pq.read_table(part).to_pylist()
    episodes.sort(key=lambda e: e["episode_index"])
    if args.episodes:
        episodes = episodes[: args.episodes]
    tasks = {row["task"]: row["task_index"] for row in pq.read_table(src / "meta/tasks.parquet").to_pylist()}

    # The v3 data files, each read once and sliced per episode.
    tables: dict[tuple[int, int], pa.Table] = {}
    (dst / "meta").mkdir(parents=True)
    total_frames = 0
    with (dst / "meta/episodes.jsonl").open("w") as episodes_out:
        for ep in episodes:
            index, length = ep["episode_index"], ep["length"]
            chunk = index // CHUNKS_SIZE
            key = (ep["data/chunk_index"], ep["data/file_index"])
            if key not in tables:
                tables[key] = pq.read_table(src / info["data_path"].format(chunk_index=key[0], file_index=key[1]))
            table = tables[key]
            frame_index = table["index"].to_numpy()
            first = int((frame_index == ep["dataset_from_index"]).nonzero()[0][0])
            rows = table.slice(first, length)
            assert rows["episode_index"].to_pylist() == [index] * length, f"episode {index}: rows are not its own"
            out = dst / f"data/chunk-{chunk:03d}/episode_{index:06d}.parquet"
            out.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(rows, out)

            for video_key in video_keys:
                prefix = f"videos/{video_key}"
                clip = dst / f"videos/chunk-{chunk:03d}/{video_key}/episode_{index:06d}.mp4"
                packed = src / info["video_path"].format(
                    video_key=video_key,
                    chunk_index=ep[f"{prefix}/chunk_index"],
                    file_index=ep[f"{prefix}/file_index"],
                )
                start = ep[f"{prefix}/from_timestamp"]
                cut(packed, clip, start, start + length / fps, fps, args.crf)
                got = frames_in(clip)
                if got != length:
                    sys.exit(f"episode {index} {video_key}: clip has {got} frames, the episode {length}")

            episodes_out.write(json.dumps({"episode_index": index, "tasks": ep["tasks"], "length": length}) + "\n")
            total_frames += length
            print(f"episode {index:3d}: {length} frames", flush=True)

    with (dst / "meta/tasks.jsonl").open("w") as tasks_out:
        for task, task_index in sorted(tasks.items(), key=lambda kv: kv[1]):
            tasks_out.write(json.dumps({"task_index": task_index, "task": task}) + "\n")

    v21 = {k: v for k, v in info.items() if k not in {"data_files_size_in_mb", "video_files_size_in_mb"}}
    v21.update(
        codebase_version="v2.1",
        total_episodes=len(episodes),
        total_frames=total_frames,
        total_videos=len(episodes) * len(video_keys),
        total_chunks=(len(episodes) + CHUNKS_SIZE - 1) // CHUNKS_SIZE,
        chunks_size=CHUNKS_SIZE,
        splits={"train": f"0:{len(episodes)}"},
        data_path="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        video_path="videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    )
    for video_key in video_keys:
        v21["features"][video_key]["info"]["video.crf"] = args.crf
    (dst / "meta/info.json").write_text(json.dumps(v21, indent=4))
    print(f"wrote {len(episodes)} episodes, {total_frames} frames to {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
