#!/usr/bin/env python3
import argparse
import csv
import json
import shutil
import subprocess
import wave
from pathlib import Path


BASELINE_SAMPLE_COUNT = 6
MIN_BASELINE_DURATION_SECONDS = 1.5


def read_rows(path, count):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))[:count]


def copy_audio(source, target):
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def copy_browser_audio(source, target, convert_to_pcm=False):
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not convert_to_pcm:
        shutil.copy2(source, target)
        return
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-c:a",
            "pcm_s16le",
            str(target),
        ],
        check=True,
    )


def read_utmos(path):
    with path.open(newline="") as handle:
        return {
            row["relpath"]: float(row["utmos"])
            for row in csv.DictReader(handle)
        }


def wav_duration_seconds(path):
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / float(handle.getframerate())


def prepare_baselines(vocoder4, destination):
    output = vocoder4 / "codecs_output"
    datasets = {
        "test": vocoder4.parent / "datasets/LibriTTS-16k/test",
        "testother": vocoder4.parent / "datasets/LibriTTS-16k/LibriTTS/test16000",
    }
    systems = {
        "dac": {
            "test": output / "dac_code8192_00500000_test",
            "testother": output / "dac_code8192_00500000_testother",
        },
        "bigcodec": {
            "test": output / "bigcodec_code8192_00400000_test",
            "testother": output / "bigcodec_code8192_00400000_testother",
        },
        "fmelcodec": {
            "test": output / "fmelcodec_stage2_cfm_code8192_00200000_4steps_hifigan_01000000_test",
            "testother": output / "fmelcodec_stage2_cfm_code8192_00200000_4steps_hifigan_01000000_testother",
        },
        "semanticodec": {
            "test": output / "SemantiCodec_25tps_vocab8192_033kbps/test",
            "testother": output / "SemantiCodec_25tps_vocab8192_033kbps/testother",
        },
        "focalcodec": {
            "test": output / "output_official_focalcodec/test",
            "testother": output / "output_official_focalcodec/testother",
        },
        "umelcodec": {
            "test": output / "dacstyle_code8192_01900000_melmdct_vocos_mpd_mrd_01000000_test",
            "testother": output / "dacstyle_code8192_01900000_melmdct_vocos_mpd_mrd_01000000_testother",
        },
    }
    manifest = {}
    for source_split, output_split in (("test", "test-clean"), ("testother", "test-other")):
        umel_scores = read_utmos(systems["umelcodec"][source_split] / "utmos_scores.csv")
        focal_scores = read_utmos(systems["focalcodec"][source_split] / "utmos_scores.csv")
        candidates = []
        for original in sorted(umel_scores.keys() & focal_scores.keys()):
            gt = datasets[source_split] / original
            if not gt.is_file():
                continue
            duration = wav_duration_seconds(gt)
            if duration < MIN_BASELINE_DURATION_SECONDS:
                continue
            candidates.append(
                {
                    "original_filename": original,
                    "duration_seconds": duration,
                    "umelcodec_utmos": umel_scores[original],
                    "focalcodec_utmos": focal_scores[original],
                    "utmos_difference": umel_scores[original] - focal_scores[original],
                }
            )
        candidates.sort(
            key=lambda row: (-row["utmos_difference"], row["original_filename"])
        )
        rows = candidates[:BASELINE_SAMPLE_COUNT]
        if len(rows) != BASELINE_SAMPLE_COUNT:
            raise RuntimeError("not enough baseline candidates for {}".format(source_split))
        for index, row in enumerate(rows, 1):
            row["index"] = "{:02d}".format(index)
        manifest[output_split] = rows
        for row in rows:
            number = row["index"]
            original = row["original_filename"]
            root = destination / "baselines" / output_split / number
            copy_browser_audio(datasets[source_split] / original, root / "gt.wav")
            for key, split_sources in systems.items():
                copy_browser_audio(
                    split_sources[source_split] / original,
                    root / (key + ".wav"),
                    convert_to_pcm=(key == "focalcodec"),
                )
    return manifest


def prepare_ablations(vocoder4, destination):
    subjective = vocoder4 / "主观测听/test"
    output = vocoder4 / "codecs_output"
    sources = {
        "umelcodec": output / "dacstyle_code8192_01900000_melmdct_vocos_mpd_mrd_01000000_test",
        "no-msmd": output / "ablation_no_discriminator_code8192_01100000_melmdct_vocos_mpd_mrd_01000000_test",
        "no-af": output / "ablation_no_af_resampling_code8192_01100000_melmdct_vocos_mpd_mrd_01000000_test",
        "no-vq-factorization": output / "ablation_direct32_vq_code8192_01000000_melmdct_vocos_mpd_mrd_01000000_test",
    }
    rows = read_rows(subjective / "ablation_selection_manifest.csv", 6)
    for row in rows:
        number = row["index"]
        original = row["original_filename"]
        root = destination / "ablations/test-clean" / number
        copy_audio(subjective / "J" / ("J_{}.wav".format(number)), root / "gt.wav")
        for key, source in sources.items():
            copy_audio(source / original, root / (key + ".wav"))
    return rows


def prepare_codebooks(vocoder4, destination):
    subjective = vocoder4 / "主观测听/test/three"
    output = vocoder4 / "codecs_output"
    checkpoints = {"1024": "01300000", "2048": "01000000", "4096": "02200000", "8192": "01900000", "16384": "01200000"}
    rows = read_rows(subjective / "selection_manifest.csv", 6)
    for row in rows:
        number = row["index"]
        original = row["original_filename"]
        root = destination / "codebooks/test-clean" / number
        copy_audio(subjective / "F" / ("F_{}.wav".format(number)), root / "gt.wav")
        for size, checkpoint in checkpoints.items():
            source = output / ("dacstyle_code{}_{}_melmdct_vocos_mpd_mrd_01000000_test".format(size, checkpoint))
            copy_audio(source / original, root / (size + ".wav"))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocoder4-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("audio"))
    args = parser.parse_args()
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {
        "baselines": prepare_baselines(args.vocoder4_root, destination),
        "ablations": prepare_ablations(args.vocoder4_root, destination),
        "codebooks": prepare_codebooks(args.vocoder4_root, destination),
        "note": "All UMelCodec, ablation, and codebook samples use the MelMDCT backend.",
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Prepared audio under", destination)


if __name__ == "__main__":
    main()
