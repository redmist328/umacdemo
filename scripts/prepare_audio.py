#!/usr/bin/env python3
import argparse
import csv
import json
import shutil
from pathlib import Path


def read_rows(path, count):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))[:count]


def copy_audio(source, target):
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def prepare_baselines(vocoder4, destination):
    subjective = vocoder4 / "主观测听"
    melmdct = {
        "test": vocoder4 / "codecs_output/dacstyle_code8192_01900000_melmdct_vocos_mpd_mrd_01000000_test",
        "testother": vocoder4 / "codecs_output/dacstyle_code8192_01900000_melmdct_vocos_mpd_mrd_01000000_testother",
    }
    keys = {"A": "dac", "B": "bigcodec", "C": "fmelcodec", "D": "semanticodec", "E": "focalcodec"}
    manifest = {}
    for source_split, output_split in (("test", "test-clean"), ("testother", "test-other")):
        rows = read_rows(subjective / source_split / "selection_manifest.csv", 6)
        manifest[output_split] = rows
        for row in rows:
            number = row["index"]
            original = row["original_filename"]
            root = destination / "baselines" / output_split / number
            copy_audio(subjective / source_split / "GT" / ("G_{}.wav".format(number)), root / "gt.wav")
            for letter, key in keys.items():
                copy_audio(subjective / source_split / letter / ("{}_{}.wav".format(letter, number)), root / (key + ".wav"))
            copy_audio(melmdct[source_split] / original, root / "umelcodec.wav")
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
