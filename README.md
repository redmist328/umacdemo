# UMelCodec Audio Demo

Static audio demo for **UMelCodec: An Ultra-Low-Bitrate and Lightweight
Mel-Domain Neural Speech Codec**.

The site contains three listening comparisons:

- Comparison with baseline codecs
- Ablation studies
- Analysis of the effect of codebook size

All UMelCodec-related samples use the MelMDCT waveform synthesis backend.
The selected utterances are reproduced from the manifests under the original
subjective-listening directory by `scripts/prepare_audio.py`.

Serve locally with:

```bash
python3 -m http.server 8000
```
