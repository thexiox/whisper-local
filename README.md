# whisper-local

**Local speech-to-text with hotkey. Record, transcribe, clipboard. No API, no cloud, no cost.**

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License MIT](https://img.shields.io/badge/license-MIT-green)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)

## What is this?

A local speech-to-text tool that runs entirely on your machine. Press a hotkey, speak, press again, and your transcription is in the clipboard. Uses OpenAI's Whisper model via [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for fast local inference. No API keys, no internet required (after first model download), no recurring costs. Your audio never leaves your computer.

## Features

- **Global hotkey** to start/stop recording from anywhere
- **100% local** inference with faster-whisper (CTranslate2 backend)
- **GPU acceleration** with CUDA, or runs on CPU
- **Continuous mode** for long transcriptions (meeting notes, lectures)
- **Auto-copies** transcription to clipboard
- **Configurable** via YAML file or CLI flags
- **Zero temp files** - all audio processing happens in memory

## Quick Start

```bash
git clone https://github.com/thexiox/whisper-local.git
cd whisper-local
pip install -r requirements.txt
python -m whisper_local
```

The first run downloads the Whisper model (~3GB for `large-v3`). After that, it works offline.

## Available Models

| Model | Disk Size | RAM (CPU) | VRAM (GPU) | Quality | Speed |
|-------|-----------|-----------|------------|---------|-------|
| `tiny` | ~75 MB | ~1 GB | ~1 GB | Low | Fastest |
| `base` | ~150 MB | ~1 GB | ~1 GB | Fair | Fast |
| `small` | ~500 MB | ~2 GB | ~2 GB | Good | Medium |
| `medium` | ~1.5 GB | ~5 GB | ~4 GB | Great | Slow |
| `large-v3` | ~3 GB | ~10 GB | ~6 GB | Best | Slowest |

Start with `tiny` to test, then move up based on your hardware and accuracy needs:

```bash
python -m whisper_local --model tiny
```

## Configuration

Copy the example config and edit as needed:

```bash
cp config.example.yaml config.yaml
```

See `config.example.yaml` for all options with documentation. CLI flags override config file values.

Key settings:

```yaml
model_size: "large-v3"    # tiny, base, small, medium, large-v3
language: "es"            # Language code or "auto" for detection
device: "auto"            # auto, cpu, cuda
hotkey: "<ctrl>+<shift>+r"
```

## Usage

### Basic

```bash
# Run with defaults (loads config.yaml or uses built-in defaults)
python -m whisper_local

# Use a specific model
python -m whisper_local --model small

# Force Spanish transcription
python -m whisper_local --language es

# Auto-detect language
python -m whisper_local --language auto
```

### Microphone Selection

```bash
# List available microphones
python -m whisper_local --list-mics

# Use a specific microphone by index
python -m whisper_local --mic 2
```

### Continuous Mode

For long recordings (meetings, lectures). Transcribes in chunks while you speak:

```bash
python -m whisper_local --continuous
```

Press the hotkey to start, speak as long as you want, press again to stop. The full transcription is copied to clipboard.

### Force CPU

```bash
python -m whisper_local --cpu
```

### Debug

```bash
python -m whisper_local --verbose
```

## System Requirements

### CPU

Any modern CPU works. 4+ cores recommended for `large-v3`.

### RAM

Depends on the model (see table above). `tiny` runs on 1 GB, `large-v3` needs ~10 GB.

### GPU (Optional)

NVIDIA GPU with CUDA 11.x or 12.x for hardware acceleration. Not required - CPU works fine, just slower.

### Operating System

- **Windows** 10+
- **Linux** Ubuntu 20.04+ (needs PortAudio: `sudo apt install portaudio19-dev`)
- **macOS** 12+ (PortAudio via Homebrew: `brew install portaudio`)

### Python

Python 3.10 or higher.

## Troubleshooting

### No microphone detected

Check that your mic is connected and recognized by the OS. Run `python -m whisper_local --list-mics` to see what's available.

### PortAudio not found (Linux)

```bash
sudo apt install portaudio19-dev
pip install sounddevice --force-reinstall
```

### PortAudio not found (macOS)

```bash
brew install portaudio
pip install sounddevice --force-reinstall
```

### CUDA not found

If you have an NVIDIA GPU but CUDA isn't detected:
1. Make sure you have the [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) installed
2. Verify with `nvidia-smi`
3. Or just use `--cpu` to skip GPU

### Model download fails

The model downloads from Hugging Face on first run. If you're behind a proxy or have no internet:
1. Download the model manually from [Hugging Face](https://huggingface.co/Systran)
2. Place it in the huggingface cache directory (`~/.cache/huggingface/`)

### Hotkey doesn't work

- **Linux**: May need to run with `sudo` or add your user to the `input` group
- **macOS**: Grant accessibility permissions in System Preferences > Privacy & Security > Accessibility
- **Windows**: Run as administrator if the hotkey is captured by another app

## License

MIT - see [LICENSE](LICENSE).
