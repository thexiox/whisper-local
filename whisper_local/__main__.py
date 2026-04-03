"""Entry point for whisper-local: python -m whisper_local."""

import argparse
import logging
import sys

from rich.console import Console
from rich.table import Table

from whisper_local import __version__
from whisper_local.config import VALID_MODELS, load_config
from whisper_local.recorder import AudioRecorder


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="whisper-local",
        description="Local speech-to-text. Hotkey → record → transcribe → clipboard.",
        epilog=(
            "Examples:\n"
            "  python -m whisper_local                    # Run with defaults\n"
            "  python -m whisper_local --model tiny       # Use tiny model (fastest)\n"
            "  python -m whisper_local --language auto     # Auto-detect language\n"
            "  python -m whisper_local --continuous        # Continuous transcription mode\n"
            "  python -m whisper_local --cpu               # Force CPU even if GPU available\n"
            "  python -m whisper_local --list-mics         # Show available microphones\n"
            "  python -m whisper_local --mic 2             # Use microphone at index 2\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--list-mics",
        action="store_true",
        help="List available microphones and exit",
    )
    parser.add_argument(
        "--mic",
        type=int,
        metavar="N",
        help="Use microphone at index N (see --list-mics)",
    )
    parser.add_argument(
        "--model",
        choices=VALID_MODELS,
        metavar="SIZE",
        help=f"Whisper model size ({', '.join(VALID_MODELS)})",
    )
    parser.add_argument(
        "--language",
        metavar="LANG",
        help='Language code (e.g., "es", "en") or "auto" for detection',
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Continuous transcription mode (transcribes in chunks while recording)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference even if CUDA GPU is available",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    return parser


def _list_microphones() -> None:
    """Print available microphones and exit."""
    console = Console()
    devices = AudioRecorder.list_devices()

    if not devices:
        console.print("[red]No input devices found.[/red]")
        sys.exit(1)

    table = Table(title="Available Microphones")
    table.add_column("Index", style="cyan", justify="right")
    table.add_column("Name", style="white")
    table.add_column("Channels", justify="right")
    table.add_column("Sample Rate", justify="right")

    for dev in devices:
        table.add_row(
            str(dev["index"]),
            dev["name"],
            str(dev["channels"]),
            f"{dev['sample_rate']:.0f} Hz",
        )

    console.print(table)


def main() -> None:
    """Main entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Handle --list-mics
    if args.list_mics:
        _list_microphones()
        return

    # Load config with CLI overrides
    console = Console()
    try:
        config = load_config(vars(args))
    except ValueError as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        sys.exit(1)

    # Run the app
    try:
        from whisper_local.app import WhisperLocalApp
        app = WhisperLocalApp(config)
        app.run()
    except ImportError as e:
        console.print(f"[red]Missing dependency: {e}[/red]")
        console.print("[dim]Run: pip install -r requirements.txt[/dim]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger = logging.getLogger(__name__)
        logger.exception("Unhandled error")
        sys.exit(1)


if __name__ == "__main__":
    main()
