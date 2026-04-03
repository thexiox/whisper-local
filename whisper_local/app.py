"""Main application orchestrator for whisper-local."""

import logging
import signal
import threading
import time
from typing import Callable

from pynput import keyboard
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from whisper_local.clipboard import copy_to_clipboard
from whisper_local.config import AppConfig
from whisper_local.recorder import AudioRecorder
from whisper_local.transcriber import WhisperTranscriber

logger = logging.getLogger(__name__)

MIN_RECORDING_SECONDS = 0.5


class WhisperLocalApp:
    """Orchestrates recording, transcription, and clipboard operations."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._console = Console()
        self._shutdown_event = threading.Event()
        self._is_recording = False
        self._hotkey_lock = threading.Lock()

        # Check clipboard availability
        self._clipboard_available = copy_to_clipboard("")
        if not self._clipboard_available:
            self._console.print(
                "[yellow]Warning: Clipboard unavailable. "
                "Text will be shown in terminal only.[/yellow]"
            )

        # Initialize components
        self._console.print("[dim]Loading Whisper model...[/dim]")
        self._transcriber = WhisperTranscriber(
            model_size=config.model_size,
            device=config.device,
            compute_type=config.compute_type,
            language=config.language,
        )
        self._recorder = AudioRecorder(
            sample_rate=config.sample_rate,
            device_index=config.mic_index,
        )

    def run(self) -> None:
        """Main application loop."""
        self._show_status_panel()

        # Setup signal handler for Ctrl+C
        original_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._signal_handler)

        # Parse hotkey combination
        hotkey_combo = self._parse_hotkey(self._config.hotkey)

        try:
            with keyboard.GlobalHotKeys({hotkey_combo: self._on_hotkey}) as listener:
                listener.daemon = True
                self._console.print(
                    f"\nReady. Press [bold cyan]{self._config.hotkey}[/bold cyan] to record.\n"
                )

                while not self._shutdown_event.is_set():
                    self._shutdown_event.wait(timeout=0.5)

        except Exception as e:
            self._console.print(f"[red]Error with hotkey listener: {e}[/red]")
            logger.exception("Hotkey listener failed")
        finally:
            signal.signal(signal.SIGINT, original_sigint)
            self._cleanup()

    def _on_hotkey(self) -> None:
        """Called when the global hotkey is pressed. Toggles recording."""
        with self._hotkey_lock:
            if self._is_recording:
                self._stop_and_transcribe()
            else:
                self._start_recording()

    def _start_recording(self) -> None:
        """Start recording audio."""
        self._is_recording = True
        self._recording_start_time = time.perf_counter()
        self._recorder.start()

        if self._config.continuous:
            self._continuous_texts: list[str] = []
            self._continuous_stop = threading.Event()
            self._continuous_thread = threading.Thread(
                target=self._continuous_worker, daemon=True
            )
            self._continuous_thread.start()
            self._console.print(
                "[bold red]Recording (continuous)... "
                f"press {self._config.hotkey} to stop[/bold red]"
            )
        else:
            self._console.print(
                f"[bold red]Recording... press {self._config.hotkey} to stop[/bold red]"
            )

    def _stop_and_transcribe(self) -> None:
        """Stop recording and transcribe the audio."""
        self._is_recording = False
        recording_duration = time.perf_counter() - self._recording_start_time

        if self._config.continuous:
            self._continuous_stop.set()
            # Get any remaining audio
            remaining = self._recorder.stop()
            if len(remaining) > 0:
                self._console.print("[dim]Transcribing final chunk...[/dim]")
                result = self._transcriber.transcribe(remaining)
                if result.text.strip():
                    self._continuous_texts.append(result.text.strip())

            full_text = " ".join(self._continuous_texts)
            self._show_result(full_text, recording_duration)
        else:
            audio = self._recorder.stop()

            if len(audio) / self._config.sample_rate < MIN_RECORDING_SECONDS:
                self._console.print(
                    "[yellow]Recording too short. Try holding the hotkey longer.[/yellow]\n"
                )
                return

            self._console.print("[dim]Transcribing...[/dim]")
            result = self._transcriber.transcribe(audio)
            self._show_result(result.text, result.duration_seconds)

    def _continuous_worker(self) -> None:
        """Background worker for continuous mode. Transcribes chunks periodically."""
        chunk_seconds = self._config.continuous_chunk_seconds

        while not self._continuous_stop.is_set():
            self._continuous_stop.wait(timeout=chunk_seconds)
            if self._continuous_stop.is_set():
                break

            chunk = self._recorder.get_chunk()
            if len(chunk) == 0:
                continue

            try:
                result = self._transcriber.transcribe(chunk)
                if result.text.strip():
                    self._continuous_texts.append(result.text.strip())
                    self._console.print(f"[dim]  > {result.text.strip()}[/dim]")
            except Exception as e:
                logger.warning("Chunk transcription failed: %s", e)

    def _show_result(self, text: str, duration: float) -> None:
        """Display transcription result and copy to clipboard."""
        if not text.strip():
            self._console.print("[yellow]No speech detected.[/yellow]\n")
            return

        self._console.print(f"\n[bold white]{text}[/bold white]")
        self._console.print(f"[dim]Transcription took {duration:.1f}s[/dim]")

        if copy_to_clipboard(text):
            self._console.print("[green]Copied to clipboard[/green]\n")
        else:
            self._console.print("[yellow]Clipboard unavailable — text shown above[/yellow]\n")

    def _show_status_panel(self) -> None:
        """Display the startup status panel."""
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column(style="bold cyan", width=12)
        table.add_column()

        table.add_row("Model", self._config.model_size)
        table.add_row(
            "Device",
            f"{self._config.device.upper()} ({self._config.compute_type})"
        )
        table.add_row("Microphone", AudioRecorder.get_default_device_name())
        table.add_row("Hotkey", self._config.hotkey)
        table.add_row("Language", self._config.language)

        if self._config.continuous:
            table.add_row("Mode", f"Continuous ({self._config.continuous_chunk_seconds}s chunks)")

        panel = Panel(table, title="[bold]whisper-local[/bold]", border_style="cyan")
        self._console.print(panel)

    def _signal_handler(self, signum: int, frame: object) -> None:
        """Handle Ctrl+C for graceful shutdown."""
        self._shutdown_event.set()

    def _cleanup(self) -> None:
        """Clean up resources on exit."""
        if self._recorder.is_recording:
            self._recorder.stop()
        self._console.print("\n[dim]Bye![/dim]")

    @staticmethod
    def _parse_hotkey(hotkey_str: str) -> str:
        """Convert config hotkey format to pynput GlobalHotKeys format."""
        return hotkey_str
