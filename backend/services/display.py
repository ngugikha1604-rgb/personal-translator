"""
display.py — Glasses-style TUI overlay.

Simulates smart glasses HUD in terminal using ANSI cursor positioning.
Two lenses: LEFT = other-person analysis (intent + reply), RIGHT = user-speech verification.
In-place updates — never scrolls, never appends.
Frame stays static; only lens content + status bar update per turn.
"""

import os
import re
import shutil
import sys
import threading

# ─── ANSI helpers ─────────────────────────────────────────────────────────────

def _pos(row: int, col: int) -> str:
    return f"\033[{row};{col}H"

def _hide_cursor() -> str:
    return "\033[?25l"

def _show_cursor() -> str:
    return "\033[?25h"

# ─── Colors ───────────────────────────────────────────────────────────────────

RST = "\033[0m"
BLD = "\033[1m"
DIM = "\033[90m"
GRN = "\033[92m"
YLW = "\033[93m"
RED = "\033[91m"
CYN = "\033[96m"
WHT = "\033[97m"

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _vlen(text: str) -> int:
    """Visible length — strips ANSI codes."""
    return len(re.sub(r"\033\[[0-9;]*m", "", text))

def _clip(text: str, max_len: int) -> str:
    """Truncate with ellipsis if needed."""
    if len(text) <= max_len:
        return text
    return text[:max_len - 1] + "…"


def _word_wrap(text: str, width: int, max_lines: int = 2) -> list[str]:
    """Word-wrap text into at most max_lines lines of width chars each."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        w = word[:width]          # hard-clip a single very long word
        if not current:
            current = w
        elif len(current) + 1 + len(w) <= width:
            current += " " + w
        else:
            lines.append(current)
            current = w
            if len(lines) >= max_lines:
                break
    if current and len(lines) < max_lines:
        lines.append(current)
    return lines[:max_lines]


# ─── Display ──────────────────────────────────────────────────────────────────

class Display:
    """In-place terminal display simulating glasses HUD.

    Layout (60 chars wide):
        ╔══════════════ GLASSES VIEW ═══════════════╗   ← row 1
        ║                                           ║   ← row 2
        ║  ┌─ OTHER ─────────────────┐  ┌─ YOU ──┐ ║   ← row 3
        ║  │                         │  │         │ ║   ← row 4-7 (content)
        ║  └─────────────────────────┘  └─────────┘ ║   ← row 8
        ║                                           ║   ← row 9
        ║  · listening...                 ⚡450ms   ║   ← row 10
        ╚═══════════════════════════════════════════╝   ← row 11
    """

    # Layout constants (ANSI columns are 1-indexed)
    W           = 60
    LW          = 26                           # left lens inner width
    RW          = 16                           # right lens inner width
    C_LEFT      = 5                            # left content starts here (after "║  │")
    C_RIGHT     = 41                           # right content starts here
    LEFT_TITLE  = " OTHER "
    RIGHT_TITLE = " YOU "
    # Headers: "┌─{title}{'─' * (LW - len(title) - 1)}┐"  → total = LW + 2
    LEFT_HEADER = f"┌─{LEFT_TITLE}{'─' * (LW - len(LEFT_TITLE) - 1)}┐"
    RIGHT_HEADER= f"┌─{RIGHT_TITLE}{'─' * (RW - len(RIGHT_TITLE) - 1)}┐"
    LEFT_FOOT   = f"└{'─' * LW}┘"
    RIGHT_FOOT  = f"└{'─' * RW}┘"
    # Gap between the two lens panels
    _GAP        = W - LW - RW - 10             # = 60 - 26 - 16 - 10 = 8

    def __init__(self):
        self._tokens_accumulated = 0
        self._last_timing_ms = 0
        self._lock = threading.Lock()
        self._stream_buf = ""
        self._enable_windows_ansi()
        self._enabled = self._is_real_terminal()

        if self._enabled:
            try:
                tw, th = shutil.get_terminal_size((60, 24))
            except Exception:
                tw, th = 60, 24
            if tw < 50:
                self._enabled = False

    @staticmethod
    def _enable_windows_ansi() -> None:
        """Enable ANSI/VT100 in Windows console.

        Without this, cursor-positioning codes like \033[6;5H are printed
        as raw text on Windows instead of moving the cursor — which is why
        the glasses frame appears but content updates go to the wrong place.
        """
        if os.name != "nt":
            return
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11)   # STD_OUTPUT_HANDLE
            mode   = ctypes.c_ulong()
            kernel32.GetConsoleMode(handle, ctypes.byref(mode))
            # ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
        except Exception:
            pass

    @staticmethod
    def _is_real_terminal() -> bool:
        """Detect real terminal — works around MSYS2/mintty isatty() issues."""
        if sys.stdout.isatty():
            return True
        # MSYS2 / Cygwin / mintty
        if os.environ.get("MSYSTEM", ""):
            return True
        # Windows Terminal
        if os.environ.get("WT_SESSION", ""):
            return True
        # VS Code integrated terminal
        if os.environ.get("TERM_PROGRAM", ""):
            return True
        # Generic: TERM=xterm-256color etc.
        term = os.environ.get("TERM", "")
        if term and term != "dumb":
            return True
        # Windows native console handle (catches isatty() lying on Windows)
        if os.name == "nt":
            try:
                import ctypes
                h = ctypes.windll.kernel32.GetStdHandle(-11)
                if h and h != -1:
                    return True
            except Exception:
                pass
        # User override
        if os.environ.get("FORCE_TTY", ""):
            return True
        return False

    # ─── Public API ──────────────────────────────────────────────────────────

    def header(self) -> None:
        """Draw static frame, hide cursor, prepare for updates."""
        if not self._enabled:
            print(f"\n{BLD}Conversation Copilot{RST}")
            print(f"{DIM}Hold SPACE • Q to quit{RST}\n")
            return

        with self._lock:
            sys.stdout.write(_hide_cursor())
            sys.stdout.write("\033[2J\033[H")
            sys.stdout.flush()
            self._draw_frame()
            sys.stdout.flush()

    def transcript(self, text: str) -> None:
        """Show what the OTHER person said in rows 4-5 of LEFT lens.

        Called immediately after STT completes — before the LLM call —
        so the user sees the transcript at once while analysis is running.
        Row 6 shows '···' as a visual cue that the AI is thinking.
        """
        if not self._enabled:
            return

        with self._lock:
            self._stream_buf = ""
            self._clear_left()
            lines = _word_wrap(text, self.LW, max_lines=2)
            for i, line in enumerate(lines[:2]):
                sys.stdout.write(
                    _pos(4 + i, self.C_LEFT)
                    + f"{DIM}{WHT}{_clip(line, self.LW)}{RST}"
                )
            # Row 6 placeholder while LLM is running
            sys.stdout.write(_pos(6, self.C_LEFT) + f"{DIM}···{RST}")
            sys.stdout.flush()

    def stream_reply_char(self, ch: str) -> None:
        """Append one decoded reply character to row 7 as the LLM streams.

        Called by _StreamingReplyExtractor in copilot.py for each char of
        the 'reply' JSON field. Gives the glasses-style 'text appearing'
        effect in real time.
        """
        if not self._enabled:
            return

        with self._lock:
            self._stream_buf += ch
            visible = _clip(self._stream_buf, self.LW)
            pad     = self.LW - len(visible)
            sys.stdout.write(
                _pos(7, self.C_LEFT)
                + f"{BLD}{GRN}{visible}{' ' * pad}{RST}"
            )
            sys.stdout.flush()

    def result(self, intent: str, reply: str,
               understanding_check: str | None = None,
               timing_ms: int = 0,
               tokens: dict | None = None) -> None:
        """Finalize LEFT lens: check @ row 5, intent @ row 6, reply @ row 7."""
        if not self._enabled:
            self._fallback_result(intent, reply, understanding_check, timing_ms, tokens)
            return

        with self._lock:
            self._stream_buf = ""
            pad = " " * self.LW
            # Clear rows 5-7 (understanding_check replaces transcript line 2)
            sys.stdout.write(_pos(5, self.C_LEFT) + pad)
            sys.stdout.write(_pos(6, self.C_LEFT) + pad)
            sys.stdout.write(_pos(7, self.C_LEFT) + pad)
            # Understanding check @ row 5 (yellow dim, only when non-null)
            if understanding_check:
                sys.stdout.write(_pos(5, self.C_LEFT) + f"{DIM}{YLW}⚠ {_clip(understanding_check, self.LW - 2)}{RST}")
            # Intent (dim) @ row 6
            sys.stdout.write(_pos(6, self.C_LEFT) + f"{DIM}{_clip(intent, self.LW)}{RST}")
            # Reply (bold green) @ row 7
            sys.stdout.write(_pos(7, self.C_LEFT) + f"{BLD}{GRN}{_clip(reply, self.LW)}{RST}")
            self._write_metrics(timing_ms, tokens)
            sys.stdout.flush()

    def verification(self, correct: bool, warning: str | None,
                     llm_ms: int) -> None:
        """Fill RIGHT lens: ✓ understood or ⚠ warning."""
        if not self._enabled:
            return

        with self._lock:
            self._clear_right()
            if correct:
                sys.stdout.write(_pos(6, self.C_RIGHT) +
                                 f"{GRN}✓ understood{RST}")
                sys.stdout.write(_pos(7, self.C_RIGHT) +
                                 f"{DIM}({llm_ms}ms){RST}")
            else:
                msg = _clip(warning or "check!", self.RW - 2)
                sys.stdout.write(_pos(6, self.C_RIGHT) +
                                 f"{YLW}⚠ {msg}{RST}")
            sys.stdout.flush()

    def status(self, message: str) -> None:
        """Write status + timing to row 10."""
        if not self._enabled:
            return

        with self._lock:
            # Clear inner area cols 2→W-2 (leave both ║ borders)
            sys.stdout.write(_pos(10, 2) + " " * (self.W - 3))
            # Status message — col 4 gives "║  · message" indent
            sys.stdout.write(_pos(10, 4) + f"{DIM}· {message}{RST}")
            # Timing right-aligned — ⚡ is 2 visual cols, -2 keeps 1-char buffer before ║
            if self._last_timing_ms:
                t = f"⚡{self._last_timing_ms}ms"
                col = self.W - len(t) - 2
                sys.stdout.write(_pos(10, col) + f"{DIM}{t}{RST}")
            sys.stdout.flush()

    def error(self, message: str) -> None:
        self.status(f"{RED}✗ {message}{RST}")

    def session_summary(self, turn_count: int, elapsed_sec: float) -> None:
        """Restore cursor and print summary below frame."""
        if not self._enabled:
            return

        with self._lock:
            sys.stdout.write(_pos(12, 0) + _show_cursor())

        avg = self._tokens_accumulated // max(turn_count, 1)
        cost = (self._tokens_accumulated / 1_000_000) * 0.59
        print(f"\n{DIM}┌─ Session ──────────────────────────┐{RST}")
        print(f"{DIM}│{RST}  {turn_count} turns in {elapsed_sec:.0f}s")
        print(f"{DIM}│{RST}  {self._tokens_accumulated} tok (~{avg}/turn)")
        print(f"{DIM}│{RST}  Cost ~${cost:.5f}")
        print(f"{DIM}└─────────────────────────────────────┘{RST}")

    # ─── Drawing ─────────────────────────────────────────────────────────────

    def _draw_frame(self) -> None:
        W = self.W
        g = self._GAP

        # Row 1: top border
        title = " GLASSES VIEW "
        side = (W - len(title) - 2) // 2
        print(f"{DIM}╔{'═' * side}{BLD}{title}{RST}"
              f"{DIM}{'═' * (W - side - len(title) - 2)}╗{RST}")

        # Row 2: blank
        print(f"{DIM}║{' ' * (W - 2)}║{RST}")

        # Row 3: lens headers
        print(f"{DIM}║{RST}  {DIM}{self.LEFT_HEADER}{RST}"
              f"{' ' * g}{DIM}{self.RIGHT_HEADER}{RST}  {DIM}║{RST}")

        # Rows 4-7: lens content (empty)
        for _ in range(4):
            print(f"{DIM}║{RST}  {DIM}│{' ' * self.LW}│{RST}"
                  f"{' ' * g}{DIM}│{' ' * self.RW}│{RST}  {DIM}║{RST}")

        # Row 8: lens footers
        print(f"{DIM}║{RST}  {DIM}{self.LEFT_FOOT}{RST}"
              f"{' ' * g}{DIM}{self.RIGHT_FOOT}{RST}  {DIM}║{RST}")

        # Row 9: blank
        print(f"{DIM}║{' ' * (W - 2)}║{RST}")

        # Row 10: status (initially blank)
        print(f"{DIM}║{' ' * (W - 2)}║{RST}")

        # Row 11: bottom border
        print(f"{DIM}╚{'═' * (W - 2)}╝{RST}")

    def _clear_left(self) -> None:
        s = " " * self.LW
        for row in (4, 5, 6, 7):
            sys.stdout.write(_pos(row, self.C_LEFT) + s)

    def _clear_right(self) -> None:
        s = " " * self.RW
        for row in (4, 5, 6, 7):
            sys.stdout.write(_pos(row, self.C_RIGHT) + s)

    def _write_metrics(self, timing_ms: int, tokens: dict | None) -> None:
        """Track tokens for session_summary; timing stored for status bar."""
        if tokens:
            p = tokens.get("prompt_tokens", 0)
            c = tokens.get("completion_tokens", 0)
            t = p + c or tokens.get("total_tokens", 0)
            if t:
                self._tokens_accumulated += t
        if timing_ms:
            self._last_timing_ms = timing_ms

    # ─── Fallback (non-TTY / narrow terminal) ────────────────────────────────

    def _fallback_result(self, intent: str, reply: str,
                         understanding_check: str | None = None,
                         timing_ms: int = 0,
                         tokens: dict | None = None) -> None:
        div = f"{DIM}{'─' * 50}{RST}"
        print(f"\n{div}")
        print(f"  {BLD}{GRN}{reply}{RST}")
        if understanding_check:
            print(f"  {DIM}{YLW}⚠ {understanding_check}{RST}")
        print(f"  {DIM}{intent}{RST}")

        parts = []
        if timing_ms:
            parts.append(f"{timing_ms}ms total")
        if tokens:
            p = tokens.get("prompt_tokens", 0)
            c = tokens.get("completion_tokens", 0)
            if p or c:
                parts.append(f"{p}p + {c}c = {p + c} tok")
            elif tokens.get("total_tokens"):
                parts.append(f"{tokens['total_tokens']} tok")
        if parts:
            print(f"  {DIM}⚡ {' | '.join(parts)}{RST}")
        print(div)


# Singleton
display = Display()
