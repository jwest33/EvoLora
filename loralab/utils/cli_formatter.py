"""CLI formatting utilities for professional output

Provides colored text, progress indicators, and formatted tables.
"""

import sys
import shutil
from typing import Optional, List, Tuple, Any
from enum import Enum

# Try to import colorama for cross-platform support
try:
    from colorama import init, Fore, Back, Style
    init(autoreset=True)  # Auto-reset colors after each print
    COLORS_AVAILABLE = True
except ImportError:
    COLORS_AVAILABLE = False
    # Fallback - create dummy color constants
    class Fore:
        BLACK = BLUE = CYAN = GREEN = MAGENTA = RED = WHITE = YELLOW = ""
        LIGHTBLACK_EX = LIGHTBLUE_EX = LIGHTCYAN_EX = LIGHTGREEN_EX = ""
        LIGHTMAGENTA_EX = LIGHTRED_EX = LIGHTWHITE_EX = LIGHTYELLOW_EX = ""
        RESET = ""

    class Back:
        BLACK = BLUE = CYAN = GREEN = MAGENTA = RED = WHITE = YELLOW = ""
        RESET = ""

    class Style:
        BRIGHT = DIM = NORMAL = RESET_ALL = ""


class Level(Enum):
    """Output level types"""
    INFO = "INFO"
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    ERROR = "ERROR"
    DEBUG = "DEBUG"
    HEADER = "HEADER"
    SUBHEADER = "SUBHEADER"


class CLIFormatter:
    """Formatter for CLI output with colors and styling"""

    @staticmethod
    def get_terminal_width() -> int:
        """Get terminal width for dynamic formatting"""
        return shutil.get_terminal_size((80, 20)).columns

    @staticmethod
    def print_header(text: str, char: str = "=", color: str = Fore.CYAN):
        """Print a formatted header"""
        width = CLIFormatter.get_terminal_width()
        line = char * width

        print(f"\n{color}{line}")
        print(f"{color}{Style.BRIGHT}{text.center(width)}")
        print(f"{color}{line}{Style.RESET_ALL}")

    @staticmethod
    def print_subheader(text: str, char: str = "-", color: str = Fore.BLUE):
        """Print a formatted subheader"""
        width = CLIFormatter.get_terminal_width()
        line = char * min(60, width)

        print(f"\n{color}{text}")
        print(f"{color}{line}{Style.RESET_ALL}")

    @staticmethod
    def print_status(label: str, value: str, label_color: str = Fore.CYAN,
                    value_color: str = Fore.WHITE):
        """Print a status line with colored label and value"""
        print(f"{label_color}{label}: {value_color}{value}{Style.RESET_ALL}")

    @staticmethod
    def print_progress(current: int, total: int, label: str = "Progress",
                      bar_length: int = 40):
        """Print a progress bar"""
        percent = current / total if total > 0 else 0
        filled = int(bar_length * percent)

        # Choose color based on progress
        if percent < 0.33:
            color = Fore.RED
        elif percent < 0.66:
            color = Fore.YELLOW
        else:
            color = Fore.GREEN

        bar = f"[{color}{'█' * filled}{Fore.WHITE}{'░' * (bar_length - filled)}{Style.RESET_ALL}]"

        print(f"\r{label}: {bar} {percent:6.1%} ({current}/{total})", end="")
        if current >= total:
            print()  # New line when complete

    @staticmethod
    def print_table(headers: List[str], rows: List[List[Any]],
                   colors: Optional[List[str]] = None):
        """Print a formatted table"""
        # Calculate column widths
        widths = [len(str(h)) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(str(cell)))

        # Adjust for terminal width
        terminal_width = CLIFormatter.get_terminal_width()
        total_width = sum(widths) + len(widths) * 3 - 1

        if total_width > terminal_width:
            scale = terminal_width / total_width
            widths = [int(w * scale) for w in widths]

        # Print header
        header_line = " | ".join(str(h).ljust(w) for h, w in zip(headers, widths))
        print(f"\n{Fore.CYAN}{Style.BRIGHT}{header_line}")
        print(f"{Fore.CYAN}{'-' * len(header_line)}{Style.RESET_ALL}")

        # Print rows
        for row in rows:
            row_str = " | ".join(str(cell)[:w].ljust(w) for cell, w in zip(row, widths))
            if colors and len(colors) > rows.index(row):
                print(f"{colors[rows.index(row)]}{row_str}{Style.RESET_ALL}")
            else:
                print(row_str)

    @staticmethod
    def print_result(success: bool, message: str):
        """Print a result message with appropriate styling"""
        if success:
            icon = "[SUCCESS]"
            color = Fore.GREEN
        else:
            icon = "[FAILED]"
            color = Fore.RED

        print(f"{color}{Style.BRIGHT}{icon} {Style.NORMAL}{message}{Style.RESET_ALL}")

    @staticmethod
    def print_metric(name: str, value: float, unit: str = "",
                    good_threshold: Optional[float] = None,
                    bad_threshold: Optional[float] = None):
        """Print a metric with color coding based on thresholds"""
        # Determine color based on thresholds
        color = Fore.WHITE
        if good_threshold is not None and bad_threshold is not None:
            if good_threshold > bad_threshold:  # Higher is better
                if value >= good_threshold:
                    color = Fore.GREEN
                elif value <= bad_threshold:
                    color = Fore.RED
                else:
                    color = Fore.YELLOW
            else:  # Lower is better
                if value <= good_threshold:
                    color = Fore.GREEN
                elif value >= bad_threshold:
                    color = Fore.RED
                else:
                    color = Fore.YELLOW

        formatted_value = f"{value:.2f}" if isinstance(value, float) else str(value)
        print(f"  {Fore.CYAN}{name}: {color}{formatted_value}{unit}{Style.RESET_ALL}")

    @staticmethod
    def print_list_item(text: str, level: int = 0, bullet: str = "•",
                       color: str = Fore.WHITE):
        """Print a list item with indentation"""
        indent = "  " * level
        print(f"{indent}{Fore.CYAN}{bullet} {color}{text}{Style.RESET_ALL}")

    @staticmethod
    def print_generation_header(generation: int, total: int):
        """Print a generation header for evolution"""
        width = CLIFormatter.get_terminal_width()
        progress = generation / total if total > 0 else 0

        # Choose color based on progress
        if progress < 0.33:
            color = Fore.YELLOW
        elif progress < 0.66:
            color = Fore.CYAN
        else:
            color = Fore.GREEN

        print(f"\n{color}{'=' * width}")
        print(f"{color}{Style.BRIGHT}GENERATION {generation}/{total}".center(width))
        print(f"{color}{'=' * width}{Style.RESET_ALL}")

    @staticmethod
    def print_variant_status(variant_id: str, status: str, metrics: dict = None):
        """Print variant status with metrics"""
        # Status badge
        status_colors = {
            'TRAINING': Fore.YELLOW,
            'EVALUATING': Fore.CYAN,
            'COMPLETE': Fore.GREEN,
            'FAILED': Fore.RED,
            'SURVIVED': Fore.LIGHTGREEN_EX,
            'ELIMINATED': Fore.LIGHTRED_EX
        }

        color = status_colors.get(status.upper(), Fore.WHITE)

        print(f"\n{Fore.BLUE}[{variant_id}] {color}[{status.upper()}]{Style.RESET_ALL}")

        if metrics:
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {Fore.CYAN}{key}: {Fore.WHITE}{value:.4f}")
                else:
                    print(f"  {Fore.CYAN}{key}: {Fore.WHITE}{value}")

    @staticmethod
    def print_summary_box(title: str, content: dict, color: str = Fore.GREEN):
        """Print a summary box with key-value pairs"""
        width = min(60, CLIFormatter.get_terminal_width() - 4)

        # Top border
        print(f"\n{color}╔{'═' * (width - 2)}╗")

        # Title
        title_text = f" {title} "
        padding = (width - len(title_text) - 2) // 2
        print(f"{color}║{' ' * padding}{Style.BRIGHT}{title_text}{Style.NORMAL}{' ' * (width - padding - len(title_text) - 2)}║")

        # Separator
        print(f"{color}╠{'═' * (width - 2)}╣")

        # Content
        for key, value in content.items():
            if isinstance(value, float):
                line = f" {key}: {value:.2%}" if value < 1.5 else f" {key}: {value:.2f}"
            else:
                line = f" {key}: {value}"

            # Truncate if too long
            if len(line) > width - 4:
                line = line[:width - 7] + "..."

            print(f"{color}║{Fore.WHITE}{line:<{width - 2}}{color}║")

        # Bottom border
        print(f"{color}╚{'═' * (width - 2)}╝{Style.RESET_ALL}")

    @staticmethod
    def clear_line():
        """Clear current line"""
        print('\r' + ' ' * (CLIFormatter.get_terminal_width() - 1), end='\r')

    @staticmethod
    def print_error(message: str):
        """Print an error message"""
        print(f"{Fore.RED}{Style.BRIGHT}[ERROR] {Style.NORMAL}{message}{Style.RESET_ALL}")

    @staticmethod
    def print_warning(message: str):
        """Print a warning message"""
        print(f"{Fore.YELLOW}[WARNING] {message}{Style.RESET_ALL}")

    @staticmethod
    def print_info(message: str):
        """Print an info message"""
        print(f"{Fore.BLUE}[INFO] {message}{Style.RESET_ALL}")

    @staticmethod
    def print_success(message: str):
        """Print a success message"""
        print(f"{Fore.GREEN}{Style.BRIGHT}[SUCCESS] {Style.NORMAL}{message}{Style.RESET_ALL}")

    @staticmethod
    def format_time(seconds: float) -> str:
        """Format time in human-readable format"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"


class SpinnerProgress:
    """Simple spinner for long-running operations"""

    SPINNERS = {
        'dots': ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'],
        'line': ['-', '\\', '|', '/'],
        'arrow': ['←', '↖', '↑', '↗', '→', '↘', '↓', '↙'],
        'bar': ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█', '▇', '▆', '▅', '▄', '▃', '▂']
    }

    def __init__(self, text: str = "Processing", spinner_type: str = 'dots'):
        self.text = text
        self.spinner = self.SPINNERS.get(spinner_type, self.SPINNERS['dots'])
        self.index = 0

    def update(self):
        """Update spinner animation"""
        CLIFormatter.clear_line()
        frame = self.spinner[self.index % len(self.spinner)]
        print(f"{Fore.CYAN}{frame} {self.text}{Style.RESET_ALL}", end='', flush=True)
        self.index += 1

    def complete(self, message: str = "Complete"):
        """Complete spinner with message"""
        CLIFormatter.clear_line()
        CLIFormatter.print_success(f"{self.text} - {message}")
