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
    """Formatter for CLI output with synthwave colors and styling"""

    # Synthwave color palette
    NEON_PINK = Fore.MAGENTA + Style.BRIGHT
    NEON_CYAN = Fore.CYAN + Style.BRIGHT
    NEON_PURPLE = Fore.LIGHTMAGENTA_EX
    ELECTRIC_BLUE = Fore.LIGHTBLUE_EX + Style.BRIGHT
    NEON_GREEN = Fore.LIGHTGREEN_EX + Style.BRIGHT
    HOT_PINK = Fore.LIGHTRED_EX + Style.BRIGHT
    SUNSET_ORANGE = Fore.LIGHTYELLOW_EX
    DEEP_PURPLE = Fore.MAGENTA

    @staticmethod
    def get_terminal_width() -> int:
        """Get terminal width for dynamic formatting"""
        return shutil.get_terminal_size((80, 20)).columns

    @staticmethod
    def print_header(text: str, char: str = "=", color: str = None):
        """Print a formatted header with synthwave styling"""
        if color is None:
            color = CLIFormatter.NEON_PINK
        width = CLIFormatter.get_terminal_width()
        line = char * width

        print(f"\n{color}{line}")
        print(f"{CLIFormatter.ELECTRIC_BLUE}{text.center(width)}")
        print(f"{color}{line}{Style.RESET_ALL}")

    @staticmethod
    def print_subheader(text: str, char: str = "-", color: str = None):
        """Print a formatted subheader with synthwave styling"""
        if color is None:
            color = CLIFormatter.NEON_CYAN
        width = CLIFormatter.get_terminal_width()
        line = char * min(60, width)

        print(f"\n{color}{text}")
        print(f"{CLIFormatter.NEON_PURPLE}{line}{Style.RESET_ALL}")

    @staticmethod
    def print_status(label: str, value: str, label_color: str = None,
                    value_color: str = None):
        """Print a status line with synthwave colors"""
        if label_color is None:
            label_color = CLIFormatter.NEON_CYAN
        if value_color is None:
            value_color = CLIFormatter.HOT_PINK
        print(f"{label_color}{label}: {value_color}{value}{Style.RESET_ALL}")

    @staticmethod
    def print_progress(current: int, total: int, label: str = "Progress",
                      bar_length: int = 40):
        """Print a progress bar with synthwave gradient"""
        percent = current / total if total > 0 else 0
        filled = int(bar_length * percent)

        # Synthwave gradient based on progress
        if percent < 0.33:
            color = CLIFormatter.HOT_PINK
        elif percent < 0.66:
            color = CLIFormatter.SUNSET_ORANGE
        else:
            color = CLIFormatter.NEON_GREEN

        bar = f"[{color}{'█' * filled}{CLIFormatter.DEEP_PURPLE}{'░' * (bar_length - filled)}{Style.RESET_ALL}]"

        print(f"\r{label}: {bar} {percent:6.1%} ({current}/{total})", end="")
        if current >= total:
            print()  # New line when complete

    @staticmethod
    def print_box_start(title: str, color: str = Fore.CYAN, width: Optional[int] = None):
        """Print the start of a box with a title"""
        if width is None:
            width = min(80, CLIFormatter.get_terminal_width())

        top_line = "╔" + "═" * (width - 2) + "╗"
        title_line = f"║ {title.center(width - 4)} ║"
        separator = "╚" + "═" * (width - 2) + "╝"

        print(f"{color}{top_line}")
        print(f"{color}{Style.BRIGHT}{title_line}{Style.RESET_ALL}")
        print(f"{color}{separator}{Style.RESET_ALL}")

    @staticmethod
    def print_box_end(color: str = Fore.CYAN, width: Optional[int] = None):
        """Print the end of a box"""
        if width is None:
            width = min(80, CLIFormatter.get_terminal_width())

        bottom_line =  "═" * (width - 2)
        print(f"{color}{bottom_line}{Style.RESET_ALL}")

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

            print(f"{color}║{CLIFormatter.ELECTRIC_BLUE}{line:<{width - 2}}{color}║")

        # Bottom border
        print(f"{color}╚{'═' * (width - 2)}╝{Style.RESET_ALL}")

    @staticmethod
    def clear_line():
        """Clear current line"""
        print('\r' + ' ' * (CLIFormatter.get_terminal_width() - 1), end='\r')

    @staticmethod
    def print_error(message: str):
        """Print an error message with synthwave hot pink"""
        print(f"{CLIFormatter.HOT_PINK}[ERROR] {Style.NORMAL}{message}{Style.RESET_ALL}")

    @staticmethod
    def print_warning(message: str):
        """Print a warning message with synthwave sunset orange"""
        print(f"{CLIFormatter.SUNSET_ORANGE}[WARNING] {message}{Style.RESET_ALL}")

    @staticmethod
    def print_info(message: str):
        """Print an info message with synthwave electric blue"""
        print(f"{CLIFormatter.ELECTRIC_BLUE}[INFO] {Style.NORMAL}{message}{Style.RESET_ALL}")

    @staticmethod
    def print_success(message: str):
        """Print a success message with synthwave neon green"""
        print(f"{CLIFormatter.NEON_GREEN}[SUCCESS] {Style.NORMAL}{message}{Style.RESET_ALL}")

    @staticmethod
    def format_time(seconds: float) -> str:
        """Format time in human-readable format"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"

    @staticmethod
    def print_memory_status(gpu_memory: dict, ram: dict, variant_id: str = ""):
        """Print formatted memory status for monitoring

        Args:
            gpu_memory: Dict with GPU memory stats (allocated, reserved, total, utilization)
            ram: Dict with RAM stats (process, total, percent)
            variant_id: Optional variant identifier
        """
        # Determine GPU memory color based on utilization
        gpu_util = gpu_memory.get('utilization', 0)
        if gpu_util < 70:
            gpu_color = Fore.GREEN
        elif gpu_util < 85:
            gpu_color = Fore.YELLOW
        else:
            gpu_color = Fore.RED

        # Determine RAM color based on percentage
        ram_percent = ram.get('percent', 0)
        if ram_percent < 70:
            ram_color = Fore.GREEN
        elif ram_percent < 85:
            ram_color = Fore.YELLOW
        else:
            ram_color = Fore.RED

        # Format the status line
        variant_str = f"[{variant_id}] " if variant_id else ""

        gpu_str = (f"GPU: {gpu_color}{gpu_memory['allocated']:.2f}/"
                   f"{gpu_memory['reserved']:.2f}/{gpu_memory['total']:.2f}GB "
                   f"({gpu_util:.1f}%){Style.RESET_ALL}")

        ram_str = (f"RAM: {ram_color}{ram['process']:.2f}GB "
                   f"(System: {ram_percent:.1f}%){Style.RESET_ALL}")

        print(f"{Fore.BLUE}[Memory]{Style.RESET_ALL} {variant_str}{gpu_str} | {ram_str}")

    @staticmethod
    def print_memory_diagnostic(title: str, metrics: dict):
        """Print a formatted memory diagnostic section

        Args:
            title: Section title
            metrics: Dict of metric names to (value, unit) tuples or just values
        """
        CLIFormatter.print_subheader(title)

        for name, value in metrics.items():
            if isinstance(value, tuple):
                metric_value, unit = value
                CLIFormatter.print_metric(name, metric_value, unit)
            else:
                CLIFormatter.print_metric(name, value, "")

    @staticmethod
    def format_memory_size(size_gb: float) -> str:
        """Format memory size with appropriate units

        Args:
            size_gb: Size in gigabytes

        Returns:
            Formatted string with appropriate units
        """
        if size_gb < 0.001:
            return f"{size_gb * 1024 * 1024:.1f}KB"
        elif size_gb < 1:
            return f"{size_gb * 1024:.1f}MB"
        else:
            return f"{size_gb:.2f}GB"

    @staticmethod
    def format_training_metrics(metrics: dict, step: Optional[int] = None,
                               total_steps: Optional[int] = None):
        """Format training metrics in a clean, readable layout

        Args:
            metrics: Dictionary of training metrics
            step: Current step number
            total_steps: Total number of steps
        """
        # Header with step info if provided
        if step is not None:
            if total_steps:
                progress = step / total_steps
                CLIFormatter.print_progress(step, total_steps, "Training Progress")
            else:
                print(f"\n{Fore.CYAN}[Step {step}]{Style.RESET_ALL}")

        # Group metrics by category
        core_metrics = {}
        reward_metrics = {}
        completion_metrics = {}
        other_metrics = {}

        for key, value in metrics.items():
            if key in ['loss', 'grad_norm', 'learning_rate', 'kl', 'epoch']:
                core_metrics[key] = value
            elif 'reward' in key.lower():
                reward_metrics[key] = value
            elif 'completion' in key.lower() or 'length' in key.lower():
                completion_metrics[key] = value
            else:
                other_metrics[key] = value

        # Print core metrics
        if core_metrics:
            print(f"\n{Fore.CYAN}{'─' * 40}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}{Style.BRIGHT}Core Metrics:{Style.RESET_ALL}")
            for key, value in core_metrics.items():
                CLIFormatter._print_metric_value(key, value)

        # Print reward metrics
        if reward_metrics:
            print(f"\n{Fore.YELLOW}{Style.BRIGHT}Rewards:{Style.RESET_ALL}")
            for key, value in reward_metrics.items():
                # Clean up nested reward names
                clean_key = key.replace('rewards/', '').replace('custom_reward_func/', '')
                CLIFormatter._print_metric_value(clean_key, value)

        # Print completion metrics
        if completion_metrics:
            print(f"\n{Fore.YELLOW}{Style.BRIGHT}Completions:{Style.RESET_ALL}")
            for key, value in completion_metrics.items():
                clean_key = key.replace('completions/', '')
                CLIFormatter._print_metric_value(clean_key, value)

        # Print other metrics
        if other_metrics:
            print(f"\n{Fore.YELLOW}{Style.BRIGHT}Other:{Style.RESET_ALL}")
            for key, value in other_metrics.items():
                CLIFormatter._print_metric_value(key, value)

        print(f"{Fore.CYAN}{'─' * 40}{Style.RESET_ALL}")

    @staticmethod
    def _print_metric_value(name: str, value: Any):
        """Print a single metric with appropriate formatting

        Args:
            name: Metric name
            value: Metric value
        """
        # Format name
        formatted_name = name.replace('_', ' ').title()

        # Determine color based on metric type and value
        color = Fore.WHITE

        if 'loss' in name.lower():
            # Lower loss is better
            if isinstance(value, (int, float)):
                if value < 0.01:
                    color = Fore.GREEN
                elif value < 0.1:
                    color = Fore.YELLOW
                else:
                    color = Fore.RED
        elif 'reward' in name.lower():
            # Higher reward is usually better
            if isinstance(value, (int, float)):
                if value > 0.5:
                    color = Fore.GREEN
                elif value > 0:
                    color = Fore.YELLOW
                else:
                    color = Fore.RED
        elif 'learning_rate' in name.lower():
            color = Fore.CYAN
        elif 'grad_norm' in name.lower():
            # Check for gradient issues
            if isinstance(value, (int, float)):
                if value == 0:
                    color = Fore.RED  # No gradients
                elif value < 0.001:
                    color = Fore.YELLOW  # Very small gradients
                elif value > 100:
                    color = Fore.RED  # Exploding gradients
                else:
                    color = Fore.GREEN

        # Format value
        if isinstance(value, float):
            if abs(value) < 0.0001:
                formatted_value = f"{value:.2e}"
            elif abs(value) < 1:
                formatted_value = f"{value:.6f}"
            elif abs(value) > 1000:
                formatted_value = f"{value:.2e}"
            else:
                formatted_value = f"{value:.4f}"
        else:
            formatted_value = str(value)

        # Print with padding
        print(f"  {Fore.CYAN}{formatted_name:25s}: {color}{formatted_value}{Style.RESET_ALL}")

    @staticmethod
    def format_grpo_stats(metrics: dict, step: int, total_steps: int):
        """Format GRPO training statistics with synthwave colors

        Args:
            metrics: Dictionary of GRPO training metrics
            step: Current step number
            total_steps: Total number of training steps
        """
        print(f"\n{CLIFormatter.NEON_PINK}{'═' * 60}{Style.RESET_ALL}")
        print(f"{CLIFormatter.ELECTRIC_BLUE}{Style.BRIGHT}Step: {step}/{total_steps}{Style.RESET_ALL}")

        # Extract and display key metrics - use actual values from metrics
        epoch = metrics.get('epoch', 0.0)
        reward = metrics.get('reward', 0.0)
        reward_std = metrics.get('reward_std', 0.0)

        # Display epoch and main reward with actual values
        print(f"{CLIFormatter.NEON_CYAN}Epoch: {CLIFormatter.ELECTRIC_BLUE}{epoch:.1f}{Style.RESET_ALL}")
        print(f"{CLIFormatter.NEON_CYAN}Reward: {CLIFormatter.NEON_GREEN if reward > 0.5 else CLIFormatter.SUNSET_ORANGE if reward > 0 else CLIFormatter.HOT_PINK}{reward:.3f} ± {reward_std:.3f}{Style.RESET_ALL}")

        # Always display reward breakdown
        print(f"\n{CLIFormatter.NEON_PURPLE}Reward Components:{Style.RESET_ALL}")

        # Format individual reward components
        format_exact = metrics.get('rewards/match_format_exactly/mean', 0.0)
        format_approx = metrics.get('rewards/match_format_approximately/mean', 0.0)
        answer_check = metrics.get('rewards/check_answer/mean', 0.0)
        number_extract = metrics.get('rewards/check_numbers/mean', 0.0)

        # Display each component with color coding
        print(f"  {CLIFormatter.NEON_CYAN}Format (exact): {CLIFormatter.NEON_GREEN if format_exact > 0.5 else CLIFormatter.SUNSET_ORANGE if format_exact > 0 else CLIFormatter.HOT_PINK}{format_exact:.3f}{Style.RESET_ALL}")
        print(f"  {CLIFormatter.NEON_CYAN}Format (approx): {CLIFormatter.NEON_GREEN if format_approx > 0 else CLIFormatter.SUNSET_ORANGE if format_approx > -0.5 else CLIFormatter.HOT_PINK}{format_approx:.3f}{Style.RESET_ALL}")
        print(f"  {CLIFormatter.NEON_CYAN}Answer check: {CLIFormatter.NEON_GREEN if answer_check > 0.5 else CLIFormatter.SUNSET_ORANGE if answer_check > 0 else CLIFormatter.HOT_PINK}{answer_check:.3f}{Style.RESET_ALL}")
        print(f"  {CLIFormatter.NEON_CYAN}Number extract: {CLIFormatter.NEON_GREEN if number_extract > 0.5 else CLIFormatter.SUNSET_ORANGE if number_extract > 0 else CLIFormatter.HOT_PINK}{number_extract:.3f}{Style.RESET_ALL}")

        # Training metrics
        print(f"\n{CLIFormatter.NEON_PURPLE}Training Metrics:{Style.RESET_ALL}")

        learning_rate = metrics.get('learning_rate', 0)
        grad_norm = metrics.get('grad_norm', 0)
        kl = metrics.get('kl', 0)
        loss = metrics.get('loss', None)

        print(f"  {CLIFormatter.NEON_CYAN}Learning Rate: {CLIFormatter.ELECTRIC_BLUE}{learning_rate:.2e}{Style.RESET_ALL}")
        print(f"  {CLIFormatter.NEON_CYAN}Gradient Norm: {CLIFormatter.NEON_GREEN if 0.001 < grad_norm < 100 else CLIFormatter.HOT_PINK}{grad_norm:.2f}{Style.RESET_ALL}")
        print(f"  {CLIFormatter.NEON_CYAN}KL Divergence: {CLIFormatter.ELECTRIC_BLUE}{kl:.4f}{Style.RESET_ALL}")

        if loss is not None:
            print(f"  {CLIFormatter.NEON_CYAN}Loss: {CLIFormatter.NEON_GREEN if loss < 0.1 else CLIFormatter.SUNSET_ORANGE if loss < 1 else CLIFormatter.HOT_PINK}{loss:.4f}{Style.RESET_ALL}")

        # Completion stats
        avg_length = metrics.get('completions/mean_length', 0)
        min_length = metrics.get('completions/min_length', 0)
        max_length = metrics.get('completions/max_length', 0)
        clipped_ratio = metrics.get('completions/clipped_ratio', 0)

        if avg_length > 0:
            print(f"\n{CLIFormatter.NEON_PURPLE}Completion Stats:{Style.RESET_ALL}")
            print(f"  {CLIFormatter.NEON_CYAN}Avg Length: {CLIFormatter.ELECTRIC_BLUE}{int(avg_length)} tokens{Style.RESET_ALL}")

            if min_length > 0 or max_length > 0:
                print(f"  {CLIFormatter.NEON_CYAN}Range: {CLIFormatter.ELECTRIC_BLUE}{int(min_length)}-{int(max_length)} tokens{Style.RESET_ALL}")

            if clipped_ratio > 0:
                color = CLIFormatter.NEON_GREEN if clipped_ratio < 0.1 else CLIFormatter.SUNSET_ORANGE if clipped_ratio < 0.3 else CLIFormatter.HOT_PINK
                print(f"  {CLIFormatter.NEON_CYAN}Clipped: {color}{clipped_ratio:.1%}{Style.RESET_ALL}")

        print(f"\n{CLIFormatter.NEON_PINK}{'═' * 60}{Style.RESET_ALL}")

    @staticmethod
    def format_training_batch(metrics_list: List[dict]):
        """Format multiple training metric dictionaries as a batch

        Args:
            metrics_list: List of metric dictionaries
        """
        print(f"\n{CLIFormatter.NEON_PINK}{Style.BRIGHT}{'═' * 60}")
        print(f"{CLIFormatter.ELECTRIC_BLUE}Training Metrics Batch - {len(metrics_list)} Updates".center(60))
        print(f"{CLIFormatter.NEON_PINK}{'═' * 60}{Style.RESET_ALL}\n")

        for i, metrics in enumerate(metrics_list, 1):
            print(f"{CLIFormatter.NEON_PURPLE}{Style.BRIGHT}Update {i}/{len(metrics_list)}{Style.RESET_ALL}")
            CLIFormatter.format_training_metrics(metrics)
            if i < len(metrics_list):
                print()  # Add space between updates


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
        self.initial_text = text

    def __enter__(self):
        """Enter context manager"""
        self.update()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager"""
        if exc_type is None:
            self.complete("Complete")
        else:
            CLIFormatter.clear_line()
            CLIFormatter.print_error(f"{self.initial_text} - Failed")
        return False

    def update(self, text: str = None):
        """Update spinner animation with synthwave colors"""
        if text:
            self.text = text
        CLIFormatter.clear_line()
        frame = self.spinner[self.index % len(self.spinner)]
        # Cycle through synthwave colors for animation
        colors = [CLIFormatter.NEON_PINK, CLIFormatter.NEON_CYAN, CLIFormatter.ELECTRIC_BLUE, CLIFormatter.NEON_GREEN]
        color = colors[self.index % len(colors)]
        print(f"{color}{frame} {self.text}{Style.RESET_ALL}", end='', flush=True)
        self.index += 1

    def complete(self, message: str = "Complete"):
        """Complete spinner with message"""
        CLIFormatter.clear_line()
        CLIFormatter.print_success(f"{self.initial_text} - {message}")
