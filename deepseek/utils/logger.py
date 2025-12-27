"""
DeepSeek V3 Logger Module
========================

A unified logging module with colored output for different log levels.
Includes timestamp, process ID, thread ID, filename, and line number.

Usage:
    from logger import get_logger
    logger = get_logger(__name__)
    
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
"""

import logging
import sys
import os
import threading
from typing import Optional


# ANSI Color Codes
class Colors:
    """ANSI color codes for terminal output."""
    # Reset
    RESET = "\033[0m"
    
    # Regular colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    # Bold colors
    BOLD_BLACK = "\033[1;30m"
    BOLD_RED = "\033[1;31m"
    BOLD_GREEN = "\033[1;32m"
    BOLD_YELLOW = "\033[1;33m"
    BOLD_BLUE = "\033[1;34m"
    BOLD_MAGENTA = "\033[1;35m"
    BOLD_CYAN = "\033[1;36m"
    BOLD_WHITE = "\033[1;37m"
    
    # Background colors
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    
    # Dim
    DIM = "\033[2m"
    
    @classmethod
    def supports_color(cls) -> bool:
        """Check if the terminal supports color output."""
        # Check for explicit no-color flag
        if os.environ.get('NO_COLOR'):
            return False
        
        # Check for explicit force color
        if os.environ.get('FORCE_COLOR'):
            return True
        
        # Check if stdout is a tty
        if not hasattr(sys.stdout, 'isatty'):
            return False
        
        if not sys.stdout.isatty():
            return False
        
        # Check for known terminals that support color
        term = os.environ.get('TERM', '')
        return term != 'dumb'


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output for different log levels."""
    
    # Log level colors
    LEVEL_COLORS = {
        logging.DEBUG: Colors.CYAN,       # Cyan for debug
        logging.INFO: Colors.GREEN,       # Green for info
        logging.WARNING: Colors.YELLOW,   # Yellow for warning
        logging.ERROR: Colors.RED,        # Red for error
        logging.CRITICAL: Colors.BOLD_RED # Bold red for critical
    }
    
    # Log level symbols
    LEVEL_SYMBOLS = {
        logging.DEBUG: "🔍",
        logging.INFO: "ℹ️ ",
        logging.WARNING: "⚠️ ",
        logging.ERROR: "❌",
        logging.CRITICAL: "🔥"
    }
    
    def __init__(self, use_color: bool = True, show_symbols: bool = True):
        """
        Initialize the colored formatter.
        
        Args:
            use_color: Whether to use colored output
            show_symbols: Whether to show emoji symbols for log levels
        """
        self.use_color = use_color and Colors.supports_color()
        self.show_symbols = show_symbols
        
        # Base format string
        # Format: [TIME] [PID:TID] [LEVEL] [FILE:LINE] MESSAGE
        super().__init__()
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with colors."""
        # Get basic info
        timestamp = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        
        # Process and thread IDs
        pid = os.getpid()
        tid = threading.current_thread().ident
        
        # Get filename and line number
        filename = os.path.basename(record.pathname)
        lineno = record.lineno
        
        # Get level info
        level_name = record.levelname
        level_color = self.LEVEL_COLORS.get(record.levelno, Colors.WHITE)
        level_symbol = self.LEVEL_SYMBOLS.get(record.levelno, "")
        
        # Format message
        message = record.getMessage()
        
        # Handle exceptions
        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
            message = f"{message}\n{record.exc_text}"
        
        if self.use_color:
            # Colored format
            time_str = f"{Colors.DIM}{timestamp}{Colors.RESET}"
            pid_tid_str = f"{Colors.DIM}[{pid}:{tid}]{Colors.RESET}"
            level_str = f"{level_color}[{level_name:^8}]{Colors.RESET}"
            location_str = f"{Colors.MAGENTA}[{filename}:{lineno}]{Colors.RESET}"
            msg_str = f"{level_color}{message}{Colors.RESET}"
            
            if self.show_symbols:
                formatted = f"{time_str} {level_symbol} {location_str} {msg_str}"
            else:
                formatted = f"{time_str} {level_str} {location_str} {msg_str}"
        else:
            # Plain format (no color)
            if self.show_symbols:
                formatted = f"{timestamp} {level_symbol} [{filename}:{lineno}] {message}"
            else:
                formatted = f"{timestamp} [{level_name:^8}] [{filename}:{lineno}] {message}"
        
        return formatted


class DeepSeekLogger:
    """
    Logger wrapper for DeepSeek V3 project.
    
    Provides convenient methods for logging with colored output.
    """
    
    _loggers: dict = {}
    _default_level: int = logging.INFO
    _handler: Optional[logging.Handler] = None
    
    @classmethod
    def set_default_level(cls, level: int):
        """Set the default log level for all loggers."""
        cls._default_level = level
        # Update existing loggers
        for logger in cls._loggers.values():
            logger.setLevel(level)
    
    @classmethod
    def get_logger(
        cls,
        name: str = "deepseek_v3",
        level: Optional[int] = None,
        use_color: bool = True,
        show_symbols: bool = True
    ) -> logging.Logger:
        """
        Get a logger instance with colored output.
        
        Args:
            name: Logger name (typically __name__)
            level: Log level (default: INFO)
            use_color: Whether to use colored output
            show_symbols: Whether to show emoji symbols
        
        Returns:
            Configured logger instance
        """
        # Return existing logger if already created
        if name in cls._loggers:
            return cls._loggers[name]
        
        # Create new logger
        logger = logging.getLogger(name)
        logger.setLevel(level or cls._default_level)
        
        # Avoid adding duplicate handlers
        if not logger.handlers:
            # Create console handler
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(level or cls._default_level)
            
            # Create formatter and add to handler
            formatter = ColoredFormatter(use_color=use_color, show_symbols=show_symbols)
            handler.setFormatter(formatter)
            
            logger.addHandler(handler)
        
        # Prevent propagation to root logger
        logger.propagate = False
        
        cls._loggers[name] = logger
        return logger
    
    @classmethod
    def setup_file_handler(
        cls,
        filepath: str,
        level: Optional[int] = None,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5
    ):
        """
        Add a file handler to all loggers.
        
        Args:
            filepath: Path to log file
            level: Log level for file output
            max_bytes: Max file size before rotation
            backup_count: Number of backup files to keep
        """
        from logging.handlers import RotatingFileHandler
        
        # Create directory if needed
        log_dir = os.path.dirname(filepath)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Create file handler with rotation
        file_handler = RotatingFileHandler(
            filepath,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(level or cls._default_level)
        
        # Use plain formatter for file (no colors)
        file_formatter = ColoredFormatter(use_color=False, show_symbols=False)
        file_handler.setFormatter(file_formatter)
        
        # Add to all existing loggers
        for logger in cls._loggers.values():
            logger.addHandler(file_handler)
        
        cls._handler = file_handler


# Convenience function for getting a logger
def get_logger(
    name: str = "deepseek_v3",
    level: Optional[int] = None,
    use_color: bool = True,
    show_symbols: bool = True
) -> logging.Logger:
    """
    Get a logger instance with colored output.
    
    This is the main entry point for getting a logger in the project.
    
    Args:
        name: Logger name (typically __name__)
        level: Log level (default: INFO)
        use_color: Whether to use colored output
        show_symbols: Whether to show emoji symbols
    
    Returns:
        Configured logger instance
    
    Example:
        >>> from logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Training started")
        >>> logger.debug("Batch size: 32")
        >>> logger.warning("Learning rate might be too high")
        >>> logger.error("Failed to load checkpoint")
    """
    return DeepSeekLogger.get_logger(name, level, use_color, show_symbols)


def set_log_level(level: int):
    """
    Set the log level for all loggers.
    
    Args:
        level: Log level (logging.DEBUG, logging.INFO, etc.)
    
    Example:
        >>> from logger import set_log_level
        >>> import logging
        >>> set_log_level(logging.DEBUG)
    """
    DeepSeekLogger.set_default_level(level)


def setup_file_logging(
    filepath: str = "logs/deepseek_v3.log",
    level: Optional[int] = None
):
    """
    Enable file logging for the project.
    
    Args:
        filepath: Path to log file
        level: Log level for file output
    
    Example:
        >>> from logger import setup_file_logging
        >>> setup_file_logging("logs/training.log")
    """
    DeepSeekLogger.setup_file_handler(filepath, level)


# Pre-defined loggers for different modules
def get_training_logger() -> logging.Logger:
    """Get logger for training module."""
    return get_logger("deepseek_v3.training")


def get_model_logger() -> logging.Logger:
    """Get logger for model module."""
    return get_logger("deepseek_v3.model")


def get_data_logger() -> logging.Logger:
    """Get logger for data module."""
    return get_logger("deepseek_v3.data")


def get_inference_logger() -> logging.Logger:
    """Get logger for inference module."""
    return get_logger("deepseek_v3.inference")


# Demonstration code
if __name__ == "__main__":
    # Demo the logger
    print("=" * 70)
    print("DeepSeek V3 Logger Demo")
    print("=" * 70)
    
    # Create logger
    logger = get_logger("demo", level=logging.DEBUG)
    
    # Test different log levels
    logger.debug("This is a DEBUG message - for detailed debugging info")
    logger.info("This is an INFO message - for general information")
    logger.warning("This is a WARNING message - for potential issues")
    logger.error("This is an ERROR message - for errors that occurred")
    
    # Test with exception
    print("\n--- Testing exception logging ---")
    try:
        raise ValueError("This is a test exception")
    except ValueError:
        logger.exception("Caught an exception")
    
    # Test without symbols
    print("\n--- Testing without emoji symbols ---")
    logger_no_symbols = get_logger("demo_no_symbols", level=logging.DEBUG, show_symbols=False)
    logger_no_symbols.info("Info message without emoji symbols")
    logger_no_symbols.error("Error message without emoji symbols")
    
    # Test without color
    print("\n--- Testing without color ---")
    logger_no_color = get_logger("demo_no_color", level=logging.DEBUG, use_color=False)
    logger_no_color.info("Info message without color")
    logger_no_color.warning("Warning message without color")
    
    print("\n" + "=" * 70)
    print("Logger demo complete!")
    print("=" * 70)
