"""
Logging configuration utilities for the Pokémon RL project.
"""

import logging
from datetime import datetime
from pathlib import Path

try:
    import colorlog
except ImportError:
    colorlog = None


def setup_logging():
    """Setup logging with colors and file output."""
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # Create timestamp for log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"pokemon_rl_{timestamp}.log"

    # Set up formatters
    file_formatter = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] %(message)s"
    )

    # Console formatter with colors if colorlog is available
    if colorlog:
        console_formatter = colorlog.ColoredFormatter(
            "%(log_color)s[%(asctime)s][%(name)s][%(levelname)s]%(reset)s %(message)s",
            datefmt="%H:%M:%S",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        )
    else:
        console_formatter = logging.Formatter(
            "[%(asctime)s][%(name)s][%(levelname)s] %(message)s", datefmt="%H:%M:%S"
        )

    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Configure PokeEnv loggers to be less verbose
    poke_loggers = [
        "poke_env",
        "poke_env.player",
        "poke_env.environment",
        "poke_env.player.player",
        "poke_env.player.singles_env",
        "websockets",
        "websockets.client",
        "websockets.protocol",
    ]

    for logger_name in poke_loggers:
        poke_logger = logging.getLogger(logger_name)
        poke_logger.setLevel(logging.WARNING)  # Only show warnings and errors

    # Also suppress any logger that starts with "PokeEnv"
    for name in logging.Logger.manager.loggerDict:
        if name.startswith("PokeEnv") or "PokeEnv" in name:
            logging.getLogger(name).setLevel(logging.WARNING)

    return logging.getLogger("Main")


def configure_poke_env_logging():
    """Configure PokeEnv logging to be less verbose during training."""

    class WebSocketErrorFilter(logging.Filter):
        """Filter out websocket connection errors that happen during server shutdown."""

        def filter(self, record):
            # Suppress common websocket connection errors during shutdown
            error_messages = [
                "no close frame received or sent",
                "ConnectionClosedError",
                "connection closed",
                "websocket connection closed",
            ]
            return not any(
                msg in str(record.getMessage()).lower() for msg in error_messages
            )

    # Configure known PokeEnv loggers to be very quiet
    poke_loggers = [
        "poke_env",
        "poke_env.player",
        "poke_env.environment",
        "poke_env.player.player",
        "poke_env.player.singles_env",
        "websockets",
        "websockets.client",
        "websockets.protocol",
        "websockets.asyncio",
        "websockets.exceptions",
    ]

    websocket_filter = WebSocketErrorFilter()

    for logger_name in poke_loggers:
        poke_logger = logging.getLogger(logger_name)
        poke_logger.setLevel(logging.CRITICAL)  # Only show critical errors
        poke_logger.addFilter(websocket_filter)  # Add websocket error filter

    # Also configure any loggers that start with known patterns
    for name in list(logging.Logger.manager.loggerDict.keys()):
        if any(
            pattern in name
            for pattern in ["RandomPlayer", "PokeEnvSinglesWr", "Player"]
        ):
            logger = logging.getLogger(name)
            logger.setLevel(logging.CRITICAL)  # Only critical errors
            logger.addFilter(websocket_filter)  # Add websocket error filter
