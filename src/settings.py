import logging
import rich.logging
from rich.console import Console

# This is the root logger that will be exported and used by other modules.
logger = logging.getLogger("<project_src>")


def setup_logging(level: int = logging.INFO) -> None:

	"""
	Configure logging for the application with Rich formatting.
	This function configures the root logger. The module-level 'logger'
	will propagate its messages to the configured root logger.

	Args:
		level: Logging level (default: logging.INFO)
	"""

	logging.captureWarnings(True) # Redirect warnings to the logging system


	# Clear any existing handlers from the root logger
	root_logger = logging.getLogger() # Get the root logger
	for handler in root_logger.handlers[:]:
		root_logger.removeHandler(handler)

	# Configure rich handler
	rich_handler = rich.logging.RichHandler(
		rich_tracebacks  = True,
		show_time        = True,
		show_level       = True,
		show_path        = True,
		enable_link_path = False, # Do not create explicit hyperlinks for paths
		markup           = True,  # Enable Rich markup in log messages
		console          = Console() # Use default console settings
	)

	# Set up basic config for the root logger
	logging.basicConfig(
		level    = level,
		format   = "%(message)s",  # RichHandler handles its own formatting
		datefmt  = "[%X]",         # Standard date format for non-Rich parts
		handlers = [rich_handler],
		force    = True  # Ensures this configuration takes precedence
	)

	# Set log level for specific noisy loggers
	logging.getLogger('h5py').setLevel(logging.WARNING)
	logging.getLogger('matplotlib').setLevel(logging.WARNING)
	logging.getLogger('PIL').setLevel(logging.WARNING)
	logging.getLogger('tensorflow').setLevel(logging.WARNING)
	logging.getLogger('sklearn').setLevel(logging.WARNING)
	logging.getLogger('ucimlrepo').setLevel(logging.WARNING)
	logging.getLogger('streamlit').setLevel(logging.WARNING)
	logging.getLogger('crowdkit').setLevel(logging.WARNING)

