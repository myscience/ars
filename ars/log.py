import logging
import logging.config

# ANSI escape codes for colors
RESET = "\033[0m"
RED     = "\033[91m"
BLUE    = "\033[94m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
MAGENTA = "\033[95m"

class FancyFormatter(logging.Formatter):
    LOG_COLORS = {
        logging.DEBUG: BLUE,
        logging.INFO: GREEN,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: MAGENTA
    }

    EMOJIS = {
        logging.DEBUG   : "🐛",      # Debug
        logging.INFO    : "ℹ️",      # Info
        logging.WARNING : "⚠️",       # Warning
        logging.ERROR   : "❌",      # Error
        logging.CRITICAL: "🔥",      # Critical
    }

    def format(self, record):
        log_color = self.LOG_COLORS.get(record.levelno, RESET)
        emoji = self.EMOJIS.get(record.levelno, "")
        
        record.message = record.getMessage()
        if self.usesTime():
            record.asctime = self.formatTime(record, self.datefmt)
            record.asctime = f'[{record.asctime}]'
        msg = self.formatMessage(record)
        if record.exc_info:
            # Cache the traceback text to avoid converting it multiple times
            # (it's constant anyway)
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            if msg[-1:] != "\n":
                msg = msg + "\n"
            msg = msg + record.exc_text
        if record.stack_info:
            if msg[-1:] != "\n":
                msg = msg + "\n"
            msg = msg + self.formatStack(record.stack_info)
        
        return f"{log_color}{emoji} {msg}{RESET}"

def setup_logging():
    LOGGING_CONFIG = {
        'version': 1,
        'disable_existing_loggers': True,
        'formatters': {
            'colored_emoji': {  # Reference to the custom formatter
                '()': FancyFormatter,  # The '()' syntax specifies a custom formatter class
                'format': "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
            'standard': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'colored_emoji',  # Use colored emoji formatter for console
            },
            'file': {
                'class': 'logging.FileHandler',
                'filename': 'ars.log',
                'formatter': 'standard',  # You can use the standard format for files
            },
        },
        'loggers': {
            '': {  # Root logger
                'level': 'DEBUG',
                'handlers': ['console', 'file'],
                'propagate': False,
            },
            'ars.utils': {  # Logger for ars.utils
                'level': 'DEBUG',
                'handlers': ['console', 'file'],
                'propagate': False,
            },
            'ars.sprinkle': {  # Logger for ars.sprinkle
                'level': 'DEBUG',
                'handlers': ['console', 'file'],
                'propagate': False,
            },
            'ars.geometry': {  # Logger for ars.geometry
                'level': 'DEBUG',
                'handlers': ['console', 'file'],
                'propagate': False,
            },
            'ars.core': {  # Logger for ars.core
                'level': 'DEBUG',
                'handlers': ['console', 'file'],
                'propagate': False,
            },
        }
    }
    
    logging.config.dictConfig(LOGGING_CONFIG)