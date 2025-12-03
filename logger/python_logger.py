import logging
import structlog
import inspect
import os

class KeyRenameProcessor:
    """
    Safe field renaming for structlog.
    DO NOT rename 'event' to 'message'.
    """

    def __init__(self, mapping: dict):
        self.mapping = mapping

    def __call__(self, logger, method_name, event_dict):
        for old_key, new_key in self.mapping.items():
            if old_key in event_dict:
                event_dict[new_key] = event_dict.pop(old_key)
        return event_dict

class LoggerConfig:

    @staticmethod
    def processors():
        return [
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
            structlog.processors.add_log_level,
            structlog.contextvars.merge_contextvars,
            KeyRenameProcessor({
                "timestamp": "asctime",
                "level": "levelname",
                "logger": "logger_name",
            }),
            structlog.processors.JSONRenderer(),
        ]


class ABCLogger:

    def __init__(self, name="app"):
        self.name = name
        self._setup_library_logging()

        structlog.configure(
            processors=LoggerConfig.processors(),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=False,
        )

        self.logger = structlog.get_logger()

    def _setup_library_logging(self):
        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)

    # allow logger.info("text", key=value)
    def __getattr__(self, level):
        if level in ["info", "error", "warning", "debug", "critical"]:
            def wrapper(message, **kwargs):
                return self._log(level, message, **kwargs)
            return wrapper
        raise AttributeError(level)

    def _log(self, level, message, **extra):
        caller = self._get_caller()

        # structlog expects 'event' key for the main message
        log_data = {
            "event": message,               # main message
            "file": caller["filename"],
            "line": caller["lineno"],
            "function": caller["function"],
            **extra
        }

        getattr(self.logger, level)(**log_data)

    def _get_caller(self):
        frame = inspect.currentframe().f_back.f_back
        return {
            "filename": os.path.basename(frame.f_code.co_filename),
            "lineno": frame.f_lineno,
            "function": frame.f_code.co_name,
        }
