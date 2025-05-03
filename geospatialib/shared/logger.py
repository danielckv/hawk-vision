import logging
import os
from pathlib import Path

import structlog


def log_instance():
    return Logger().logger_client


class Logger:

    _instance = None
    logger_client = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Logger, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        root_dir = os.path.abspath(os.curdir)

        if Path(root_dir + "/_log_lines").with_suffix(".log").exists():
            os.remove(Path(root_dir + "/_log_lines").with_suffix(".log"))

        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,  # Add filtering by level
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),  # Timestamp format
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,  # Prepare for stdlib formatter
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),  # Use stdlib logger factory
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        self.logger_client = structlog.get_logger()
        self.logger_client.setLevel(logging.DEBUG)

        file_handler = logging.FileHandler(
            Path(root_dir + "/_log_lines").with_suffix(".log")
        )
        file_handler.setFormatter(structlog.stdlib.ProcessorFormatter(
            processor=structlog.processors.JSONRenderer(),
            foreign_pre_chain=structlog.contextvars.merge_contextvars,
        ))

        stdout_handler = logging.StreamHandler()
        self.logger_client.addHandler(file_handler)
        self.logger_client.addHandler(stdout_handler)
