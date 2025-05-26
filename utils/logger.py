# utils/loggers.py
# -*- encoding: utf-8 -*-

import logging
import os
import logging.handlers


class Logger:
    def __init__(self):
        self.logger = logging.getLogger('snsbvae')
        self.timestamp = None
        self.is_configured = False

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)

        return cls._instance

    def configure(self, timestamp: str, log_level=logging.INFO):
        self.is_configured = True
        current_dir = os.getcwd()
        log_dir = os.path.join(current_dir, 'log')
        self.timestamp = timestamp

        if not os.path.isdir(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        log_file_path = os.path.join(log_dir, f'{timestamp}.log')
        debug_log_path = os.path.join(log_dir, f'{timestamp}_debug.log')

        self.logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')

        # clears handlers
        self.logger.handlers.clear()

        # console print handler
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(log_level)
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

        # log file handler
        timed_file_handler = logging.handlers.TimedRotatingFileHandler(
            filename=log_file_path,
            when='midnight',
            encoding='utf-8'
        )
        timed_file_handler.setLevel(log_level)
        timed_file_handler.setFormatter(formatter)
        timed_file_handler.suffix = "%Y%m%d"
        self.logger.addHandler(timed_file_handler)

        # debug log file handler
        class DebugFilter(logging.Filter):
            def filter(self, record):
                return record.levelno == logging.DEBUG

        debug_file_handler = logging.FileHandler(debug_log_path, encoding='utf-8')
        debug_file_handler.setLevel(logging.DEBUG)
        debug_file_handler.setFormatter(formatter)
        debug_file_handler.addFilter(DebugFilter())
        self.logger.addHandler(debug_file_handler)

    def get_logger(self):
        return self.logger