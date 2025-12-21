from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, Optional


class HydraLogger:
    def __init__(self, config: Any, rank: int = 0) -> None:
        self.config = config
        self.rank = rank
        self.enabled = rank == 0

        self._logger = logging.getLogger("HYDRA")
        self._logger.propagate = False
        self._logger.setLevel(logging.INFO)

        self._file_handler: Optional[logging.Handler] = None

        if not self.enabled:
            return

        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        if self._logger.handlers:
            self._logger.handlers.clear()

        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        self._logger.addHandler(console)

        log_dir = getattr(config, "log_dir", "logs")
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, "training.log"))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self._logger.addHandler(file_handler)
        self._file_handler = file_handler

    def info(self, msg: str, *args: object, **kwargs: object) -> None:
        if self.enabled:
            self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args: object, **kwargs: object) -> None:
        if self.enabled:
            self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args: object, **kwargs: object) -> None:
        if self.enabled:
            self._logger.error(msg, *args, **kwargs)

    def debug(self, msg: str, *args: object, **kwargs: object) -> None:
        if self.enabled:
            self._logger.debug(msg, *args, **kwargs)

    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = "") -> None:
        if not self.enabled:
            return
        for key, value in metrics.items():
            self._logger.info("%s%s=%s step=%s", prefix, key, value, step)

    def close(self) -> None:
        if self._file_handler is not None:
            self._file_handler.close()
