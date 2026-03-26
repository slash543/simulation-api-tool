"""Tests for src/utils/logging_utils.py"""
import logging

from src.utils.logging_utils import get_logger


class TestGetLogger:
    def test_returns_logger(self):
        log = get_logger("test.module")
        assert isinstance(log, logging.Logger)

    def test_correct_name(self):
        log = get_logger("my.logger")
        assert log.name == "my.logger"

    def test_default_level_is_info(self):
        log = get_logger("info.test")
        assert log.level == logging.INFO

    def test_custom_level(self):
        log = get_logger("debug.test", level=logging.DEBUG)
        assert log.level == logging.DEBUG

    def test_no_duplicate_handlers(self):
        log = get_logger("dup.test")
        count_before = len(log.handlers)
        get_logger("dup.test")  # call again
        assert len(log.handlers) == count_before
