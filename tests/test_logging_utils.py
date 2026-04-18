"""Tests for the logging helpers."""

from __future__ import annotations

import logging

import pytest

from depthcast.logging_utils import configure_logging, get_logger


def test_get_logger_namespaces_under_root() -> None:
    logger = get_logger("depthcast.data.fi2010")
    assert logger.name == "depthcast.data.fi2010"


def test_get_logger_reroots_foreign_names() -> None:
    logger = get_logger("__main__")
    assert logger.name == "depthcast.__main__"


def test_configure_logging_attaches_single_handler() -> None:
    configure_logging(logging.DEBUG)
    configure_logging(logging.INFO)
    root = logging.getLogger("depthcast")
    assert len(root.handlers) == 1
    assert root.level == logging.INFO


def test_configure_logging_accepts_level_names() -> None:
    configure_logging("WARNING")
    assert logging.getLogger("depthcast").level == logging.WARNING


def test_configure_logging_rejects_unknown_level() -> None:
    with pytest.raises(ValueError, match="Unknown logging level"):
        configure_logging("CHATTY")
