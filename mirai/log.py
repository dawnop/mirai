import logging

logger = logging.getLogger("mirai")

# Torch compile subsystems that produce noisy E/W log lines and debug traces.
_TORCH_NOISY_LOGGERS = [
    "torch._inductor",
    "torch._dynamo",
    "torch._functorch",
]


def _setup_torch_logging(level):
    """Configure torch.compile log verbosity.

    INFO level (default):
      - AUTOTUNE benchmark tables are **kept** so the user sees compilation
        progress during the long Stage 1.
      - E/W log lines from torch._inductor (shared-memory warnings etc.)
        are suppressed (level -> CRITICAL).
      - UserWarnings from torch internals are suppressed.

    DEBUG level:
      - Everything is shown.
    """
    # AUTOTUNE progress tables (sys.stderr.write) are always useful.
    import torch._inductor.select_algorithm as _sa

    _sa.PRINT_AUTOTUNE = True

    if level <= logging.DEBUG:
        for name in _TORCH_NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.DEBUG)
    else:
        for name in _TORCH_NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.CRITICAL)
        import warnings

        warnings.filterwarnings("ignore", module=r"torch\.")


def setup_default_logging(level=logging.INFO):
    """Configure default console output for direct-run scenarios.

    Library consumers who ``import mirai`` can attach their own handlers;
    this helper is only called from CLI / programmatic entry points.

    Args:
        level: Logging level for the mirai logger (default: INFO).
               When set to DEBUG, all torch.compile logs are also shown.
    """
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[MIRAI %(levelname)s] %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)

    _setup_torch_logging(level)
