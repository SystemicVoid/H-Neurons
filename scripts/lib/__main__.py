"""Allow `uv run python -m scripts.lib` invocation (delegates to pipeline CLI)."""

from scripts.lib.pipeline import main

raise SystemExit(main())
