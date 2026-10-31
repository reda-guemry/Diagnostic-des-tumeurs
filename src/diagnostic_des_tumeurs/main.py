"""Top-level entry point: run one workflow or all of them."""

from __future__ import annotations

import json
import sys
from typing import Any

from diagnostic_des_tumeurs.config.hyperparameters import InceptionConfig, YoloConfig
from diagnostic_des_tumeurs.helpers.io import json_safe


def run_all(
    root: str | None = None,
    inception_config: InceptionConfig | None = None,
    yolo_config: YoloConfig | None = None,
    register: bool = True,
    tracking: bool = True,
) -> dict[str, Any]:
    """Run the Inception workflow, then the YOLO workflow, in one process."""
    from diagnostic_des_tumeurs.inception.main import run as run_inception
    from diagnostic_des_tumeurs.yolo.main import run as run_yolo

    inception_summary = run_inception(
        root=root,
        config=inception_config,
        register=register,
        tracking=tracking,
    )
    yolo_summary = run_yolo(
        root=root,
        config=yolo_config,
        register=register,
        tracking=tracking,
    )
    return {
        "inception": inception_summary,
        "yolo": yolo_summary,
    }


def _print_all_summary(summary: dict[str, Any]) -> None:
    payload = {
        "inception": summary["inception"].to_dict(),
        "yolo": summary["yolo"].to_dict(),
    }
    print(json.dumps(json_safe(payload), indent=2))


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if argv and argv[0] in {"-h", "--help"}:
        _print_help()
        return 0

    workflow = "all"
    if argv and argv[0] in {"all", "inception", "yolo"}:
        workflow = argv.pop(0)

    if workflow == "inception":
        from diagnostic_des_tumeurs.inception.main import main as inception_main

        return inception_main(argv)

    if workflow == "yolo":
        from diagnostic_des_tumeurs.yolo.main import main as yolo_main

        return yolo_main(argv)

    if argv:
        print(
            "The 'all' workflow does not accept hyperparameter flags; "
            "run the workflow separately to override them.",
            file=sys.stderr,
        )
        return 2

    summary = run_all()
    _print_all_summary(summary)
    return 0


def _print_help() -> None:
    print(
        "Usage: diagnostic-des-tumeurs [all|inception|yolo] [options]\n"
        "\n"
        "  all        Run the Inception and YOLO workflows (default).\n"
        "  inception  Run the InceptionV3 classification workflow.\n"
        "  yolo       Run the YOLOv8 detection workflow.\n"
        "\n"
        "Run `diagnostic-des-tumeurs inception --help` or\n"
        "`diagnostic-des-tumeurs yolo --help` for the few hyperparameters."
    )


if __name__ == "__main__":
    raise SystemExit(main())
