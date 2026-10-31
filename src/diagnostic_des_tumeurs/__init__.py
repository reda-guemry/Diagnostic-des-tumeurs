"""Brain-tumour diagnosis pipelines: InceptionV3 classification and YOLOv8 detection."""

__version__ = "0.1.0"


def main() -> None:
    from diagnostic_des_tumeurs.main import main as cli_main

    raise SystemExit(cli_main())


__all__ = ["main"]
