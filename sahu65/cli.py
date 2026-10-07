"""sahu65 command line interface.

    sahu65 detect photo.jpg            # local inference, bundled model
    sahu65 serve --port 8000           # run the HTTP API
    sahu65 info                        # bundled checkpoint + threshold state
"""
import argparse
import json
import sys


def _cmd_detect(args) -> int:
    from .local import detect

    result = detect(args.image)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    print(f"verdict          : {result.verdict}")
    print(f"ai_probability   : {result.ai_probability}")
    print(f"confidence       : {result.confidence}")
    print(f"classifier score : {result.classifier_probability}")
    print(f"c2pa             : present={result.c2pa_present} ai_declared={result.c2pa_ai_declared}")
    print(f"calibrated       : {result.calibrated}")
    print(f"latency_ms       : {result.latency_ms}")
    if result.warnings:
        print(f"warnings         : {', '.join(result.warnings)}")
    return 0 if result.verdict != "inconclusive" else 2


def _cmd_serve(args) -> int:
    import uvicorn

    uvicorn.run(
        "sahu65.main:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
    )
    return 0


def _cmd_info(args) -> int:
    from .config import get_settings, is_calibrated, load_fusion, load_thresholds
    from .local import model_info

    settings = get_settings()
    print(json.dumps(
        {
            "model": model_info(),
            "thresholds": load_thresholds(settings.config_dir),
            "fusion": load_fusion(settings.config_dir),
            "calibrated": is_calibrated(settings.config_dir),
            "document_gate": settings.document_gate,
        },
        indent=2,
        default=str,
    ))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sahu65", description="AI-generated image detection")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("detect", help="detect on a local image using the bundled model")
    d.add_argument("image", help="path to a JPEG, PNG or WebP file")
    d.add_argument("--json", action="store_true", help="emit JSON instead of text")
    d.set_defaults(func=_cmd_detect)

    s = sub.add_parser("serve", help="run the HTTP API server")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--log-level", default="info")
    s.set_defaults(func=_cmd_serve)

    i = sub.add_parser("info", help="print bundled checkpoint and threshold state")
    i.set_defaults(func=_cmd_info)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())