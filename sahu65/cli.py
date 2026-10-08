"""sahu65 command line interface.

    sahu65 detect photo.jpg            # local inference, bundled model
    sahu65 serve --port 8000           # run the HTTP API
    sahu65 info                        # bundled checkpoint + threshold state
    sahu65 --key myapp                 # mint a Bear Token (API key), printed once
    sahu65 --key-myapp                 # sugar for --key myapp
    sahu65 --list-keys                 # names of minted tokens, never the tokens
"""
import argparse
import json
import sys

_KEY_SUGAR_PREFIX = "--key-"


def _cmd_detect(args) -> int:
    from .local import detect

    result = detect(args.image, explain_result=args.explain, explain_provider=args.explain_provider)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    print(f"verdict          : {result.verdict}")
    print(f"ai_probability   : {result.ai_probability}")
    print(f"confidence       : {result.confidence}")
    print(f"detector score   : {result.detector_probability}")
    print(f"c2pa             : present={result.c2pa_present} ai_declared={result.c2pa_ai_declared}")
    print(f"calibrated       : {result.calibrated}")
    print(f"latency_ms       : {result.latency_ms}")
    if result.warnings:
        print(f"warnings         : {', '.join(result.warnings)}")
    if args.explain:
        if result.explanation:
            print()
            print(f"explanation ({result.explanation_provider}):")
            print(f"  {result.explanation}")
        else:
            print()
            print("explanation      : unavailable (no provider key set, or the provider failed)")
            print("                   set GEMINI_API_KEY or GROQ_API_KEY to enable it")
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
    from .services.explain import configured as explain_configured

    settings = get_settings()
    print(json.dumps(
        {
            "model": model_info(),
            "thresholds": load_thresholds(settings.config_dir),
            "fusion": load_fusion(settings.config_dir),
            "calibrated": is_calibrated(settings.config_dir),
            "document_gate": settings.document_gate,
            "explanation": explain_configured(),
        },
        indent=2,
        default=str,
    ))
    return 0


def _cmd_key(args) -> int:
    from .keys import create_key, keyfile_path

    token, rotated = create_key(args.key)
    print(f"Bear Token minted: {args.key}")
    print(f"  token : {token}")
    print(f"  store : {keyfile_path()} (SHA-256 digest only)")
    if rotated:
        print("  NOTE   : a token with this name already existed - it has been rotated")
        print("           and the previous token no longer works.")
    print()
    print("This token is shown ONCE. Use it as either request header:")
    print(f'  Authorization: Bearer {token}')
    print(f'  X-API-Key: {token}')
    print()
    print("In code:")
    print('  from sahu65 import Client')
    print(f'  client = Client(api_key="{token}", base_url="http://localhost:8000")')
    print("  export SAHU65_API_KEY=<token>   # picks it up automatically")
    return 0


def _cmd_list_keys(args) -> int:
    from .keys import keyfile_path, list_keys

    keys = list_keys()
    print(f"key store: {keyfile_path()}")
    if not keys:
        print("no Bear Tokens minted yet - run `sahu65 --key <name>`")
        return 0
    for record in keys:
        created = record["created"] or "unknown"
        print(f"  {record['name']}  (minted {created})")
    print(f"{len(keys)} token(s). Tokens themselves are never printed here.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sahu65", description="AI-generated image detection")
    p.add_argument(
        "--key",
        metavar="NAME",
        help="mint a Bear Token (API key) named NAME for POST /deep-guard/detect and exit; "
        "also accepted as --key-NAME",
    )
    p.add_argument(
        "--list-keys",
        action="store_true",
        help="list minted Bear Token names (never the tokens) and exit",
    )
    # Not required: `sahu65 --key myapp` / `sahu65 --list-keys` are flag-only invocations.
    sub = p.add_subparsers(dest="command")

    d = sub.add_parser("detect", help="detect on a local image using the bundled model")
    d.add_argument("image", help="path to a JPEG, PNG or WebP file")
    d.add_argument("--json", action="store_true", help="emit JSON instead of text")
    d.add_argument(
        "--explain",
        action="store_true",
        help="attach a plain-language explanation (needs GEMINI_API_KEY or GROQ_API_KEY)",
    )
    d.add_argument(
        "--explain-provider",
        choices=["gemini", "groq"],
        default=None,
        help="force a narration provider instead of auto-detecting from the environment",
    )
    d.set_defaults(func=_cmd_detect)

    s = sub.add_parser("serve", help="run the HTTP API server")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--log-level", default="info")
    s.set_defaults(func=_cmd_serve)

    i = sub.add_parser("info", help="print bundled checkpoint and threshold state")
    i.set_defaults(func=_cmd_info)

    return p


def _expand_key_sugar(argv):
    """Rewrite ``--key-NAME`` into ``--key NAME``.

    Lets `sahu65 --key-myapp` work as sugar for `sahu65 --key myapp`. A bare
    ``--key`` (or ``--key=NAME``) is left for argparse to handle normally.
    """
    out = []
    for arg in argv:
        if arg.startswith(_KEY_SUGAR_PREFIX) and len(arg) > len(_KEY_SUGAR_PREFIX):
            out.extend(["--key", arg[len(_KEY_SUGAR_PREFIX):]])
        else:
            out.append(arg)
    return out


def main(argv=None) -> int:
    argv = list(sys.argv[1:]) if argv is None else list(argv)
    parser = build_parser()
    args = parser.parse_args(_expand_key_sugar(argv))
    try:
        if getattr(args, "key", None):
            return _cmd_key(args)
        if getattr(args, "list_keys", False):
            return _cmd_list_keys(args)
        if not getattr(args, "func", None):
            parser.print_help()
            return 2
        return args.func(args)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())