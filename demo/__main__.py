"""Launch the demo:  python -m demo [--preset quick|full|smoke] [--port 8000] [--server auto|fastapi|stdlib]"""
import argparse
import threading
import webbrowser

from aircraft_heat.config import PRESETS


def main():
    ap = argparse.ArgumentParser(prog="python -m demo", description=__doc__)
    ap.add_argument("--preset", default="quick", choices=sorted(PRESETS),
                    help="training budget (smoke = seconds/sanity check, quick = ~15 min once, full = longer, more accurate)")
    ap.add_argument("--retrain", action="store_true", help="ignore any saved checkpoint and train again")
    ap.add_argument("--ckpt-dir", default="checkpoints", help="where trained weights / FEM data are cached")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--server", default="auto", choices=["auto", "fastapi", "stdlib"])
    ap.add_argument("--no-adapt", action="store_true", help="skip the start-up error-driven fine-tuning")
    ap.add_argument("--adapt-target", type=float, default=10.0, help="stop adapting when mean MAE on fresh scenarios <= this (K)")
    ap.add_argument("--adapt-worst", type=float, default=2.0, help="... and the worst scenario <= this factor x target")
    ap.add_argument("--adapt-budget", type=float, default=900.0, help="max seconds spent adapting (start-up loading time)")
    ap.add_argument("--adapt-probes", type=int, default=10, help="fresh FEM scenarios measured per round")
    ap.add_argument("--adapt-n", type=int, default=3, help="N: samples for a 10 K case (a 20 K case gets 2N, ...)")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    adapt = None if args.no_adapt else dict(target=args.adapt_target, worst_factor=args.adapt_worst,
                                            budget_s=args.adapt_budget, n_probe=args.adapt_probes, N=args.adapt_n)
    url = f"http://{args.host}:{args.port}"
    use_fastapi = args.server == "fastapi"
    if args.server == "auto":
        try:
            import fastapi, uvicorn  # noqa: F401
            use_fastapi = True
        except ImportError:
            use_fastapi = False
    print(f"\n  Demo running at {url}  (preset: {args.preset}, server: {'fastapi' if use_fastapi else 'stdlib'})\n")
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    if use_fastapi:
        import uvicorn
        from demo.server import create_app
        uvicorn.run(create_app(args.preset, args.ckpt_dir, args.retrain, adapt), host=args.host, port=args.port, log_level="warning")
    else:
        from demo.core import DemoApp
        from demo.stdlib_server import serve
        serve(DemoApp(args.preset, args.ckpt_dir, args.retrain, adapt), args.host, args.port)


if __name__ == "__main__":
    main()
