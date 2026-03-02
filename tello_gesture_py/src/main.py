import argparse
from .config import ControllerConfig
from .controller import Controller

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="Optional joblib model for gesture classification.")
    ap.add_argument("--labels", default=None, help="labels.json mapping numeric id -> name.")
    ap.add_argument("--log", default=None, help="Telemetry CSV output path.")
    args = ap.parse_args()

    cfg = ControllerConfig()
    if args.log:
        cfg.log_path = args.log

    ctrl = Controller(cfg, model_path=args.model, labels_path=args.labels)
    raise SystemExit(ctrl.run())

if __name__ == "__main__":
    main()
