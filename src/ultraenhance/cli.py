import argparse
import json
from scripts.run_eval import run_eval
from scripts.run_grid_search import run_grid_search

def main():
    parser = argparse.ArgumentParser("ultraenhance")
    parser.add_argument("--config", default="configs/default.json", help="path to config json")
    parser.add_argument("--mode", choices=["eval", "grid"], default="grid")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if args.mode == "eval":
        run_eval(cfg)
    else:
        run_grid_search(cfg)

if __name__ == "__main__":
    main()
