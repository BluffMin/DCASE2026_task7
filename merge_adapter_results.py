import argparse
from pathlib import Path
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirs", type=str, nargs="+", required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dfs = []
    for d in args.dirs:
        path = Path(d) / "summary_table.csv"
        if not path.exists():
            print(f"[skip] missing: {path}")
            continue
        df = pd.read_csv(path)
        df["source_dir"] = str(Path(d))
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.sort_values("D3_after_D3", ascending=False)

    merged.to_csv(out_dir / "summary_table.csv", index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("[Best by D3_after_D3]")
    print(merged.head(10).to_string(index=False))

    print("\n[Best by Avg_after_D3]")
    print(merged.sort_values("Avg_after_D3", ascending=False).head(10).to_string(index=False))
    print("=" * 80)

if __name__ == "__main__":
    main()