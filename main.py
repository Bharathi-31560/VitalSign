"""Convenience launcher.

    python main.py            -> start the web application (same as: python app/app.py)
    python main.py --train    -> re-run the full pipeline (EDA, preprocessing, models) in the terminal
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

if __name__ == "__main__":
    if "--train" in sys.argv:
        from src.pipeline import run_pipeline
        run_pipeline()
    else:
        import runpy
        runpy.run_path(str(Path(__file__).resolve().parent / "app" / "app.py"), run_name="__main__")
