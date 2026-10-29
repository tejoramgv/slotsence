import math
import heapq
from pathlib import Path
import pandas as pd
import numpy as np

# We can import run_phase6 or test the route simulator
from src.simulation.pipeline import run_phase6
from src.models.config import Config

project_root = Path(__file__).resolve().parent.parent
processed = project_root / "dataset" / "processed"
print("Running test...")
