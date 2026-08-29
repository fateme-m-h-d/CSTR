"""Configuration copied from PL-KKT-hPINN/2D/src/config.py.

These values are kept here only so the standalone 2D data generator reproduces
PL-KKT-hPINN's 2D sampling protocol. ENFORCE and KKT-HardNet do NOT use the
piecewise-linear regions during training or inference.
"""

SEGMENT_SCENARIOS = [1, 2, 3, 5, 7, 9, 11]
N_C_REGIONS = 3
N_TOTAL_POINTS = 170
N_REPEATS = 50
SEED = 0
