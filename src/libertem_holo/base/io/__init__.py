"""Basic I/O for holography data.

We mostly want to support loading holograms from DM{3,4} files, and save
results as numpy .npz files.
"""

from .results import Results
from .reader import InputData

__all__ = ["Results", "InputData"]
