# ECGFinetuneModel has moved to models/backbone.py (v2)
# This file is kept so existing imports don't raise FileNotFoundError.
from .backbone import ECGFinetuneModel
__all__ = ["ECGFinetuneModel"]
