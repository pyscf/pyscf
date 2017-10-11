from __future__ import division

def is_contiguous(array, dtype=None):
  """Check for contiguity and type."""
  if dtype is None:
    return array.flags.c_contiguous
  else:
    return array.flags.c_contiguous and array.dtype == dtype
