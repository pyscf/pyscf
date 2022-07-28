from typing import Optional
from dataclasses import dataclass

@dataclass
class IterationInfo:
    '''
        converged : bool
            Iteration steps are converged or not
        cycle : int
            Iteration cycle count
        message : str (optional)
            Additional message about iteration
    '''

    converged: bool = False
    cycle: int = 0
    message: Optional[str] = None
