"""Agents module for R-Zero implementation"""
from .teacher import TeacherAgent
from .solver import SolverAgent

# Export the main classes
__all__ = ['TeacherAgent', 'SolverAgent']

# Also export the format constants for convenience
from .teacher import REASONING_START, REASONING_END, SOLUTION_START, SOLUTION_END

__all__.extend(['REASONING_START', 'REASONING_END', 'SOLUTION_START', 'SOLUTION_END'])
