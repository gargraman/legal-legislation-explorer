"""Python equivalents of the project notebooks.

Each module mirrors one notebook's logic, refactored into importable functions
with a ``main()`` entry point so the pipeline can run outside Jupyter and be
unit-tested. The original ``.ipynb`` files remain the source of truth and are
left untouched.

Pipeline order (same as the notebooks):

    crawler  ->  loader  ->  vectorize  ->  indices
                                   |
                                 queries   (examples.ipynb: ad-hoc queries + viz)
"""

__all__ = ["config", "crawler", "loader", "vectorize", "indices", "queries"]
