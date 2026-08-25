"""Shared pieces the report scripts are built from.

``ranking_payload`` is the one schema both extractors emit, so a single-user run
and a 50-user run produce the same per-user JSON and downstream consumers have
one shape to read rather than two that drift apart.
"""
