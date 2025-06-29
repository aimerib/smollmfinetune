"""
Telemetry SDK CLI Entry Point

Enables usage: python -m telemetry_sdk.report <run_id>
"""

from .cli import main

if __name__ == '__main__':
    main() 