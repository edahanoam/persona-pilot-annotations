"""Deployment entry point for Streamlit Community Cloud."""

from pathlib import Path
import runpy


runpy.run_path(Path(__file__).with_name("app.py"))
