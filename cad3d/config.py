"""
Application Configuration Module.

This module centralizes the configuration for the CAD 3D converter application.
It uses environment variables to allow for flexible configuration without
hardcoding values. For local development, it supports loading these variables
from a `.env` file in the project root.

The settings are encapsulated in a `Settings` dataclass, which provides
default values for essential parameters. This makes the application's
configuration explicit and easy to manage.

Key Configuration Parameters:
- `DEFAULT_EXTRUDE_HEIGHT`: The default height for extrusion operations.
- `ODA_CONVERTER_PATH`: The path to the ODA File Converter executable, which is
  necessary for handling DWG files.
- `MIDAS_ONNX_PATH`: The path to the ONNX model file used for depth estimation
  in the image-to-3D conversion process.
"""
import os
from dataclasses import dataclass

try:
    # For local development, load environment variables from a .env file.
    # This allows for easy configuration without modifying system-wide settings.
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # If python-dotenv is not installed, the application will proceed without it.
    # In production environments, variables should be set directly by the system.
    pass


@dataclass
class AppSettings:
    """
    A dataclass to hold and manage all configuration settings for the application.
    
    This class loads settings from environment variables upon instantiation,
    providing sensible default values for common parameters. This approach
    makes the configuration transparent and easily accessible throughout the app.
    """
    
    # Default height for extrusion operations if not otherwise specified by the user.
    # This is used by commands such as 'dxf-extrude' and 'auto-extrude'.
    # Environment Variable: DEFAULT_EXTRUDE_HEIGHT
    default_extrude_height: float = float(os.getenv("DEFAULT_EXTRUDE_HEIGHT") or "3000.0")
    
    # An explicit path to the ODA File Converter executable.
    # If this is not set, the application will attempt to find the executable in
    # common installation directories or the system PATH. This is essential for
    # any conversions involving DWG files.
    # Environment Variable: ODA_CONVERTER_PATH
    oda_converter_path: str | None = os.getenv("ODA_CONVERTER_PATH")
    
    # The path to the ONNX model file for depth estimation (e.g., a MiDaS model).
    # This is a required setting for the 'img-to-3d' command to function.
    # Environment Variable: MIDAS_ONNX_PATH
    midas_onnx_path: str | None = os.getenv("MIDAS_ONNX_PATH")


# Create a single, global instance of the settings to be imported by other modules.
# This ensures that configuration is loaded only once and is consistent across the app.
settings = AppSettings()
