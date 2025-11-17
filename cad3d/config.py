import os
from dataclasses import dataclass

try:
    # Load environment variables from a local .env file if present
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass


@dataclass
class Settings:
    default_height: float = float(os.getenv("DEFAULT_EXTRUDE_HEIGHT", "3000"))
    oda_converter_path: str | None = os.getenv("ODA_CONVERTER_PATH")
    midas_onnx_path: str | None = os.getenv("MIDAS_ONNX_PATH")


settings = Settings()
