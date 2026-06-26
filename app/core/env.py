from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_ENV_FILE = PROJECT_ROOT / "app" / ".env"
ROOT_ENV_FILE = PROJECT_ROOT / ".env"


def load_project_env() -> None:
    """Load environment from the project's root .env file."""
    load_dotenv(ROOT_ENV_FILE, override=True)
