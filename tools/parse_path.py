import os

import dotenv

dotenv.load_dotenv()


def parse_allowed_paths() -> list[str]:
    raw = os.getenv("ALLOWED_PATHS", "")  # noqa: F821
    if not raw:
        return []

    # Support comma/newline separated values and trim accidental spaces.
    normalized = raw.replace("\n", ",")
    return [p.strip() for p in normalized.split(",") if p.strip()]


print(*parse_allowed_paths())

print(*[p for p in os.getenv("ALLOWED_PATHS", "").split(",") if p])
