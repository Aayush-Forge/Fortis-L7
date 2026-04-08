from __future__ import annotations

import os

from fastapi import FastAPI
import uvicorn

app = FastAPI(title="fortis-l7-env")


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "fortis-l7-env", "status": "ok"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def main() -> None:
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
