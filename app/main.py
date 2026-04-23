from fastapi import FastAPI


def create_app() -> FastAPI:
    app = FastAPI(
        title="CloudEco Wildfire & Smoke Detection API",
        description="Local FastAPI service for wildfire and smoke detection.",
        version="0.1.0",
    )

    @app.get("/")
    def read_root() -> dict[str, str]:
        return {
            "message": "CloudEco Wildfire & Smoke Detection API",
            "docs": "/docs",
        }

    @app.get("/health")
    def health_check() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
