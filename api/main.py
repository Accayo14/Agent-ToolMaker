import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

from api.routes.jobs import router as jobs_router
from api.routes.tools import router as tools_router
from registry.store import init_db

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="ToolMaker API", version="0.1.0", lifespan=lifespan)
app.include_router(jobs_router)
app.include_router(tools_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
