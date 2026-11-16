"""A small web server for the demo: POST /guess with the strokes drawn so far.

    pip install -r demo/requirements.txt
    DEMO_RUN=<package>/runs/cnn_imagenet_s42 PYTHONPATH=src uvicorn demo.server:app --port 8000
    open http://localhost:8000

POST /guess   {"strokes": [[[x0, x1, ...], [y0, y1, ...]], ...]}   canvas pixels, strokes in drawing order
GET  /info    run name, class names, thresholds
GET  /        the sample page in demo/static/index.html
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from quickdraw_exp.demo import Guesser

RUN = os.environ.get("DEMO_RUN")
if not RUN:
    raise SystemExit("set DEMO_RUN to a run folder of the checkpoint package, e.g. <package>/runs/cnn_imagenet_s42")
MAX_POINTS = 5000

guesser = Guesser(RUN, device=os.environ.get("DEMO_DEVICE", "cpu"))
app = FastAPI(title="AI Pictionary demo")
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("DEMO_ORIGINS", "*").split(","),
                   allow_methods=["GET", "POST"], allow_headers=["*"])


class Drawing(BaseModel):
    strokes: list[list[list[float]]]


@app.post("/guess")
def guess(drawing: Drawing) -> dict:
    points = 0
    for stroke in drawing.strokes:
        if len(stroke) != 2 or len(stroke[0]) != len(stroke[1]):
            raise HTTPException(422, "each stroke must be [[xs], [ys]] with as many xs as ys")
        points += len(stroke[0])
    if points > MAX_POINTS:
        raise HTTPException(413, f"at most {MAX_POINTS} points")
    return guesser.guess(drawing.strokes)


@app.get("/info")
def info() -> dict:
    return {"run": guesser.run, "classes": guesser.classes, "stop_threshold": guesser.stop_tau,
            "tree_threshold": guesser.tree_tau}


@app.get("/")
def page() -> FileResponse:
    return FileResponse(Path(__file__).with_name("static") / "index.html")
