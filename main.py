
from fastapi import FastAPI
from routers import beyblade_detection
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(beyblade_detection.router)
@app.get("/")
def read_root():
    return {"Hello": "World"}

