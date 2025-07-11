
from fastapi import FastAPI
from routers import beyblade_detection

app = FastAPI()


app.include_router(beyblade_detection.router)
@app.get("/")
def read_root():
    return {"Hello": "World"}

