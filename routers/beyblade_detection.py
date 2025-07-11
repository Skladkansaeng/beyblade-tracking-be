import io
import shutil
import tempfile
from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from detector.yoloV8 import inference
router = APIRouter(
    prefix="/beyblade-detection",
    tags=["Beyblade Detection"],
    responses={404: {"description": "Not found"}},
)
@router.post("")
async def yolo_video_upload(file: UploadFile):
    """Takes a multi-part upload image and runs yolov8 on it to detect objects
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    return inference(tmp_path)