import math
import os
import subprocess
import tempfile
import cv2
import threading
import queue
from concurrent.futures import ThreadPoolExecutor

from fastapi.responses import StreamingResponse

from inference.inferencer import InferenceModel, MovementTrailVideo, add_point, log_inference_time


def reencode_for_web(input_path, output_path):
    """Re-encode video for web compatibility using FFmpeg"""
    cmd = [
        'ffmpeg',
        '-i', input_path,
        '-c:v', 'libx264',
        '-profile:v', 'baseline',
        '-level', '3.0',
        '-pix_fmt', 'yuv420p',
        '-crf', '23',
        '-preset', 'fast',  # Faster for real-time processing
        '-movflags', '+faststart',
        '-y',
        output_path
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


@log_inference_time
def ultra_fast_process_all_frames(video_path, num_threads=8):

    def frame_reader(cap, frame_queue):
        """Read frames in separate thread"""
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_queue.put(frame)
        frame_queue.put(None)  # Signal end

    def frame_processor(frame_queue, results_queue):
        """Process frames in separate thread"""
        while True:
            frame = frame_queue.get()
            if frame is None:
                break
            results_queue.put(frame)
            frame_queue.task_done()

    cap = cv2.VideoCapture(video_path)

    cap.set(cv2.CAP_PROP_FPS, 60)
    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_queue = queue.Queue(maxsize=50)  # Buffer size
    results_queue = queue.Queue()

    # Start reader thread
    reader_thread = threading.Thread(
        target=frame_reader, args=(cap, frame_queue))
    reader_thread.start()

    # Start processor threads
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for _ in range(num_threads):
            future = executor.submit(
                frame_processor, frame_queue, results_queue)
            futures.append(future)

        # Wait for reader to finish
        reader_thread.join()

        # Signal all processors to stop
        for _ in range(num_threads):
            frame_queue.put(None)

        # Wait for all processors
        for future in futures:
            future.result()

    # cap.release()

    # Collect results
    results = []
    while not results_queue.empty():
        results.append(results_queue.get())

    return results, fps, width, height


def batch_list(lst, batch_size):
    """Split a list into batches of a given size."""
    return [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]


def inference(tmp_path):

    frame_results, fps, width, height = ultra_fast_process_all_frames(
        tmp_path, num_threads=8)

    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tmpfile_path = tmpfile.name
    tmpfile.close()

    # Create VideoWriter with same properties
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(tmpfile_path,
                          fourcc, fps, (width, height))
    model = InferenceModel().get_model()

    batch_results = batch_list(frame_results, 50)
    results = []
    points = []

    for frame_result in batch_results:
        results += model.predict(source=frame_result, verbose=False)

    for idx, result in enumerate(results):
        # result = model.inference(frame)
        frame = frame_results[idx]
        # Draw OBBs on the frame
        for r in result:
            for obb in r.obb:
                if obb is not None:
                    pred = obb.xywhr.tolist()[0]
                    [x, y, w, h, r] = [int(x) for x in pred]

                    if len(points) == 0:
                        points.append(
                            {'point': [x, y, w, h, r], 'video': MovementTrailVideo()})
                    else:
                        for idx, point in enumerate(points):
                            distance = math.dist(
                                [point['point'][0], point['point'][1]], [x, y])
                            if distance < 100:
                                point['point'] = [x, y, w, h, r]

                            elif distance > 100:
                                if len(points) < 50:
                                    add_point(idx, points, distance,
                                              [x, y, w, h, r])
                                else:
                                    points = []

            for point in points:
                [x, y] = point['point'][:2]
                point['video'].draw_trail_opencv(frame, (x, y))

        out.write(frame)

    out.release()

    web_tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    web_tmpfile_path = web_tmpfile.name
    
    if reencode_for_web(tmpfile_path, web_tmpfile_path):
            # Use the web-compatible version
            final_path = web_tmpfile_path
            # Clean up the temporary file
            try:
                os.remove(tmpfile_path)
            except:
                pass
    else:
        # Fallback to original if re-encoding fails
        final_path = tmpfile_path
        try:
            os.remove(web_tmpfile_path)
        except:
            pass

    def iterfile():
        with open(final_path, mode="rb") as file_like:
            yield from file_like
        # Clean up files
        try:
            os.remove(final_path)
            os.remove(tmp_path)
        except:
            pass

    return StreamingResponse(iterfile(), media_type="video/mp4")
