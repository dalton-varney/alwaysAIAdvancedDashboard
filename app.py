import time
import edgeiq
from helpers import *
from sample_writer import *
from flask_socketio import SocketIO
from flask import Flask, render_template, request, send_file, url_for, redirect
import base64
import threading
import logging
from eventlet.green import threading as eventlet_threading
import cv2
from collections import deque

app = Flask(__name__, template_folder='./templates/')
socketio_logger = logging.getLogger('socketio')
socketio = SocketIO(app, logger=socketio_logger, engineio_logger=socketio_logger)
SAMPLE_RATE = 25
SESSION = time.strftime("%d%H%M%S", time.localtime())
video_stream = edgeiq.WebcamVideoStream(cam=1)

@app.route('/')
def index():
    """Home page."""
    return render_template('index.html')

@socketio.on('connect')
def connect_cv():
    print('[INFO] connected: {}'.format(request.sid))

@socketio.on('disconnect')
def disconnect_cv():
    print('[INFO] disconnected: {}'.format(request.sid))

@socketio.on('write_data')
def write_data():
    controller.start_writer()
    socketio.sleep(0.05)
    controller.update_text('Data Collection Started')
    print('start signal received')
    file_name = file_set_up("video", SESSION)

    with edgeiq.VideoWriter(output_path=file_name, fps=SAMPLE_RATE, codec='H264') as video_writer:
        if SAMPLE_RATE > video_stream.fps:
            raise RuntimeError(
                "Sampling rate {} cannot be greater than the camera's FPS {}".
                format(SAMPLE_RATE, video_stream.fps))

        print('Data Collection Started')
        while True:
            t_start = time.time()
            frame = controller.cvclient.video_frames.popleft()
            video_writer.write_frame(frame)
            t_end = time.time() - t_start
            t_wait = (1 / SAMPLE_RATE) - t_end
            if t_wait > 0:
                time.sleep(t_wait)
            time.sleep(0.01)
            if controller.is_writing() == False:
                print("ended")
                controller.update_text('Data Collection Ended')
                print('Data Collection Ended')
                break

            socketio.sleep(0.01)

@socketio.on('stop_writing')
def stop_writing():
    print('stop signal received')
    controller.stop_writer()
    socketio.sleep(0.01)

@socketio.on('start_hands')
def start_hands():
    controller.hands_activate()
    socketio.sleep(0.01)

@socketio.on('take_snapshot')
def take_snapshot():
    """Takes a single snapshot and saves it.
    """
    print('snapshot signal received')
    file_name = file_set_up("image", SESSION)
    controller.update_text('Taking Snapshot')
    print('Taking Snapshot')
    frame = controller.cvclient.all_frames.pop()
    cv2.imwrite(file_name, frame)
    controller.update_text('Snapshot Saved')
    print('Snapshot Saved')

@socketio.on('close_app')
def close_app():
    print('Stop Signal Received')
    controller.close_writer()
    controller.close()

@app.route('/download/<filename>', methods=['GET'])
def download(filename):
    file = os.path.join(".", get_file(filename))
    return send_file(file, as_attachment=True)

@app.route('/videos', methods=['GET'])
def videos():
    videos = {}
    files = get_all_files()
    if files:
        for f in files:
            videos[f] = (os.path.join(os.path.sep, get_file(f)))
    return render_template('videos.html', videos=videos)

@app.route('/analytics', methods=['GET'])
def analytics():

    return render_template('analytics.html')

@app.route('/view_video/<filename>', methods=['GET'])
def view_video(filename):
    file = os.path.join(os.path.sep, get_file(filename))
    if '.jpeg' in file:
        return render_template('view_video.html', image=file, filename=filename)
    else:
        return render_template('view_video.html', video=file, filename=filename)

@app.route('/delete/<filename>', methods=['GET'])
def delete(filename):
    file = os.path.join(".", get_file(filename))
    if file is not None:
        delete_file(file)
    return redirect(url_for('videos'))


class CVClient(eventlet_threading.Thread):
    def __init__(self, fps, exit_event):
        """The original code was created by Eric VanBuhler
        (https://github.com/alwaysai/video-streamer) and is modified here.

        Initializes a customizable streamer object that
        communicates with a flask server via sockets.

        Args:
            stream_fps (float): The rate to send frames to the server.
            exit_event: Threading event
        """
        self._stream_fps = SAMPLE_RATE
        self.fps = fps
        self._last_update_t = time.time()
        self._wait_t = (1/self._stream_fps)
        self.exit_event = exit_event
        self.writer = SampleWriter()
        self.all_frames = deque()
        self.video_frames = deque()
        self.HANDS = False
        super().__init__()

    def setup(self):
        """Starts the thread running.

        Returns:
            CVClient: The CVClient object
        """
        self.start()
        time.sleep(1)
        return self

    def run(self):
        print("Starting Up")
        #Object Detection
        obj_detect=edgeiq.ObjectDetection("alwaysai/mobilenet_ssd_face")
        hand_detect=edgeiq.ObjectDetection("alwaysai/hand_detection")

        obj_detect.load(engine=edgeiq.Engine.DNN)
        hand_detect.load(engine=edgeiq.Engine.DNN)
        tracker = edgeiq.CentroidTracker(deregister_frames=30)

        print("Loaded model:\n{}\n".format(obj_detect.model_id))
        print("Engine: {}".format(obj_detect.engine))
        print("Accelerator: {}\n".format(obj_detect.accelerator))

        video_stream.start()
        # Allow Webcam to warm up
        socketio.sleep(2.0)
        self.fps.start()

        prev_tracked_people = {}
        logs = []
        currentPeople = 0

        # loop detection
        while True:
            #Apply Object Detection model to each frame
            ogframe = video_stream.read()
            results = obj_detect.detect_objects(ogframe, confidence_level=.5)
            if self.HANDS:
                hand_results = hand_detect.detect_objects(ogframe, confidence_level=.5)

            #Sort out the people and count them
            if self.HANDS:
                people = edgeiq.filter_predictions_by_label(hand_results.predictions, ['hand'])
            else:
                people = edgeiq.filter_predictions_by_label(results.predictions, ['Face'])
            tracked_people = tracker.update(people)

            people = []
            for (object_id, prediction) in tracked_people.items():
                if self.HANDS:
                    prediction.label = 'Hand'
                else:
                    prediction.label = 'Face'
                #people.append(prediction)
            if self.HANDS:
                frame = edgeiq.markup_image(
                        ogframe, hand_results.predictions, colors=obj_detect.colors)
            else:
                frame = edgeiq.markup_image(
                        ogframe, results.predictions, colors=obj_detect.colors)

            new_entries = set(tracked_people) - set(prev_tracked_people)
            for entry in new_entries:
                logs.append('Person {} entered'.format(entry))
                currentPeople += 1

            new_exits = set(prev_tracked_people) - set(tracked_people)
            for exit in new_exits:
                logs.append('Person {} exited'.format(exit))
                currentPeople -= 1

            prev_tracked_people = dict(tracked_people)

            if self.HANDS:
                text = ["Model: {}".format(hand_detect.model_id)]
            else:
                text = ["Model: {}".format(obj_detect.model_id)]
            text.append(
                    "Neural Network FPS: {:1.2f}".format(1 / results.duration))
            text.append('\n')
            text.append('Current Occupancy:')
            if self.HANDS:
                text.append("{} Hands".format(str(currentPeople)))
                text.append('\n')
                text.append('Objects:')
                for prediction in hand_results.predictions:
                    text.append("{}: {:2.2f}%".format(
                        prediction.label, prediction.confidence * 100))
            else:
                text.append("{} Person".format(str(currentPeople)))
                text.append('\n')
                text.append('Objects:')
                for prediction in results.predictions:
                    text.append("{}: {:2.2f}%".format(
                        prediction.label, prediction.confidence * 100))
            text.append(self.writer.text)
            text.append('\n')
            text.append('\n')

            # enqueue the frames
            self.all_frames.append(frame)
            if self.writer.write == True:
                self.video_frames.append(ogframe)

            self.send_data(frame, text)

            self.fps.update()

            if self.check_exit():
                video_stream.stop()
                break


    def _convert_image_to_jpeg(self, image):
        """Converts a numpy array image to JPEG

        Args:
            image (numpy array): The input image

        Returns:
            string: base64 encoded representation of the numpy array
        """
        # Encode frame as jpeg
        frame = cv2.imencode('.jpg', image)[1].tobytes()
        # Encode frame in base64 representation and remove
        # utf-8 encoding
        frame = base64.b64encode(frame).decode('utf-8')
        return "data:image/jpeg;base64,{}".format(frame)

    def send_data(self, frame, text):
        """Sends image and text to the flask server.

        Args:
            frame (numpy array): the image
            text (string): the text
        """
        cur_t = time.time()
        if cur_t - self._last_update_t > self._wait_t:
            self._last_update_t = cur_t
            frame = edgeiq.resize(
                    frame, width=640, height=480, keep_scale=True)
            socketio.emit(
                    'server2web',
                    {
                        'image': self._convert_image_to_jpeg(frame),
                        'text': '<br />'.join(text)#,
                        #'data': get_all_files()
                    })
            socketio.sleep(0.0001)

    def check_exit(self):
        """Checks if the writer object has had
        the 'close' variable set to True.

        Returns:
            boolean: value of 'close' variable
        """
        return self.writer.close

    def close(self):
        """Disconnects the cv client socket.
        """
        self.exit_event.set()

class Controller(object):
    def __init__(self):
        self.write = False
        self.fps = edgeiq.FPS()
        self.cvclient = CVClient(self.fps, threading.Event())

    def start(self):
        self.cvclient.start()
        print('alwaysAI Dashboard http://localhost:3000')
        socketio.run(app=app, host='0.0.0.0', port=3000)

    def close(self):
        self.fps.stop()
        print("elapsed time: {:.2f}".format(self.fps.get_elapsed_seconds()))
        print("approx. FPS: {:.2f}".format(self.fps.compute_fps()))

        if self.cvclient.is_alive():
            self.cvclient.close()
            self.cvclient.join()

        print("Program Ending")

    def close_writer(self):
        self.cvclient.writer.write = False
        self.cvclient.writer.close = True

    def start_writer(self):
        self.cvclient.writer.write = True

    def hands_activate(self):
        if self.cvclient.HANDS:
            self.cvclient.HANDS = False
        else:
            self.cvclient.HANDS = True

    def stop_writer(self):
        self.cvclient.writer.write = False

    def is_writing(self):
        return self.cvclient.writer.write

    def update_text(self, text):
        self.cvclient.writer.text = text

controller = Controller()

if __name__ == "__main__":
    try:
        controller.start()
    finally:
        controller.close()
