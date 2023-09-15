from event_detector import SensorEventDetector

import argparse
import time
import sys
from tqdm import tqdm

import traceback
import warnings
warnings.filterwarnings("ignore")

import json

parser = argparse.ArgumentParser(description='Edge Processing Node')
parser.add_argument('--track_alg', type=str, default='Byte', help='Track algorithm: Byte, Sort, DeepSort or MOTDT')
parser.add_argument('--device', type=str, default='0', help="CUDA device where to run YOLO")
parser.add_argument('--config', type=str, default='configs/local.json')
parser.add_argument('--camera_id', type=int, help="This is the camera ID assigned to this client")
parser.add_argument('--solo_execution', action='store_true', help="This will ignore networking, and just test the client code without a server")
parser.add_argument('--show_detections', action='store_true')
parser.add_argument('--enable_track_patch', action='store_true')


args = parser.parse_args()


if __name__=='__main__':
    
    try:

        # Open our config file
        with open(args.config, "r") as f:
            client_config = json.load(f)

        # Load networking information
        event_detector_ip = client_config["event_detector_ip"]
        ce_server_ip = client_config["ce_server_ip"]
        event_detector_port = client_config["event_detector_port"]
        ce_server_port = client_config["ce_server_port"]

        # Load in model
        model_path = client_config["model_path"]
        model_input_size = eval(client_config["model_input_size"])

        # Get video path (the file here will get replaced anytime we run a new experiment)
        temp_video_path = client_config["temp_video_path"]
        # Get the directory where we will write data
        result_dir = client_config["result_dir"]
        stride = client_config["stride"]

        # Get watchbox information, which is mainly for testing in solo execution mode
        extra_wb_data = ""
        if args.solo_execution:
            extra_wb_data = client_config["extra_wb_data"]



        # First, check if we are using a *.engine or *.pt file
        # if ".engine" in model_path:
        #     from yolo_trt.yolov5_core import Yolov5 as Yolov5Trt
        #     yolo = Yolov5Trt(classes="coco",
        #             backend="tensorrt",
        #             weight=model_path,
        #             auto_install=False,
        #             dtype="fp16", input_shape=model_input_size)
        if ".pt" in model_path:
            from yolov5.modified_detect_simple import Yolo_Exec
            yolo = Yolo_Exec(weights=model_path, imgsz=model_input_size, \
            conf_thres=0.5, device=args.device, save_conf=True) #'../../delivery-2022-12-01/t72detect_yv5n6.pt',imgsz=[2560],conf_thres=0.5)

        # If we are using patches
        track_patches = None
        if not args.enable_track_patch:
            recover_lost_track = False
            buffer_zone = 0
            ignore_stationary = False
            enter_anywhere = True
            track_patches = (recover_lost_track, buffer_zone, ignore_stationary, enter_anywhere)
        else:
            recover_lost_track = True
            buffer_zone = 20
            ignore_stationary = False
            enter_anywhere = False
            track_patches = (recover_lost_track, buffer_zone, ignore_stationary, enter_anywhere)


        event_detectors = []

        relevant_frames = None

        # Iterate through the number of cameras
        # Loop and continue to perform event detection infinitely
        while True:
            eventDetector = SensorEventDetector(yolo, args.track_alg, args.camera_id, \
                                                (event_detector_ip, event_detector_port),
                                                (ce_server_ip, ce_server_port), relevant_frames, result_dir, \
                                                    track_patches, \
                                                    args.solo_execution, model_path, \
                                                    args.show_detections, temp_video_path, \
                                                    extra_wb_data)

            start_frame = 0

            # Now, iterate over all frames 
            last_sample_time = time.time()
            num_frames = 0
            frame_index = 0

            for frame in tqdm(eventDetector.reader):
                # Only execute when we are at the stride
                if frame_index % stride == 0:
                    # Then we can run the full detection pipeline
                    eventDetector.execute_full_detection(frame_index, frame, stride)
                
                if time.time() - last_sample_time > 1:
                    # print("Processing rate: %d fps (per camera)" % (num_frames))
                    num_frames = 0
                    last_sample_time = time.time()
                num_frames += 1
                frame_index += 1

            # Next, iterate again over all the eventDetectors to print their results to a file
            eventDetector.completed(eventDetector.serverAddr)




            # Lastly, specifically add a tag if this executed correctly
            print("Ended correctly.")
            time.sleep(20)

            # If this is a solo execution, end immediately.
            if args.solo_execution:
                break

    
    except Exception as e:
        print(traceback.format_exc())
