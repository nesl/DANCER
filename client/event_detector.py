import numpy as np
import cv2
import os
import socket
from socket import SHUT_RDWR
import pybboxes as pbx
from tqdm import tqdm
import torch
import sys
import torchvision
from byte_tracker.byte_tracker import BYTETracker
from vidsz.opencv import Reader, Writer
import time

from client_utils import non_max_suppression, xyxy2xywh, scale_boxes


def state_init(state, track, functions, arguments):
    
    for f in functions:
        #state[track][f] = {'data': [], 'results': np.full((len(arguments[f]),), False)}
        
        if f not in state[track]:
            if f == 'convoy':
                state[track][f] = {'data': [], 'results': [[] for i in range(len(arguments[f]))]}
            else:
                state[track][f] = {'data': [], 'results': np.full((len(arguments[f]),), False)}
        else:
            new_args = np.full((len(arguments[f]),), False)
            state[track][f]['results'] = np.extend(state[track][f]['results'],new_args)
        
    return state
    
def state_add(state, functions, arguments):
    
    for s in state.keys():
        state = state_init(state, s, functions, arguments)
        
    return state

#Cross trip_wire function
def cross_tripwire(tracks,state, tripwires):
    results = []
    for t_key in tracks.keys():
        if state[t_key]['cross_tripwire']['data']:

            past_state = state[t_key]['cross_tripwire']['data'] - tripwires
            current_state = tracks[t_key][0][0] - tripwires
            results_tmp = np.where(np.sign(past_state)*np.sign(current_state) < 0)[0]

            
            if results_tmp.size > 0:
                #print(results_tmp, t_key)
                results.append([results_tmp, t_key])
            
        
        state[t_key]['cross_tripwire']['data'] = tracks[t_key][0][0]
        
    return results
    

# Check which direction an event occurred (where did it enter, where did it leave)
def check_event_direction(object_position, watchbox_points):

    distance_threshold = 10 # Must be within Xpx of the watchbox edge

    # Check if this entered/exited top, left, right, or bottom, or middle
    
    event_directions = []
    for wb_item in watchbox_points:

        # If x coordinate is closest to the left x coordinate
        if abs(object_position[0] - wb_item[0]) < distance_threshold:
            # print("left")
            event_directions.append("left")
        elif abs(object_position[1] - wb_item[1]) < distance_threshold: # If y coordinate closest to top y coordinate
            # print("top")
            event_directions.append("top")
        elif abs(object_position[0] - wb_item[2]) < distance_threshold: # If x coordinate closest to right x coordinate
            # print("right")
            event_directions.append("right")
        elif abs(object_position[1] - wb_item[3]) < distance_threshold: # If y coordinate closest to bottom y coordinate
            # print("bottom")
            event_directions.append("bottom")
        else:
            event_directions.append("middle")

    return event_directions


# Given a bounding box, find the middle point
def calculate_midpoint(x1, y1, x2, y2):

    x = x1+(x2 - x1)/2, 
    y = y1 + (y2 - y1)/2
    return [x,y]

#Watchbox function
#  This will output data of the structure
#   [ [ [watchbox ID], [ object_entered ], track_id, [direction, speed] ]    ]

def watchbox(tracks,state, watchboxes, min_history = 8):
    results = []

    for t_key in tracks.keys():

            # Remember - watchboxes is like [ [x1,x2,y1,y2,[cls1, cls2]]...]

            select = np.where(watchboxes[:,4] == tracks[t_key][2])[0]
            
            if select.size > 0:
                # print(select)
                reference_point = calculate_midpoint(tracks[t_key][0][0], \
                    tracks[t_key][0][1], tracks[t_key][0][2], tracks[t_key][0][3] )
                # print(reference_point)
                p1 = reference_point[0] - watchboxes[select,0] > 0
                p2 = reference_point[1] - watchboxes[select,1] > 0
                p3 = reference_point[0] - watchboxes[select,2] < 0
                p4 = reference_point[1] - watchboxes[select,3] < 0


                ptotal = p1 & p2 & p3 & p4

                try:
                    results_tmp = np.logical_xor(ptotal,state[t_key]['watchbox']['results'][select])
                except:
                    # pdb.set_trace()
                    print("Error")
                results_tmp = np.nonzero(results_tmp)[0]
                results_tmp = select[results_tmp]
                state[t_key]['watchbox']['results'][select] = ptotal


                if results_tmp.size > 0:

                    # We have results here, so figure out direction and speed
                    watchboxes_of_interest = results_tmp.tolist()

                    watchbox_data = []
                    for wb_i in watchboxes_of_interest:
                        watchbox_data.append(watchboxes[wb_i])
                    
                    # Check the direction of this event object
                    event_directions = check_event_direction(reference_point, watchbox_data)
                    # Check the speed of the object
                    # t_speed = speed_of_track(tracks[t_key][5])


                    wb_ids_of_event = [watchboxes[x][5] for x in results_tmp]

                    results.append(
                        {"track_id": t_key, "watchboxes": wb_ids_of_event, \
                            "enters": state[t_key]['watchbox']['results'][results_tmp].tolist(), \
                            "directions": event_directions, "class": tracks[t_key][2]}
                    )

                
    return results,state     

#Speed function    
def speed(tracks, state, speeds):
    results = []
    
    speed_threshold = 0.1
    
    for t_key in tracks.keys():
        select = np.where(speeds[:,1] == tracks[t_key][2])[0]
        reference_point = [tracks[t_key][0][0]+(tracks[t_key][0][2] - tracks[t_key][0][0])/2, tracks[t_key][0][1] + (tracks[t_key][0][3] - tracks[t_key][0][1])/2]
        
        if state[t_key]['speed']['data']:
            distance = np.linalg.norm(state[t_key]['speed']['data'] - reference_point)
            v = distance/(1/fps)
            ptotal = np.absolute(v - speeds[select,0]) > speed_threshold
            
            results_tmp = np.logical_xor(ptotal,state[t_key]['speed']['results'][select])

            results_tmp = np.nonzero(results_tmp)[0]
            results_tmp = select[results_tmp]
            state[t_key]['speed']['results'][select] = ptotal
            
            if results_tmp.size > 0:
                results.append([results_tmp, state[t_key]['speed']['results'][results_tmp], t_key])
        
        state[t_key]['speed']['data'] = reference_point


        
    return results,state
    
    
def convoy(tracks, state, groups):
    results = []
    

    reference_points = np.array([[tracks[t_key][0][0]+(tracks[t_key][0][2] - tracks[t_key][0][0])/2, tracks[t_key][0][1] + (tracks[t_key][0][3] - tracks[t_key][0][1])/2, tracks[t_key][2], t_key] for t_key in tracks.keys()])
    min_number_vehicles = 3
    
    res_per_track_id = {}
    results_tmp = {}    
    for g_idx,g in enumerate(groups):
    
        rp_idxs = np.where(reference_points[:,2] == g[1])[0] #Check error here
        
        if len(reference_points) > min_number_vehicles and len(rp_idxs) > min_number_vehicles:
            
            
            clustering = None 

            labels,label_counts = np.unique(clustering, return_counts=True)
            count_res = np.where(label_counts > min_number_vehicles)[0]

            if count_res.size > 0:
                for possible_convoy in count_res:
                    track_idxs = np.where(clustering == labels[possible_convoy])[0]
                    convoy_elements = reference_points[track_idxs,3]
                    if state[convoy_elements[0]]['convoy']['results'][g_idx]:
                        
                        if not set(state[convoy_elements[0]]['convoy']['results'][g_idx]) == set(convoy_elements):
                            
                            for t_key in tracks.keys():

                                state[t_key]['convoy']['results'][g_idx] = convoy_elements.tolist()
                                if t_key not in results_tmp:
                                    results_tmp[t_key] = []
                                    
                                results_tmp[t_key].append([g_idx, convoy_elements.tolist()])
                    else:
                        for t_key in tracks.keys():
                            state[t_key]['convoy']['results'][g_idx] = convoy_elements.tolist()
                            
                            if t_key not in results_tmp:
                                results_tmp[t_key] = []
                                    
                            results_tmp[t_key].append([g_idx, convoy_elements.tolist()])
                            
                    
            else:
                for t_key in tracks.keys():
                    if state[t_key]['convoy']['results'][g_idx]:
                        if t_key not in results_tmp:
                            results_tmp[t_key] = []
                        results_tmp[t_key].append([g_idx, []])

                    state[t_key]['convoy']['results'][g_idx] = []
                    
        else:
            for t_key in tracks.keys():
                if state[t_key]['convoy']['results'][g_idx]:
                    if t_key not in results_tmp:
                        results_tmp[t_key] = []
                    results_tmp[t_key].append([g_idx, []])

                state[t_key]['convoy']['results'][g_idx] = []

    for result_key in results_tmp.keys():       

        results.append([[group[0] for group in results_tmp[result_key]], [group[1] for group in results_tmp[result_key]], result_key])
        
    
    return results, state

class ByteTrackArgs:
    def __init__(self, track_thresh, match_thresh, track_buffer):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.mot20 = False 
        self.track_buffer = track_buffer


class SensorEventDetector:

    def __init__(self, \
                     yolo_model, track_alg, camera_id, currentAddr, serverAddr,\
                    relevant_frames, result_dir, track_patches, solo_execution, model_name, \
                    show_detections, video_filepath, extra_wb_data):
        
        # Used in tracking
        self.track_alg = track_alg
        self.tracks = {}
        self.state = {}
        self.track_id = 0
        self.old_tracks = []

        # Determine if we are using ground truth.  If so, we can ignore parts of the detection
        #  and tracking.
        self.use_gt = False
        self.gt_mapping = {}  #this maps a special ID to our numbers
        
        # Determine relevant frames:
        self.relevant_frames = relevant_frames
        self.result_dir = result_dir
        # Make the directory if it isn't already there
        if not os.path.exists(self.result_dir):
            os.mkdir(self.result_dir)

        # Filter classes
        filter_classes = [float(x) for x in range(0, 100)]
        
        
        # Used to record functions
        self.functions = []
        self.function_metadata = {}
        
        # Self explanatory
        self.camera_id = camera_id
        self.yolo_model = yolo_model
        
        # This is where our full pipeline results sit (e.g. watchbox detections)
        self.write_to_file_res = []
        
        # Store data related to our video capture
        #  What frame are we on?  
        self.current_cap_frame_index = -1
        
            
        # Also initialize our trackers
        #Choose tracking algorithm
        self.tracker = None
        if track_alg == 'Byte':
            recover_lost_track, buffer_zone, ignore_stationary, enter_anywhere = track_patches
            # My buffer size is X minutes at 30fps - if a track is lost for more than X minutes, it is deleted.
            new_args = ByteTrackArgs(0.5,0.8, 10000000)  # Basically always track same ID
            self.tracker = BYTETracker(new_args, filter_classes, recover_lost_track, \
                buffer_zone, ignore_stationary, 10, enter_anywhere)
        elif track_alg == 'Sort':
            self.tracker = Sort(new_args.track_thresh)
        elif track_alg == 'MOTDT':
            self.tracker = OnlineTracker('trackers/pretrained/googlenet_part8_all_xavier_ckpt_56.h5', use_tracking=False)
        elif track_alg == 'DeepSort':
            self.tracker = DeepSort('trackers/pretrained/ckpt.t7')
            
            
        # Lastly, be sure to set up our server stuff for this detector
        self.currentAddr = currentAddr
        self.serverAddr = serverAddr
        self.sock = None
        

        # Check if we are doing this solo (no network)
        self.solo_execution = solo_execution
        self.model_name = model_name
        self.show_detections = show_detections

        # do the handshake
        # video_filepath = "data/video.mp4"
        self.handshake(currentAddr, serverAddr, video_filepath)

        # Now, send an ack back that you have received, loaded the video, and did warmup
        self.reader = Reader(video_filepath)
        self.pixel_width, self.pixel_height = \
        self.reader.read().shape[1], self.reader.read().shape[0]

        print(self.pixel_width)
        print(self.pixel_height)

        if ".engine" in self.model_name:
            #  Do a warmup for the model
            print("Starting Warmup")
            warmup = np.random.randint(0, 255, self.reader.read().shape).astype("float")
            for i in tqdm(range(10)):
                self.yolo_model(warmup)

        # Recieve watchbox data and other data
        self.recv_watchbox_data(serverAddr)

        save_folder = os.path.join(self.result_dir, self.remote_folder)
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)

        print(save_folder)

        debug_filename = '/'.join([save_folder, "ae_cam"+str(self.camera_id)+".txt"])
        self.debug_file = open(debug_filename, "w", buffering=1)

        

        # Configs for nms
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.classes = None
        self.agnostic_nms = False
        self.max_det = 1000
    
    
    def sendMessage(self, message, serverAddr):

        if self.solo_execution:
            return
        
        # Turn the message into bytes
        message = str(message).encode()
        print("sending to " + str(serverAddr))
        self.sock.send(message)
        
        
    
    # Perform the handshake
    #  This means first sending this event detector's camera ID
    #  Then listening to the CE server to get the watchbox coordinates
    def handshake(self, currentAddr, serverAddr, video_filepath):

        if self.solo_execution:
            return
        
        # First, initialize our socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        self.sock.connect(serverAddr)
        
        # Next, send data to the server address
        camera_message = "camera_id:"+str(self.camera_id)
        self.sendMessage(camera_message, serverAddr)
        

        # Now, receive video data
        print("Receiving video data...")
        
        video_file = open(video_filepath, "wb")
        while True:
            data = self.sock.recv(1024)
            if data == b"done":
                break
            video_file.write(data)

        print("Finished receiving...")
        

    # Parse and save watchbox data
    def parse_and_save_watchbox_data(self, watchbox_data):
        watchbox_data = watchbox_data.split("watchboxes:")[1]
        watchbox_data = eval(watchbox_data)
        print("received watchbox data: " + str(watchbox_data))
        
        # If we don't get any watchbox data, then we don't call any additional functions
        if watchbox_data:
            self.functions = ["watchbox"]
            self.function_metadata['watchbox'] = np.array(watchbox_data)


    # Receive watchbox data
    def recv_watchbox_data(self, serverAddr):

        if self.solo_execution:
            return

        # Send 'ready'
        print("Sending Ready")
        self.sendMessage("ready", serverAddr)

        
        # Next, temporarily listen for data
        multi_data = self.sock.recv(1024)
        
        # Set our watchbox data
        multi_data = multi_data.decode()
        multi_data = multi_data.split(":")
        watchbox_data = ':'.join(multi_data[:2])
        self.parse_and_save_watchbox_data(watchbox_data)

        # Get the other data
        self.camera_id = int(multi_data[2])
        self.remote_folder = multi_data[3]

        
    # This just tells the server that this camera process has completed
    def completed(self, serverAddr):

        if self.solo_execution:
            return
        
        self.sendMessage("quitting:", serverAddr)
        self.sock.shutdown(SHUT_RDWR)
        self.sock.close()
        self.debug_file.close()  # Close our writing file
        
    
    

        
        
    
    # Do a prediction using the model and NMS
    def do_prediction(self, image):

        # Predict using the model
        output = self.yolo_model(image)
        # Now run nms
        pred = non_max_suppression([output], self.conf_thres, self.iou_thres, \
            self.classes, self.agnostic_nms, max_det=self.max_det)
        
        line_return = []
        # Process predictions
        start_process_time = time.time()
        # print(pred[0])
        for i, det in enumerate(pred):  # per image

            im0, frame = image.copy(), 0

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):

                # Write results
                for det_row in reversed(det):

                    xyxy = det_row[:4].cpu()
                    conf = det_row[4].cpu()
                    cls = det_row[5].cpu()
                    extra = det_row[6:].cpu()


                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf, *extra) # if self.save_conf else (cls, *xywh)  # label format
                    line_return.append(('%g ' * len(line)).rstrip() % line)

        return line_return
    

    # Grab an image at an index, and get the yolo results
    def do_object_detection(self, frame, frame_index):
        
        start_read_time = time.time()

        res_lines = []


        if self.use_gt:  # Get the ground truth res lines
            res_lines = self.gt_source.generate_results_for_frame(self.camera_id, frame_index)
        else:
            if ".engine" in self.model_name:
                res_lines = self.do_prediction(frame)
            elif ".pt" in self.model_name:
                res_lines = self.yolo_model.run(frame) #Run yolo
        
        return frame, res_lines

    # Check if a newly assigned track is allowed
    def allowed_track_entrance(self, bbox):
        allowed = False

        if bbox[0] > 0 and bbox[1] > 0 and bbox[2] < 300 and bbox[3] < 1079:
            allowed = True

        return allowed

        
    # Now execute our tracker
    def update_tracker(self, image, res_lines, frame_index):
        
        detection_bboxes = np.array([])
        detection_class_ids = np.array([])
        detection_confidences = np.array([])
        detection_extra = np.array([])

        # So here's the content of each element of res_lines:
        #  [ class prediction, x1, y1, x2, y2, objectness score, something, then class scores ]

       
        # Iterate through every YOLO result, and add them to detection_bboxes
        for line in res_lines:


            coordinates_line = line.split()

            # if int(coordinates_line[0]) > 2: #Only pedestrians
            #     continue

            box_voc = None
            if self.use_gt:
                box_voc = coordinates_line[1:5]
                box_voc = [int(x) for x in box_voc]
                class_id = int(coordinates_line[0])
                #  Get the class and object ID
                # Create our tracks here.
                obj_id = coordinates_line[5]
                if obj_id not in self.gt_mapping:
                    self.gt_mapping[obj_id] = self.track_id
                    self.tracks[self.track_id] = (np.array(box_voc), frame_index, class_id)
                    
                    if self.track_id not in self.state:
                        self.state[self.track_id] = {}
                        self.state = state_init(self.state,self.track_id,self.functions,self.function_metadata)

                    self.track_id += 1

                else:
                    current_id = self.gt_mapping[obj_id]
                    self.tracks[current_id] = (np.array(box_voc), frame_index, class_id)

                    # Now, put text on the image
                    fontScale = 0.5
                    color = (255, 153, 255)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    thickness = 2
                    image = cv2.putText(image, str(current_id), (box_voc[0],box_voc[1]), font, 
                                fontScale, color, thickness, cv2.LINE_AA)
                

            else:    
                box_voc = pbx.convert_bbox((float(coordinates_line[1]),float(coordinates_line[2]),float(coordinates_line[3]),float(coordinates_line[4])), from_type="yolo", to_type="voc", \
                                            image_size=(self.pixel_width, self.pixel_height))

                if detection_bboxes.size == 0:
                    detection_bboxes = np.expand_dims(np.array(box_voc),axis=0)
                else:
                    detection_bboxes = np.concatenate((detection_bboxes, np.expand_dims(np.array(box_voc),axis=0)),axis=0)
                #detection_bboxes = np.append(detection_bboxes, np.expand_dims(np.array(box_voc),axis=0),axis=0)
                detection_class_ids = np.append(detection_class_ids, int(coordinates_line[0]))
                detection_confidences = np.append(detection_confidences, float(coordinates_line[5]))

                extra_data = np.array([float(cl) for cl in coordinates_line[6:]])
                if detection_extra.size == 0:
                    detection_extra = np.expand_dims(extra_data,axis=0)
                else:
                    detection_extra = np.concatenate((detection_extra,np.expand_dims(extra_data,axis=0)),axis=0 )


            # Finally, draw the rectangle
            cv2.rectangle(image, (box_voc[0], box_voc[1]), (box_voc[2], box_voc[3]), (0, 0, 255), 1)

        text_output = ''
        issue_pause = False
        if detection_bboxes.size > 0:

            if self.track_alg == 'MOTDT' or self.track_alg == 'DeepSort':
                online_targets, issue_pause = \
                    self.tracker.update(np.column_stack((detection_bboxes,detection_confidences)), \
                                [self.pixel_height, self.pixel_width], (self.pixel_height, self.pixel_width), image2)
            else:
                bbox_stack = np.column_stack((detection_bboxes, detection_confidences))
                online_targets, issue_pause = \
                    self.tracker.update(bbox_stack, \
                                        [self.pixel_height, self.pixel_width], (self.pixel_height, self.pixel_width),detection_class_ids, image, detection_extra)

            new_tracks = []
            for t_idx,t in enumerate(online_targets):

                self.track_id = t.track_id
                bbox = t.tlbr
                class_history = t.detected_class
                movement_history = t.location_window
                detection_extra = t.detected_extra
                new_tracks.append(self.track_id)

                class_detected = t.voted_class

                self.tracks[self.track_id] = (bbox,frame_index,class_detected, \
                    class_history, detection_extra, movement_history)
                
                
                if self.track_id not in self.state:
                    self.state[self.track_id] = {}
                    self.state = state_init(self.state,self.track_id,self.functions,self.function_metadata)

                #Put label outside bounding box
                fontScale = 0.5
                color = (255, 153, 255)
                font = cv2.FONT_HERSHEY_SIMPLEX
                thickness = 2
                image = cv2.putText(image, str(self.track_id), (int(bbox[0]),int(bbox[1])), font, 
                               fontScale, color, thickness, cv2.LINE_AA)

            if set(new_tracks) != set(self.old_tracks):
               
                for tt in list(set(self.old_tracks) - set(new_tracks)):
                    del self.tracks[tt]


            self.old_tracks = new_tracks
        
        return image, issue_pause
    
    
    # Execute the additional functions, such as watchbox recognition, etc.
    def execute_additional_functions(self, image, frame_index):
        
        frame_result = []
        for f in self.functions:
            #Apply functions according to query (right now only two tripwires are checked)
            res,state = eval(f+"(self.tracks,self.state,self.function_metadata['" + f +"'])")
            
            if f =='watchbox':
                 # Also, add the bounding boxes to check if this is working
                for wb in self.function_metadata["watchbox"]:
                    cv2.rectangle(image, (wb[0], wb[1]), \
                                      (wb[2], wb[3]), (0, 0, 255), 1)
            
            if res:
                new_res = {}
                new_res["camera_id"] = self.camera_id
                new_res["results"] = res
                new_res['time'] = frame_index

                if f == 'watchbox':
                    print("Vicinal event occurred:")
                    print(new_res)
                    frame_result.append(new_res)
                    
                   


                if f == 'cross_tripwire': #If there is an event (change of state)
                    print("Tripwires", r[0], "crossed by", r[1], "at", frame_index+1)

                if f == 'convoy':
                    m, s = divmod(frame_index/fps, 60)
                    h, m = divmod(m, 60)
                    print("Convoy", new_res, m,s)
                    
        return image, frame_result

    # Format data from our detections and tracking for writing to a file
    def format_and_write(self, frame_data):

        #  Iterate through all of tracks
        #   We only are interested in:
        #     - the track id, position, current class prediction
        track_data = {}
        for track_key in frame_data["tracks"].keys():
            single_track_data = frame_data["tracks"][track_key]

            bbox_data = single_track_data[0]
            class_detected = single_track_data[2]
            track_data[track_key] = {"bbox_data": [round(x,2) for x in list(bbox_data)], "prediction": class_detected}


        # Prepare to print
        frame_data["tracks"] = track_data
        self.debug_file.write(str(frame_data) + "\n")


    
    # Execute our full detection pipeline, from yolo to tracking to watchbox processing
    def execute_full_detection(self, frame_index, frame, stride):
            
        start_loop_time = time.time()
        
        data_out = {"frame_index": frame_index}

        # Get the object detection result
        image, res_lines = self.do_object_detection(frame, frame_index)
        # print("Time for object detection: %f" % (time.time()-start_loop_time))


        # Add the traffic camera label to images
        # font
        font = cv2.FONT_HERSHEY_SIMPLEX
        # org
        org = (50, 50)
        # fontScale
        fontScale = 1
        # COLOR in BGR
        color = (0, 255, 0)
        # Line thickness of 2 px
        thickness = 2
        # Using cv2.putText() method
        # image = cv2.putText(image, 'Traffic Camera ID: ' + str(self.camera_id), org, font, 
        #            fontScale, color, thickness, cv2.LINE_AA)

        issue_pause = False
        if res_lines:

            # Now, update our tracker
            track_start_time = time.time()
            image, issue_pause = self.update_tracker(image, res_lines, frame_index)
        
        data_out["tracks"] = self.tracks

        # Perform extra processing, such as for tripwires of watchboxes
        func_start_time = time.time()
        image, frame_result = self.execute_additional_functions(image, frame_index)
        data_out["vicinal_events"] = frame_result

        if self.show_detections:
            cv2.imshow('image'+str(self.camera_id),image)
            cv2.waitKey(1)
  

        if issue_pause:
            print("Pausing")
            input()
        
        self.format_and_write(data_out)

        
