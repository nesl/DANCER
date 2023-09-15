import os
import csv
import time
import subprocess
import psutil
import sys
sys.path.append("./LanguageCE")
from LanguageCE.test_ce import build_ce1, build_ce2, build_ce3, \
     build_carla_ce1, build_carla_ce2, build_carla_ce3, build_carla_ce4, build_carla_ce5

import socket
from socket import SHUT_RDWR
import threading
import argparse
import traceback

import json
import argparse


# Send data - encode it 
def sendMessage(message, addr, conn):
    # Turn the message into bytes
    message = str(message).encode()
    print("sending to " + str(addr))
    # sock.sendto(message, addr)
    conn.send(message)
        

# Iterate through the ae code, and see what watchboxes it involves
def match_watchboxes(wb_names, ae_code):

    matched_watchboxes = []
    for wb_name in wb_names:
        if wb_name in ae_code:
            matched_watchboxes.append(wb_name)

    return matched_watchboxes

# Recursively grab all AEs involved in each complex event
def recurse_get_ae_data(ae_list, wb_names, ae_dict):

    
    # Base case - we only have 1 item:
    if len(ae_list) == 1:
        ae_name = ae_list[0][0]
        ae_code = ae_list[0][1]        
        wb_matches = match_watchboxes(wb_names, ae_code)
        ae_dict[ae_name] = [wb_matches, ae_code]
    else:  # We continue to recurse
        for x in ae_list[:-1]:
            recurse_get_ae_data([x], wb_names, ae_dict)



#  Our CE Output.txt should have header information that
#  looks more like this:
#    The first 2 lines will be the AE name and its language content, then the watchbox data
#       e.g. {"ae_name": [[wb_name1, wb_name2, etc], "wb1.composition..."] }
#            {"wb_name": {"cam_id": 1, "watchbox_id": 1, "coords": (xx,yy)} }


# Format our output for writing header info to a file
# THis means AE_LIST, and WB_LIST
def write_header_data(debug_file, ce_obj):

    # First, get the name of every watchbox
    wb_names = list(ce_obj.watchboxes.keys())
    # Write our first line, which is the AE content and watchbox names involved
    ae_dict = {}
    for x in ce_obj.executable_functions:
        # Now, go through every AE, and get its data
        recurse_get_ae_data(x, wb_names, ae_dict)
    
    # Now, write our second line, which is all the watchbox data necessary
    watchbox_dict = {}
    for wb_name in ce_obj.watchboxes.keys():
        wb_data = ce_obj.watchboxes[wb_name]
        cam_id = wb_data.camera_id
        positions = wb_data.positions
        watchbox_id = wb_data.watchbox_id
        classes = wb_data.classes

        watchbox_dict[wb_name] = { 
            "cam_id": cam_id, \
            "watchbox_id": watchbox_id, \
            "positions": positions, \
            "classes": classes
        }

    # Write AE data into the file    
    debug_file.write(str(ae_dict) + "\n")
    # Write watchbox data into the file
    debug_file.write(str(watchbox_dict) + "\n")

    return ae_dict, watchbox_dict


#  Every line of content in the ce_output.txt will look like:
#    {
#      current_frame_index: XXXX,
#      vicinal_events: [ ],
#      atomic_event: [ {"ae_name":
#         [wb_name, time, camera, [object_track_ids]...] } ],  #This is nonempty if an atomic event occurs
#      complex_event: {}, # If an atomic event occurs,
#              this becomes nonempty with {"ae_operator": "", "ae_list": [], "result": False }
#    }

# Format our output for writing data to the file
def write_event_data(debug_file, ce_data, vicinal_event, wb_dict, frame_index):
    

    line_content = {"frame_index": frame_index, "vicinal_events": vicinal_event, \
                    "atomic_event": []}

    # Now, if we have an event which occurred, save ae data
    for atomic_event in ce_data["event_data"]:

        # Get the atomic event name
        ae_name = list(atomic_event.keys())[0]
        ae_dict = {ae_name : []}
        # For each watchbox, get the time of its event, its camera, and track ids
        for watchbox_data in atomic_event[ae_name]:
            wb_name = watchbox_data[0]
            wb_time = watchbox_data[1]["time"]
            wb_track_ids = list(watchbox_data[1]["objects"].keys())
            wb_camid = wb_dict[wb_name]["cam_id"]

            ae_dict[ae_name].append({"wb_name": wb_name, "time": wb_time, \
                    "camid": wb_camid, "track_ids": wb_track_ids})

        line_content["atomic_event"].append(ae_dict)

    # Now we have to get the complete complex event data
    ce_entry = {}
    if len(ce_data["func_tup"]) == 1:
        ce_entry["ae_operator"] = ""
        ce_entry["ae_list"] = [ce_data["func_tup"][0][0]] # Get function name
        ce_entry["result"] = ce_data["all_func_results"][-1] # Get latest result
    else:
        ce_entry["ae_operator"] = ce_data["func_tup"][-1]
        ce_entry["ae_list"] = [x[0] for x in ce_data["func_tup"][:-1]] # Get function names
        ce_entry["result"] = ce_data["all_func_results"][-1] # Get latest result
        
    line_content["complex_event"] = ce_entry
    
    print(line_content)
    # Write to our debug file
    debug_file.write(str(line_content) + "\n")






# This is an alternative server which basically loads in ae files instead
#   and runs them directly rather than communicating over the network.
def localfs_listen(ce_obj, ce_structure, ae_file, wb_dict, debug_file):


    with open(ae_file, "r") as f:
        
        # Loop through each entry
        while True:
            line = f.readline()

            # Break if we have no more lines
            if not line:
                break

            # Get the timestamp
            data = eval(line)
            frame_index = data["frame_index"]
            vicinal_event = data["vicinal_events"]

            # If we have a vicinal event, then we can evaluate using it
            if vicinal_event:
                ce_obj.update(vicinal_event, frame_index)
                results, change_of_state, current_func, extra_event_data = ce_obj.evaluate(frame_index)

                # Change of state means that any of the atomic events have toggled
                if change_of_state:

                    ce_data = {"all_func_results": results, "func_tup": current_func, "event_data": extra_event_data}

                    write_event_data(debug_file, ce_data, vicinal_event, wb_dict, frame_index)

# This is our listening server, which is where we get all of our results
#  This is the flow of information between server and client:
#  clients say hello - this establishes which clients are active
#  server sends video
#  Clients acknowledge they have received and loaded the video
#  Server sends start signal

def netserver_listen(ce_obj, ce_structure, server_addr, video_path,cam_id,remote_folder):

    # First, bind the server
    serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serverSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    serverSocket.bind(server_addr)
    serverSocket.listen()
    
    # Open our file for writing
    debug_filename = '/'.join([ce_obj.result_dir, "ce_output.txt"])
    
    print("Listening on " + str(server_addr))

    # Accept the connection from the client
    conn, addr = serverSocket.accept()
    print("Connection with " + str(addr) + " established")
    connection_open = True

    # Check to see what camera ID this is
    data = conn.recv(512)
    decoded = data.decode()
    print("Received camid of " + str(cam_id))

    print("Sending video data...")
    # Open the video file and transmit it
    with open(video_path, "rb") as video_file:
        video_data = video_file.read()
        # while video_data:
        conn.sendall(video_data)
        time.sleep(2)
        conn.send(b"done")
    print("Sent video data!")

    
    # Now for the recv logic
    while True:
        
        data = conn.recv(512)
        if data:
            decoded = data.decode()

            # If this is a message on having received the video and is ready to run
            #  Send function information
            if "ready" in decoded:
                # Be sure to hand back the corresponding watchboxes
                if cam_id in ce_obj.config_watchboxes:
                    return_message = "watchboxes:" + str(ce_obj.config_watchboxes[cam_id])
                else:
                    return_message = "watchboxes:[]"

                # Add some other things to the return message
                return_message += ":" + str(cam_id) + ":" + remote_folder

                sendMessage(return_message, addr, conn)


            elif "quitting:" in decoded:

                print("quitout!")
                conn.shutdown(SHUT_RDWR)
                conn.close()
                connection_open = False
                time.sleep(10)
                break
            else:
                
                # Format is like: [{'camera_id': '2', 'results': [[[0], [True], 3]], 'time': 17910}]
                
                # Otherwise, this is data.
                incoming_data = eval(decoded)
                print("Receiving " + str(incoming_data))
                        
                        
    if connection_open:
        conn.shutdown(SHUT_RDWR)
        conn.close()


# Set up several listening threads for capturing event data from cameras.
def setup_ce_detector(ce_index, server_addr, num_cameras, video_files, \
    class_mappings, result_dir, solo_execution, ae_files,cam_id,remote_folder,\
    ce_type):

    try:
        complexEventObj = None
        ce_structure = []

        if ce_type == "carla":
            if ce_index == 1:
                complexEventObj, ce_structure = build_carla_ce1(class_mappings)
            
            elif ce_index == 2:

                complexEventObj, ce_structure =  build_carla_ce2(class_mappings)

            elif ce_index == 3:

                complexEventObj, ce_structure =  build_carla_ce3(class_mappings)
            
            elif ce_index == 4:

                complexEventObj, ce_structure =  build_carla_ce4(class_mappings)
            
            elif ce_index == 5:

                complexEventObj, ce_structure =  build_carla_ce5(class_mappings)
        elif ce_type=="soartech":
            if ce_index == 1:
                complexEventObj, ce_structure = build_ce1(class_mappings)

        complexEventObj.set_result_dir(result_dir)

    except Exception as e:
        print(traceback.format_exc())
        input()
    
    

    debug_filename = '/'.join([complexEventObj.result_dir, "ce_output.txt"])
    debug_file = open(debug_filename, "w", buffering=1)
    ae_dict, wb_dict = write_header_data(debug_file, complexEventObj)


    # Set up some TCP threads for communication
    server_threads = []
    for c_i in range(num_cameras):
        # Set up our server

        server_listen_thread = None
        if solo_execution:  # We grab data from local files
            server_listen_thread =  threading.Thread(target=localfs_listen, \
            args=(complexEventObj,ce_structure,ae_files[c_i], wb_dict, debug_file,cam_id,remote_folder,))
        else:  # We listen over the network
            server_listen_thread = threading.Thread(target=netserver_listen, \
                args=(complexEventObj,ce_structure,server_addr,video_files[c_i],cam_id,remote_folder,))
        server_listen_thread.start()
        server_threads.append(server_listen_thread)

    # Wait for all server threads to exit
    while True:
        time.sleep(1)
        # Check if all server threads are alive
        threads_status = [x.is_alive() for x in server_threads]
        if not any(threads_status):  # If all are dead, we can move on
            break

    print("FINISHED!")



def execute_main_experiment(server_config, solo_execution):

    # Get networking information
    server_port = server_config["server_port"]
    server_ip = server_config["server_ip"]
    server_addr = (server_ip, server_port)

    # Get class mapping information
    class_mappings = server_config["class_mappings"]
    ae_files = []
    if solo_execution:
        ae_files = server_config["ae_files"]
    ce_type = server_config["ce_type"]

    # List our all the video directories
    video_parent_folder = server_config["video_folder"]

    total_cameras = 1
    video_dirs = os.listdir(video_parent_folder)
    # Iterate through each ce dir
    for ce_dir in video_dirs:

        # Get the ce number
        ce_numer = -1
        if ce_type == "carla":
            ce_number = int(ce_dir[3])
            result_dir = "ce_results"
            if ce_number == 0:
                ce_number = 5
        elif ce_type == "soartech":
            ce_number = 1
            result_dir = "ce_results"

        video_folder = os.path.join(video_parent_folder, ce_dir)
        # Now, iterate through each video file
        video_files = os.listdir(video_folder)

        for vfile in video_files:

            vfilepath = os.path.join(video_folder, vfile)
            remote_folder = ce_dir
            cam_id = int(vfile[3])

            try:
                setup_ce_detector(ce_number, server_addr, total_cameras, \
                    [vfilepath], class_mappings, result_dir, solo_execution, ae_files, \
                    cam_id, remote_folder, ce_type)
            except Exception as e: 
                print(e)
                break





parser = argparse.ArgumentParser(description='Central CE reasoner')
parser.add_argument('--config', type=str, default='configs/local.json')
parser.add_argument('--solo_execution', action='store_true', help="This will ignore networking, and just test the server code using AE files")
args = parser.parse_args()

if __name__ == "__main__":

    
    # Open our config file
    with open(args.config, "r") as f:
        server_config = json.load(f)
            
    execute_main_experiment(server_config, args.solo_execution)

            






