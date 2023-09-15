import os
import csv
import time
import subprocess
import psutil

from LanguageCE.test_ce import build_ce1, build_ce2, build_ce3, \
     build_carla_ce1, build_carla_ce2, build_carla_ce3, build_carla_ce4, build_carla_ce5

from domain_adapt import initialize_labelling_data, grab_aes, find_closest_aes, get_all_vicinal_events

import socket
from socket import SHUT_RDWR
import threading
import argparse
import traceback

import json
import argparse
import sys
from tqdm import tqdm




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

    # Vicinal event - get the camid
    ve_cam_id = vicinal_event[0]["camera_id"]

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
            # wb_camid = ve_cam_id

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
def localfs_listen(ce_obj, ce_structure, ae_files, wb_dict, debug_file):

    # Open each ae file, and get all the lines
    ae_file_lines = []
    num_lines = 0
    for ae_file in ae_files:
        with open(ae_file, "r") as f:
            lines = f.readlines()
            ae_file_lines.append(lines)
            num_lines = len(lines)
    
    # Now, Iterate through each line
    for i in range(num_lines):

        # Get the corresponding line of each file
        for ae_file_l in ae_file_lines:
            if i >= len(ae_file_l):
                continue
            line = ae_file_l[i]

            # Get the timestamp
            data = eval(line)
            frame_index = data["frame_index"]
            vicinal_event = data["vicinal_events"]
            tracks = data["tracks"]

            # If we have a vicinal event, then we can evaluate using it
            if vicinal_event:
                # print("EREERHERHEHEH")
                # print(data['frame_index'])
                ce_obj.update(vicinal_event, tracks, frame_index)
                results, change_of_state, current_func, extra_event_data = ce_obj.evaluate(frame_index)
                # print(results)
                # print(change_of_state)
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
    # debug_file = open(debug_filename, "w", buffering=1)
    # write_header_data(debug_file, ce_obj)
    
    print("Listening on " + str(server_addr))

    # Accept the connection from the client
    conn, addr = serverSocket.accept()
    print("Connection with " + str(addr) + " established")
    connection_open = True

    # Check to see what camera ID this is
    data = conn.recv(512)
    decoded = data.decode()
    # cam_id = int(decoded.split(":")[1])
    # ce_obj.client_info[cam_id] = addr
    print("Received camid of " + str(cam_id))

    print("Sending video data...")
    # Open the video file and transmit it
    with open(video_path, "rb") as video_file:
        video_data = video_file.read()
        # while video_data:
        conn.sendall(video_data)
        time.sleep(2)
        conn.send(b"done")
        # sendMessage("Hello", client_addr, serverSocket)
        # video_data = video_file.read()
    print("Sent video data!")

    
    # Now for the recv logic
    while True:
        
        data = conn.recv(512)
        if data:
            decoded = data.decode()

            # If this is a message on having received the video and is ready to run
            #  Send function information
            if "ready" in decoded:

                print(ce_obj.config_watchboxes)
                # Be sure to hand back the corresponding watchboxes
                if cam_id in ce_obj.config_watchboxes:
                    return_message = "watchboxes:" + str(ce_obj.config_watchboxes[cam_id])
                else:
                    return_message = "watchboxes:[]"

                # Add some other things to the return message
                return_message += ":" + str(cam_id) + ":" + remote_folder

                print(return_message)
                sendMessage(return_message, addr, conn)

                # Have to also send the camera id and remote folder
                # sendMessage(cam_id, addr, conn)
                # sendMessage(remote_folder, addr, conn)


            elif "quitting:" in decoded:

                print("quitout!")

                # Write our data to file
                # result_file = '/'.join([ce_obj.result_dir, "ce_output.json"])
                # with open(result_file, "w") as wfile:
                #     json.dump(ce_obj.result_output, wfile)
                # break
                # debug_file.close()
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
                
                
                # Get the timestamp of this event:
                # frame_index_incoming = incoming_data[0]["time"]
                # data_to_write = [frame_index_incoming]
                
                
                # ce_obj.result_output["incoming"].append([incoming_data, frame_index_incoming])
                # data_to_write.append(incoming_data)


  
                # for res in result:
                #     data_to_write.append([res, change_of_state, old_results])
                
                #     debug_file.write(":::".join([str(x) for x in data_to_write]) + "\n")
                #     data_to_write = data_to_write[:-1]  # Pop the last element back out
                

                        
                        
                    
        # except Exception as e:
        #     print(traceback.format_exc())
        #     input()
    if connection_open:
        conn.shutdown(SHUT_RDWR)
        conn.close()
    # debug_file.close()

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

    if solo_execution:  # We grab data from local files
            # server_listen_thread =  threading.Thread(target=localfs_listen, \
            # args=(complexEventObj,ce_structure,ae_files[c_i], wb_dict, debug_file,cam_id,remote_folder,))
            
            # No, it's better to do this deterministically
            localfs_listen(complexEventObj,ce_structure,ae_files, wb_dict, debug_file)
    else:
        for c_i in range(num_cameras):
            # Set up our server

            server_listen_thread = None
            
            server_listen_thread = threading.Thread(target=netserver_listen, \
                args=(complexEventObj,ce_structure,server_addr,video_files[c_i],cam_id,remote_folder,))
            server_listen_thread.start()
            server_threads.append(server_listen_thread)

    # Wait for all server threads to exit
    while True and not solo_execution:
        time.sleep(1)
        # Check if all server threads are alive
        threads_status = [x.is_alive() for x in server_threads]
        if not any(threads_status):  # If all are dead, we can move on
            break

    print("FINISHED!")
    return complexEventObj

# Get the length of each file
def get_num_lines_in_file(filepath):
    lines = []
    with open(filepath, "r") as f:
        lines = f.readlines()
    return len(lines)

# Go through multiple files, find the max length
def get_max_file_length(filepaths):

    lengths = []
    for filepath in filepaths:
        lengths.append(get_num_lines_in_file(filepath))
    return max(lengths)

# #Get the complex event object
# def get_complex_event(ce_index, class_mappings):
#     complexEventObj = None
#     ce_structure = []
#     if ce_index == 1:
#         complexEventObj, ce_structure = build_carla_ce1(class_mappings)
    
#     elif ce_index == 2:

#         complexEventObj, ce_structure =  build_carla_ce2(class_mappings)

#     elif ce_index == 3:

#         complexEventObj, ce_structure =  build_carla_ce3(class_mappings)
    
#     elif ce_index == 4:

#         complexEventObj, ce_structure =  build_carla_ce4(class_mappings)
    
#     elif ce_index == 5:

#         complexEventObj, ce_structure =  build_carla_ce5(class_mappings)

#     # complexEventObj.set_result_dir(result_dir)

#     return complexEventObj

# Here we re-parse the ce file for soartech
#   and check if the output makes sense
def reparse_ce(ce_file, ae_statuses):

    ae_names = [x[0] for x in ae_statuses]
    ae_result = []

    # Read ce file line by line, and verify against each ae status
    with open(ce_file, "r") as f:
        ce_lines = f.readlines()
        for ce_line in ce_lines[2:]:
            line_data = eval(ce_line)
            # Get the name
            result_ae_name = line_data["complex_event"]["ae_list"]
            if line_data["complex_event"]["ae_operator"]:
                result_ae_name.append(line_data["complex_event"]["ae_operator"])
            
            # Check if the ae matches
            if result_ae_name in ae_names:
                result_ae = line_data["complex_event"]["result"]
                if result_ae:
                    ae_result.append((result_ae_name, result_ae))

    return ae_result

def execute_main_experiment(server_config, solo_execution):


    # Get networking information
    server_port = server_config["server_port"]
    server_ip = server_config["server_ip"]
    server_addr = (server_ip, server_port)

    # Get class mapping information
    class_mappings = server_config["class_mappings"]
    ae_files = []
    ce_type = server_config["ce_type"]

    # List our all the video directories
    video_parent_folder = ""
    ae_result_folder = ""

    # Let's just pick some examples
    selected = ["ce_0_ds_0_it_1", "ce_1_ds_0_it_1", "ce_2_ds_0_it_2", \
        "ce_3_ds_0_it_1", "ce_4_ds_0_it_1"]
    selected = ["ce_0_ds_0_it_1"]


    # Results:
    ce_dict = {}  # dict of 'ceX-dsY':[number cases, num correctly detected]
    #   where X is the ce number, and Y is the domain shift number

    # Missed CEs
    missed_ces = []

    total_cameras = 1
    video_dirs = os.listdir(video_parent_folder)
    print("Creating CE output results!")
    # Iterate through each ce dir
    for ce_dir in tqdm(video_dirs):

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
        

        # If the current ce_dir is not in our ae results continue
        if ce_dir not in os.listdir(ae_result_folder):
            continue

        # Only choose our selected ones
        # if ce_dir not in selected:
        #     continue
        # if "ce_2" not in ce_dir:
        #     continue

        video_folder = os.path.join(video_parent_folder, ce_dir)
        # Now, iterate through each video file
        video_files = os.listdir(video_folder)

        video_files_in = []
        cam_id = 0
        remote_folder = ce_dir

        for vfile in video_files:

            vfilepath = os.path.join(video_folder, vfile)
            video_files_in.append(vfilepath)
            cam_id = int(vfile[3])
            # ce_number = 5
            # video_files = [["../../example.mp4"]]#, ["../../example.mp4"]]
            

            # Set up the server
            # for videos_to_send in video_files:
            if solo_execution:
                # ae_files = server_config["ae_files"]
                ae_dir = os.path.join(ae_result_folder, ce_dir)

                if not os.path.exists(ae_dir): # AE dir does not exist
                    continue

                # List all the ae files
                ae_filepaths = os.listdir(ae_dir)
                ae_filepaths = [os.path.join(ae_dir, x) for x in ae_filepaths]
                ae_filepaths = [x for x in ae_filepaths if "ae_cam" in x]
                ae_files = sorted(ae_filepaths)
                total_cameras = len(ae_files)
                result_dir = ae_dir

        # If this result dir does not exist (e.g. no files)
        if result_dir == "" or len(ae_files) == 0:
            continue

        # Generate CE output
        # print(result_dir)
        ce_obj = setup_ce_detector(ce_number, server_addr, total_cameras, \
            video_files_in, class_mappings, result_dir, solo_execution, ae_files, \
            cam_id, remote_folder, ce_type)


        # Now perform analysis of the ce output
        max_length = get_max_file_length(ae_files)
        ce_result_filepath = os.path.join(result_dir, "ce_output.txt")
        # print(ce_result_filepath)

        # Get relevant AEs and watchbox data
        relevant_aes, ae_programs, wb_data = initialize_labelling_data(ce_result_filepath,\
            ae_files, max_length)
        # Obtain all vicinal events
        # unconfirmed_vicinal_events, search_ae = grab_aes(relevant_aes, ae_programs, wb_data)
        unconfirmed_vicinal_events = get_all_vicinal_events(ae_files, wb_data)

        # print("hi")
        # for x in unconfirmed_vicinal_events.keys():
        #     print(x)
        #     print(len(unconfirmed_vicinal_events[x]))
        # asdf
        ae_statuses = find_closest_aes(unconfirmed_vicinal_events, ce_obj, max_length)

        if ce_type == "soartech":
            ae_statuses = reparse_ce(ce_result_filepath, ae_statuses)

        print([(x[0], x[1]) for x in ae_statuses])

        # Check if ce occurred
        ce_occurred = all([y[1] for y in ae_statuses])*1
        # If it did not occur, add to our missed
        if not ce_occurred:
            missed_ces.append(ce_dir)

        # Add to our ce dict
        ds_number = ce_dir[8]
        dkey = "ce"+str(ce_number)+"-ds"+ds_number
        if dkey not in ce_dict:
            # Add element in
            ce_dict[dkey] = [1, ce_occurred]
        else:
            ce_dict[dkey][0] += 1
            ce_dict[dkey][1] += ce_occurred

    # Now, print out the ce dict
    print(ce_dict)
    print(missed_ces)



parser = argparse.ArgumentParser(description='Central CE reasoner')
parser.add_argument('--config', type=str, default='configs/local_carla.json')
parser.add_argument('--solo_execution', action='store_true', help="This will ignore networking, and just test the server code using AE files")
args = parser.parse_args()

if __name__ == "__main__":

    
    # Open our config file
    with open(args.config, "r") as f:
        server_config = json.load(f)
            
    execute_main_experiment(server_config, args.solo_execution)

            






