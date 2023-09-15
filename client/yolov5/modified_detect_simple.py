# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path
import glob
import time

import pdb

import torch


import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.augmentations import letterbox
from utils.general import cv2
from models.common import DetectMultiBackend
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


class Yolo_Exec:

    def __init__(self, weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False, # use FP16 half-precision inference
        dnn=False):  # use OpenCV DNN for ONNX inference
        
        self.weights=weights

        self.data=data
        self.imgsz= imgsz
        
        self.imgsz *= 2 if len(self.imgsz) == 1 else 1
        self.conf_thres=conf_thres
        self.iou_thres=iou_thres
        self.max_det=max_det
        self.device=device
        self.view_img=view_img
        self.save_txt=save_txt
        self.save_conf=save_conf
        self.save_crop=save_crop
        self.nosave=nosave
        self.classes=classes
        self.agnostic_nms=agnostic_nms
        self.augment=augment
        self.visualize=visualize
        self.project=project
        self.name=name
        self.exist_ok=exist_ok
        self.line_thickness=line_thickness
        self.hide_labels=hide_labels
        self.hide_conf=hide_conf
        self.half=half
        self.dnn = dnn
        
        
    
        self.save_img = not self.nosave # save inference images



        # Directories
        self.save_dir = increment_path(Path(self.project) / self.name, exist_ok=self.exist_ok)  # increment run
        (self.save_dir / 'labels' if self.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        device = select_device(self.device)
        self.model = DetectMultiBackend(self.weights, device=device, dnn=self.dnn, data=self.data, fp16=self.half)
        self.stride, self.names, pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size

        # Dataloader
        bs = 1  # batch_size
        

        # Run inference
        self.model.warmup(imgsz=(1 if pt or self.model.triton else bs, 3, *self.imgsz))  # warmup
        seen, self.windows, self.dt = 0, [], (Profile(), Profile(), Profile())
        
    @smart_inference_mode()    
    def run(self,image):

        s = ""
        #image = cv2.imread(image_path)
        image_path = './img.png'
            
        im = letterbox(image, self.imgsz, stride=self.stride, auto=True)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous

        with self.dt[0]:
            im = torch.from_numpy(im).to(self.model.device)
            im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with self.dt[1]:
            visualize = increment_path(self.save_dir / Path(image_path).stem, mkdir=True) if self.visualize else False
            start_pred_time = time.time()
            pred = self.model(im, augment=self.augment, visualize=visualize)
            # print(len(pred))
            # print(pred[0].shape)
            # print(len(pred[1]))
            # for shp in pred[1]:
            #     print(shp.shape)
            # print("Pred time: %f" %(time.time() - start_pred_time))

        # NMS
        with self.dt[2]:
            start_nms_time = time.time()
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
            # print("NMS time: %f" %(time.time() - start_nms_time))
        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        line_return = []
        # Process predictions
        start_process_time = time.time()
        # print(pred[0])
        for i, det in enumerate(pred):  # per image

            p, im0, frame = image_path, image.copy(), 0

            p = Path(p)  # to Path
            save_path = str(self.save_dir / p.name)  # im.jpg
            txt_path = str(self.save_dir / 'labels' / p.stem)
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if self.save_crop else im0  # for self.save_crop
            # annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
            if len(det):
                # print("here")
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for det_row in reversed(det):

                    xyxy = det_row[:4].cpu()
                    conf = det_row[4].cpu()
                    cls = det_row[5].cpu()
                    extra = det_row[6:].cpu()


                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf, *extra) if self.save_conf else (cls, *xywh)  # label format
                    line_return.append(('%g ' * len(line)).rstrip() % line)

                    #print(line_return)
                    # if self.save_txt:  # Write to file
                    #     with open(f'{txt_path}.txt', 'a') as f:
                    #         f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    # if self.save_img or self.save_crop or self.view_img:  # Add bbox to image
                    #     c = int(cls)  # integer class
                    #     label = None if self.hide_labels else (self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
                    #     annotator.box_label(xyxy, label, color=colors(c, True))
                    # if self.save_crop:
                    #     save_one_box(xyxy, imc, file=self.save_dir / 'crops' / self.names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            # im0 = annotator.result()
            # if self.view_img:
            #     if platform.system() == 'Linux' and p not in self.windows:
            #         self.windows.append(p)
            #         cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
            #         cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
            #     cv2.imshow(str(p), im0)
            #     cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            # if self.save_img:
            #     cv2.imwrite(save_path, im0)
                

            # Print time (inference-only)
            # LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{self.dt[1].dt * 1E3:.1f}ms")
        # print("Process time: %f" %(time.time() - start_process_time))

        return line_return




