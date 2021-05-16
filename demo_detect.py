import argparse
import time
import numpy as np
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, time_synchronized


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='./output.mp4', help='Path to video file')
    parser.add_argument('--imgsz', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show', type=bool, default=True, help='display results')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    opt = parser.parse_args()

    model_name = (opt.weights.split('.')[-2]).split('/')[-1]
    # Initialize
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    cudnn.benchmark = True  # set True to speed up constant image size inference

    # Load model
    model = attempt_load(opt.weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(opt.imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    cap = cv2.VideoCapture(opt.source)
    times_infer, times_pipe = [], []

    while True:
        ret, frame  = cap.read()

        if ret:
            frame = cv2.resize(frame, (1024,512))
            img = letterbox(frame, imgsz, stride=stride)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to (ch * w * h)
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            t0 = time.time()
            with torch.no_grad():
                pred = model(img, augment=opt.augment)[0]

                # Apply NMS
                pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

            out_img = frame.copy()
            gn = torch.tensor(out_img.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            for i, det in enumerate(pred):
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], out_img.shape).round()

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if opt.show:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = f'{names[c]} {conf:.2f}'
                            plot_one_box(xyxy, out_img, label=label, color=colors(c, True), line_thickness=opt.line_thickness)

            t1 = time.time()
            t2 = time.time()

            times_infer.append(t1-t0)
            times_pipe.append(t2-t0)
            
            imes_infer = times_infer[-20:]
            times_pipe = times_pipe[-20:]

            ms = sum(times_infer)/len(times_infer)*1000
            fps_infer = 1000 / (ms+0.00001)
            fps_pipe = 1000 / (sum(times_pipe)/len(times_pipe)*1000)

            # Stream results
            if opt.show:
                cv2.imshow(model_name, out_img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    break             # 1 millisecond

            #print("Time: {:.2f}ms, Detection FPS: {:.1f}, total FPS: {:.1f}".format(ms, fps_infer, fps_pipe))

        else:
            break
