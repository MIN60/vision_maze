import cv2
import numpy as np
import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
import threading

mazeWidth = 700
mazeHeight = 700
stageTextHeight = 100
#step = 15
        
def mouse_event(event, x, y, flags, param):
    global detect_thread
    if event == cv2.EVENT_LBUTTONDOWN:
        # cv2.destroyWindow("main")
        
        stage()
        #combined_function()

def stage():
    stageImage = cv2.imread("C:/opencv_project/vision_maze/data/stage.png", cv2.IMREAD_ANYCOLOR)
    stageImage = cv2.resize(stageImage, dsize=(0, 0), fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
    cv2.imshow("stage", stageImage)
    while True:      
        key = cv2.waitKey(1)
        if key == ord('1'):
            combined_function("C:/opencv_project/vision_maze/data/maze1.png",  350, 120, [(300, 640), (400, 700)], 23, 17)
            #maze("C:/Users/jaeeun/Desktop/final_computerVision/maze1.png",  350, 120, [(300, 740), (400, 790)], 23, 17) #1단계
        if key == ord('2'): 
            combined_function("C:/opencv_project/vision_maze/data/maze2.png",  350, 120, [(310, 650), (390, 690)], 17, 11)
            #maze("C:/Users/jaeeun/Desktop/final_computerVision/maze2.png",  350, 120, [(310, 750), (390, 790)], 17, 11) #2단계
        if key == ord('3'): 
            combined_function("C:/opencv_project/vision_maze/data/maze3.png", 350, 120, [(330, 650), (370, 690)], 10, 7)
            #maze("C:/Users/jaeeun/Desktop/final_computerVision/maze3.png", 350, 120, [(330, 750), (370, 790)], 10, 7) #3단계
        if cv2.waitKey(1) == 27:
            break



class Dot:
    def initDot(self, initX, initY, dest, radius, step): #출발지점, 도착영역, 반지름을 지정한다.
        self.initX = initX
        self.initY = initY
        self.nowX = initX
        self.nowY = initY
        self.dest = dest
        self.radius = radius
        self.step = step
        
    def changeLoc(self, x, y, resultImage):
        self.nowX = self.nowX + x
        self.nowY = self.nowY + y
        #벽 확인
        if resultImage[self.nowY, self.nowX+self.radius][0] < 200: #오른쪽
            self.nowX = self.nowX - self.step 
        if resultImage[self.nowY, self.nowX-self.radius][0] < 200:#왼쪾
            self.nowX = self.nowX + self.step 
        if resultImage[self.nowY+self.radius, self.nowX][0] < 200:#아래
            self.nowY = self.nowY - self.step 
        if resultImage[self.nowY-self.radius, self.nowX][0] < 200:#위
            self.nowY = self.nowY + self.step
        #미로 내부에만 있도록 제한
        if self.nowX<0:
            self.nowX = 0
        elif self.nowX > 700:
            self.nowX = 700
        if self.nowY<0:
            self.nowY = 0
        elif self.nowY > 700:
            self.nowY = 700
    def checkArrival(self, resultImage):
        if self.nowX> self.dest[0][0] and self.nowX < self.dest[1][0] and self.nowY> self.dest[0][1] and self.nowY < self.dest[1][1]:
            successImage = cv2.imread("C:/opencv_project/vision_maze/data/success.png", cv2.IMREAD_ANYCOLOR)
            successImage = cv2.resize(successImage, dsize=(0, 0), fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
            cv2.imshow("Success", successImage)
            self.nowX = self.initX
            self.nowY = self.initY
            if cv2.waitKey(1) == ord('q'):
                cv2.imshow("main", resultImage) # 메인 화면 열기
                cv2.destroyWindow("Success") #성공 창 없애기 
            return True


# def mouse_evenct(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         cv2.destroyWindow("main")
#         maze()
        

# main 함수 내부에서는 `maze` 함수와 `detect` 함수를 별도의 스레드로 실행합니다.


def main():
    main_image = cv2.imread("C:/opencv_project/vision_maze/data/main.png", cv2.IMREAD_COLOR)
    if main_image is None:
        print("Could not load main image")
        exit()

    while True:
        new_width, new_height = 1200, 700
        main_image = cv2.resize(main_image, (new_width, new_height))
        cv2.imshow("main", main_image)
        cv2.setMouseCallback("main", mouse_event)

        c = cv2.waitKey(1)
        if c == ord('0'):
            helpImage = cv2.imread("C:/opencv_project/vision_maze/data/help.png", cv2.IMREAD_ANYCOLOR)
            cv2.imshow("help", helpImage)
        if c == ord('q'):
            break

    cv2.destroyAllWindows()


def combined_function(imagePath, initX, initY, dest, radius, step):
    webcam = cv2.VideoCapture(0)  # 첫 번째 웹캠
    if not webcam.isOpened():
        print("Could not open webcam")
        exit()

    # 이미지 로드
    mazeImage = cv2.imread(imagePath, cv2.IMREAD_ANYCOLOR)
    mazeImage = cv2.resize(mazeImage, dsize=(700, 700), interpolation=cv2.INTER_AREA)
    mazeHeight, mazeWidth, _ = mazeImage.shape

    dot = Dot()
    dot.initDot(initX, initY, dest, radius, step)

    # Detect 관련 초기화
    device = select_device('')
    model = attempt_load('runs/train/exp4/weights/best.pt', map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(320, s=stride)
    names = model.module.names if hasattr(model, 'module') else model.names

    while True:
        status, frame = webcam.read()
        if not status:
            break
        # OpenCV는 BGR 형식으로 이미지를 읽어오므로, RGB로 변환
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 이미지 크기 조정 (모델이 기대하는 크기로)
        frame_resized = cv2.resize(frame_rgb, (imgsz, imgsz))
        
        # 이미지 차원 변환: (높이, 너비, 채널) -> (채널, 높이, 너비)
        frame_transposed = np.transpose(frame_resized, (2, 0, 1))

        # Detect 로직
        img = torch.from_numpy(frame_transposed).to(device)
        img = img.float() / 255.0  # 0 - 255 to 0.0 - 1.0
        img = img.unsqueeze(0)  # 배치 차원 추가

        pred = model(img, augment=False)[0]
        pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

        # Maze 게임 로직
        #cv2.circle(resultImage, (dot.initX, dot.initY), radius, (255, 0, 255), -1, 8, 0)
        cv2.circle(mazeImage, (dot.nowX, dot.nowY), dot.radius, (255, 0, 255), -1)
        #cv2.rectangle(mazeImage, dot.dest[0], dot.dest[1], (0, 0, 255), 1, 8, 0)
        cv2.imshow("Maze", mazeImage)

        now_detect = [] # 검출된 class 저장
        for i, det in enumerate(pred):  # detections per image
            if len(det):
                # Rescale boxes from img size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                # Draw boxes and labels on the image
                for *xyxy, conf, cls in reversed(det):
                    class_name = names[int(cls)]
                    now_detect.append(class_name)
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, frame, label=label, color=(255, 0, 0), line_thickness=1)
        print("Detected classes:", now_detect)

        cv2.imshow("Detection", frame)

        # 키 입력 처리
        # 키 입력 처리
        c = cv2.waitKey(1)
        if 'up' in now_detect:
            dot.changeLoc(0, -step, mazeImage)
        elif 'right' in now_detect:
            dot.changeLoc(step, 0, mazeImage)
        elif 'down' in now_detect:
            dot.changeLoc(0, step, mazeImage)
        elif 'left' in now_detect:
            dot.changeLoc(-step, 0, mazeImage)
        elif 'fist' in now_detect:
            dot.changeLoc(0, 0, mazeImage)
        elif c == 27:  # ESC
            break
        
        if (dot.checkArrival(mazeImage)):
            break


    webcam.release()
    cv2.destroyAllWindows()

#combined_function()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='runs/train/exp4/weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='0', help='source')  # 0 for webcam
    parser.add_argument('--img-size', type=int, default=320, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')

    #parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    #parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    #parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    #parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    main()
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                #detect()
                strip_optimizer(opt.weights)
        #else:
            #detect()