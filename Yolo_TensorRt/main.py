import os
os.chdir('/home/fengsc/CUDASTUDY/Yolo_TensorRt')
from ultralytics import YOLO
from ultralytics.utils.benchmarks import benchmark
import torch
print(torch.cuda.is_available())
model = YOLO("yolov8n.pt")
# model.export(format='onnx', dynamic=True)#导出onnx
model.export(format="engine",device=0,simplify=True)#导出tensorrt
# benchmark(model, data='coco8.yaml', imgsz=640, half=False, device=0)


