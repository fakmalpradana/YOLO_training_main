from ultralytics import YOLO

'''
    Bagian Ini Untuk Fresh Training
'''
model = YOLO('yolov8x-seg.yaml')  # build a new model from YAML
model = YOLO('yolov8x-seg.pt')  # load a pretrained model (recommended for training)
model = YOLO('yolov8x-seg.yaml').load('yolov8x.pt')  # build from YAML and transfer weights

# '''
#     Bagian Ini Untuk Pre-Trained
# '''
# model = YOLO('runs/segment/train53/weights/best.pt')


results = model.train(
    data='data.yaml', 
    epochs=10000, 
    imgsz=320, 
    name='train5', 
    # resume=True, 
    amp=False,
    batch=4, 
    patience=0, 
    mask_ratio=1,
    cls=1.0,
)