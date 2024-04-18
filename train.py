from ultralytics import YOLO

'''
    Bagian Ini Untuk Fresh Training
'''
# model = YOLO('yolov8n-seg.yaml')  # build a new model from YAML
# model = YOLO('yolov8n-seg.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8n-seg.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

'''
    Bagian Ini Untuk Pre-Trained
'''
model = YOLO('runs/segment/train02/weights/best.pt')


if __name__ == '__main__':
    results = model.train(
        data='data.yaml', 
        epochs=5, 
        imgsz=320, 
        name='train02', 
        # resume=True, 
        amp=False,
        batch=2, 
        patience=0, 
        mask_ratio=2,
        cls=1.0,
        workers=2,
    )