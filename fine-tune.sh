yolo detect train \
  model=best.pt \
  data=dataset/data.yaml \
  epochs=50 \
  imgsz=960 \
  lr0=0.0005 \
  batch=16 \
  hsv_h=0.015 \
  hsv_s=0.7 \
  hsv_v=0.4 \
  translate=0.1 \
  scale=0.5 \
  fliplr=0.5 \
  mosaic=1.0 \
  mixup=0.1