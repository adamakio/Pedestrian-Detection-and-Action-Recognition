from ultralytics import YOLO

def predict_model(model_path, project, name):
    model = YOLO(model_path)

    model.predict(
        source="yolo_0.1/images/test",   # your 10% test set images
        imgsz=640,
        device="mps",
        conf=0.25,                       # tweak if you want more/less boxes
        save=True,                       # save annotated images
        save_conf=True,                  # show confidence scores
        save_txt=False,                  # set True if you also want YOLO txt outputs
        project=project,                 # output root
        name=name,                       # subfolder name
        exist_ok=True
    )

if __name__ == "__main__":
    predict_model(
        "runs/detect/titan_person_11n/weights/best.pt",
        "runs/predict",
        "titan_11n_test_pred_finetuned"
    )



