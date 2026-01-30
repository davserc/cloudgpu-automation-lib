from vast_service import train_with_cheapest_instance

result = train_with_cheapest_instance(
    api_key=None,
    image="davse/flask-yolo-service:latest",
    ports="5000",
    dataset_dst="/work/datasets",
    dataset_gs_uri="gs://unlu-genai-labs-computer_vision_yolo/taco.tar.gz",
    gcp_sa_b64=None,  # usa TRAIN_GCP_SA_B64 del env
    install_gsutil=True,
    train_dataset_url="gs://unlu-genai-labs-computer_vision_yolo/taco.tar.gz",
    run_cmd="bash /work/run_service.sh",
    train_env={
        "DATASET_YAML": "/work/datasets/taco/taco_macro.yaml",
        "MODEL_WEIGHTS": "yolo11s.pt",
        "EPOCHS": "50",
        "IMGSZ": "640",
        "BATCH": "16",
        "DEVICE": "0",
        "PROJECT": "/work/runs",
        "RUN_NAME": "train_y11s_macro_e50",
        "SAVE": "true",
        "PATIENCE": "20",
    },
    artifact_src="/work/runs/train_y11s_macro_e50/weights/best.pt",
    artifact_dst="./",
    log_path="./logs/train.log",
    max_cuda=12.9,
    raise_on_nonzero=False,
)
print(result)
