from ultralytics import YOLO
import torch

def verify():
    # Load the model from the newly created YAML
    try:
        model = YOLO("ultralytics/cfg/models/11/yolo11-mamba.yaml")
        print("Successfully loaded YOLO11-Mamba model!")
        
        # Print model info
        # model.info()
        
        # Test a forward pass with a dummy input
        img = torch.randn(1, 3, 640, 640)
        # We use model.model because YOLO() is a wrapper
        # results = model.model(img)
        # print("Forward pass successful!")
        
    except Exception as e:
        print(f"Failed to load or run the model: {e}")
        print("\nNote: This might be due to missing 'mamba-ssm' or 'causal-conv1d' packages in the current environment.")

if __name__ == "__main__":
    verify()
