# ReadTheVoice_ASL_Detection

## 2. MediaPipe Setup

1. Create an empty folder named `ASL_Recognition`.
2. Download `hand_landmarker.task` and move it inside the `ASL_Recognition` folder.
3. Open **Anaconda Prompt** and run the following commands:

### Create Python Environment

```bash
conda create -n asl_project python=3.11 -y
```

### Activate the Environment

```bash
conda activate asl_project
```

### Install Required Libraries

```bash
pip install mediapipe opencv-python numpy pandas scikit-learn joblib
```

### Create Project File

Open **VS Code**, select the `ASL_Recognition` folder, and create a new file named `test_hand.py`. 
