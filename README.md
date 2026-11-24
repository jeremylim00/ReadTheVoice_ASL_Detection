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




## 3. Collecting the Dataset (21 Hand Landmarks)

Collect the main dataset of **21 hand landmarks** across letters **A to Z** .

### Create the Data Collection Script

Inside the `ASL_Recognition` folder, create a new file named `collect_data.py`.

### Run the Collection Program

Open **Anaconda Prompt**, activate your environment, navigate to the project directory, and run the script:

```bash
python collect_data.py
```

### Record Coordinate Samples

1. Show the ASL sign for letter **A** in front of the camera.
2. Press the **A** key on your keyboard to record **20 frames** and save **one labelled sample**.
3. Slightly change your hand position or angle, then press **A** again to collect another sample.
4. Repeat this process across the letters **A to Z** to build your dataset.
5. Press 'q' to quit the program.
