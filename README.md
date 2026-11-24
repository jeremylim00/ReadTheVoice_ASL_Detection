# ReadTheVoice_ASL_Detection
## 1. Clone the repository 
```bash
https://github.com/jeremylim00/ReadTheVoice_ASL_Detection.git
``` 
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

### Run the MediaPipe Hand Landmarker Test
```bash
python test_hand.py
```




## 3. Collecting the Dataset (21 Hand Landmarks)

Collect the main dataset of **21 hand landmarks** across letters **A to Z** .

### Create the Data Collection Script

Inside the `ASL_Recognition` folder, create a new file named `collect_data.py`.

### Run the Collection Program
```bash
python collect_data.py
```

## 4. Train Random Forest Model

### Create the Training Script

1. Inside the `ASL_Recognition` folder, create a new file named `train_model.py`.

### Run the Training Program
```bash
python train_model.py
```

## 5. Test Trained Model

### Create the Testing Script

1. Inside the `ASL_Recognition` folder, create a new file named `test_model.py`.

### Run the Model Test
```bash
python test_model.py
```
## 6. Train LSTM + CTC for Word Detection

Train a sequence model using **LSTM** and **CTC (Connectionist Temporal Classification)** to recognize multi-letter handshape sequences and assemble words.

### Install Dependencies
```bash
pip install opencv-python mediapipe numpy pandas tensorflow scikit-learn
```
### Collect Word Sequence Data
Create `collect_words.py` inside `ASL_Recognition` and run:

```bash
python collect_words.py
```

### Train the LSTM + CTC Model
Create `train_lstm.py` inside `ASL_Recognition` and run:

```bash
python train_lstm.py
```

### Predict Word Sequences
Create `predit_sequence.py` inside `ASL_Recognition` and run:

```bash
python predit_sequence.py
```
