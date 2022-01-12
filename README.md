## Facial Expression Recognition

![Face Emotion Detector](./figures/demo.gif)

### Prerequisites
* Python 3.6
* Tensorflow 2.5.0
* Keras 2.6.0

*Tested on Windows 10 - 64 bit*

---
### Dataset
* [FER-2013](https://www.kaggle.com/msambare/fer2013) from Kaggle.
* Dataset has 7 classes: angry, disgust, fear, happy, sad, surprised, neutral.

---
### How to use
* Step 1: Training model. *Skip if using the **attached model***.
    ```
    python train.py
    ```
* Step 2: Testing model through your webcam.
    ```
    python test.py
    ```

* **Notice:** This model will work best if users do not wear glasses.

---
### Evaluation

<p align='middle'>
  <img src='./figures/accuracy.jpg' width=48% />
  <img src='./figures/loss.jpg' width=48% /> 
</p>
<p align='middle'><img src='./figures/evaluation.jpg' width=50% /></p>

* The architecture of my neural network is as below

   <img src="figures/cnn-architecture.jpg" width="50%" height="50%">

---
### References
2. [Priya Dwivedi. Face-and-Emotion-Detection.](https://github.com/priya-dwivedi/face_and_emotion_detection/blob/master/src/EmotionDetector_v2.ipynb)
3. [Akamadan. Emotion-Detection-CNN.](https://github.com/akmadan/Emotion_Detection_CNN/blob/main/emotion-classification-cnn-using-keras.ipynb)
4. [Great Learning (2020). Emotion Detection using Python.](https://www.youtube.com/watch?v=m0fWjP3yIEo&ab_channel=GreatLearning)
5. [The AI Guy (2020). How to Use Webcam In Google Colab for Images and Video.](https://www.youtube.com/watch?v=YjWh7QvVH60&ab_channel=TheAIGuy)