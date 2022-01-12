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
* Dataset has 7 classes: angry, disgust, fear, happy, sad, surprised, and neutral.

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

* **Notice:** This model will work best if users **do not** wear glasses.

---
### Neural Network Architecture

<p align='middle'><img src="figures/cnn-architecture.jpg" width="50%" height="50%"></p>

---
### Evaluation

<p align='middle'>
  <img src='./figures/accuracy.jpg' width=48% />
  <img src='./figures/loss.jpg' width=48% /> 
</p>
<p align='middle'><img src='./figures/evaluation.jpg' width=50% /></p>