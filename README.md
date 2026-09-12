<h1 style="text-align: center;">Real Time Facial Detection System built using Tensorflow</h1>
<h3 style="text-align: center;">By Ruhan Shafi</h3>

## Project Goal:


The goal of this project is to create an image recognition model that can successfully identify a person's biological sex and whether they are a minor or an adult with high accuracy — classifying faces into 4 categories: Boys, Girls, Man, Woman. This is achieved by training a convolutional neural network using the TensorFlow/Keras library on the publicly available, open-source All-Age-Faces Dataset published by Tsinghua University, then deploying it in a cross-platform desktop app (PySide6) that wraps the trained model with live, multi-person webcam inference (per-person bounding boxes), static image classification, and an in-window PDF viewer explaining the model.

## Tech Stack 

| | |
|---|---|
| Model type | Convolutional Neural Network (CNN), trained from scratch |
| Framework | TensorFlow / Keras |
| Classes | Boys, Girls, Man, Woman |
| Validation accuracy | ~77% |
| Deployment format | TensorFlow Lite (`.tflite`) |
| UI framework | PySide6 |
| Supported platforms | Windows, macOS, Linux |

## Part 1: Model Construction
 
### Dataset
 
The model was trained on the **All-Age-Faces Dataset**, created by Tsinghua
University of Science and Technology — a collection of facial images spanning
a wide range of ages, sorted here into four categories: Boys, Girls, Man,
Woman.
 
### Preparing the data
 
- Images were resized to a fixed **180×180** pixels and loaded in batches of
  **32**.
- The dataset was split **80% training / 20% validation** (`image_dataset_from_directory`
  with `validation_split=0.2`), with the validation set held back purely to
  measure performance on faces the model never trained on.
- A `Rescaling` layer normalizes pixel values from the usual 0–255 range down
  to 0–1, which neural networks generally train more smoothly on.
- The training pipeline applies `.cache()`, `.shuffle(1000)`, and
  `.prefetch(AUTOTUNE)` to keep data loading from bottlenecking training.
### Model architecture
 
Built with Keras's `Sequential` API:
 
```
Rescaling(1./255)
→ Conv2D(16, 3x3, padding="same", activation="relu") → MaxPooling2D()
→ Conv2D(32, 3x3, padding="same", activation="relu") → MaxPooling2D()
→ Conv2D(64, 3x3, padding="same", activation="relu") → MaxPooling2D()
→ Flatten()
→ Dense(128, activation="relu")
→ Dropout(0.2)
→ Dense(4)  # one logit per class
```
 
**Compilation:**
- Optimizer: `Adam`
- Loss: `SparseCategoricalCrossentropy(from_logits=True)`
- Metric: `accuracy`
### Training
 
The model was trained over multiple epochs, with predictions checked against
the true label each batch and the network's weights nudged to reduce error.
 
**First attempt** (10 epochs, no augmentation): training accuracy reached
~98%, but validation accuracy plateaued around 74% and then *worsened* while
training accuracy kept climbing — a textbook case of **overfitting**, where
the model memorizes training examples rather than learning generalizable
patterns.
 
**Final version** (15 epochs, with fixes applied): two changes were made
together to address this —
 
1. **Data augmentation** — random horizontal flips, small rotations, and
   slight zooms applied to training images on the fly, so the model sees more
   variety without needing more raw data.
2. **Dropout (0.2)** — randomly disables 20% of neurons during each training
   step, preventing the network from over-relying on any single feature.
With both in place, training and validation accuracy tracked much more
closely together, and the model reached a final **validation accuracy of
~77%** across the four categories.
 
Some misclassification is expected and not necessarily a flaw: distinguishing
an adult man from a teenage boy, for instance, is a genuinely subjective call
even for a human working from a single photo — most of the model's confusion
occurs between these adjacent, inherently ambiguous categories rather than
across sexes.
