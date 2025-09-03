# Digit-Recognition

🖊️ Handwritten Digit Recognition (MNIST + CNN)
This project is a handwritten digit recognition system built using Convolutional Neural Networks (CNNs) with TensorFlow/Keras.
It can predict digits (0–9) from input images and has been deployed using Streamlit.

🔥 Features
1. Trained on the MNIST dataset (60,000 training, 10,000 test images).
2. Uses a CNN model for high accuracy digit classification.
3. Allows users to upload their own digit images for prediction.
4. Deployed with Streamlit for an interactive web interface.
5. Model persistence with .h5 file (no need to retrain every time).

⚙️ Tech Stack
Python
TensorFlow/Keras
NumPy, Pandas, Matplotlib
OpenCV (for image preprocessing)
Streamlit (for deployment)

🚀 Installation & Usage
1. Clone the repository
git clone https://github.com/adirawat-cmd/Digit-Recognition
2. Install dependencies
pip install -r requirements.txt
3. Run the Streamlit app
streamlit run app.py
4. Upload an image of a digit (28x28 or any size, it’ll be resized automatically) and see the prediction

🌐 Deployment
The project is deployed using Streamlit Cloud.
 Link: https://digit-recognition-mzsew7ps2rcvymrzsqpazm.streamlit.app/

 
