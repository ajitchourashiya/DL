{
 "cells": [
  {
   "cell_type": "raw",
   "id": "4494e526",
   "metadata": {},
   "source": [
    "# Title of Assignment-2:\n",
    "#     Implementing Feedforward neural networks with Keras and TensorFlow\n",
    "#     a. Import the necessary packages\n",
    "#     b. Load the training and testing data (MNIST)\n",
    "#     c. Define the network architecture using Keras\n",
    "#     d. Train the model using SGD\n",
    "#     e. Evaluate the network\n",
    "#     f. Plot the training loss and accuracy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "3f7bf568",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\ProgramData\\Anaconda3\\lib\\site-packages\\scipy\\__init__.py:155: UserWarning: A NumPy version >=1.18.5 and <1.25.0 is required for this version of SciPy (detected version 1.26.0\n",
      "  warnings.warn(f\"A NumPy version >={np_minversion} and <{np_maxversion}\"\n"
     ]
    }
   ],
   "source": [
    "#importing necessary libraries\n",
    "import tensorflow as tf\n",
    "from tensorflow import keras"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "d97cdf97",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import random\n",
    "get_ipython().run_line_magic('matplotlib', 'inline')"
   ]
  },
  {
   "cell_type": "raw",
   "id": "56995a78",
   "metadata": {},
   "source": [
    "# # Loading and preparing the data\n",
    "# MNIST stands for “Modified National Institute of Standards and Technology”. \n",
    "# It is a dataset of 70,000 handwritten images. Each image is of 28x28 pixels \n",
    "# i.e. about 784 features. Each feature represents only one pixel’s intensity i.e. from 0(white) to 255(black). \n",
    "# This database is further divided into 60,000 training and 10,000 testing images."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "bed5e471",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz\n",
      "11490434/11490434 [==============================] - 80s 7us/step\n"
     ]
    }
   ],
   "source": [
    "#import dataset and split into train and test data\n",
    "mnist = tf.keras.datasets.mnist\n",
    "(x_train, y_train), (x_test, y_test) = mnist.load_data()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "98703353",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "60000"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#to see length of training dataset\n",
    "len(x_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "9aa2fe58",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "10000"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "##to see length of testing dataset\n",
    "len(x_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "1c9bf5d9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(60000, 28, 28)"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#shape of training dataset  60,000 images having 28*28 size\n",
    "x_train.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "cd63c95b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(10000, 28, 28)"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#shape of testing dataset  10,000 images having 28*28 size\n",
    "x_test.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "5f2ab9ab",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.image.AxesImage at 0x14d55eadd30>"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 480x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#to see how first image look\n",
    "plt.matshow(x_train[0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "4af93803",
   "metadata": {},
   "outputs": [],
   "source": [
    "#normalize the images by scaling pixel intensities to the range 0,1\n",
    "\n",
    "x_train = x_train / 255\n",
    "x_test = x_test / 255\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "1870eae1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        ],\n",
       "       [0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        ],\n",
       "       [0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        ],\n",
       "       [0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        ],\n",
       "       [0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        ],\n",
       "       [0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.01176471, 0.07058824, 0.07058824,\n",
       "        0.07058824, 0.49411765, 0.53333333, 0.68627451, 0.10196078,\n",
       "        0.65098039, 1.        , 0.96862745, 0.49803922, 0.        ,\n",
       "        0.        , 0.        , 0.        ],\n",
       "       [0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.11764706, 0.14117647,\n",
       "        0.36862745, 0.60392157, 0.66666667, 0.99215686, 0.99215686,\n",
       "        0.99215686, 0.99215686, 0.99215686, 0.88235294, 0.6745098 ,\n",
       "        0.99215686, 0.94901961, 0.76470588, 0.25098039, 0.        ,\n",
       "        0.        , 0.        , 0.        ],\n",
       "       [0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.19215686, 0.93333333, 0.99215686,\n",
       "        0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.99215686,\n",
       "        0.99215686, 0.99215686, 0.98431373, 0.36470588, 0.32156863,\n",
       "        0.32156863, 0.21960784, 0.15294118, 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        ],\n",
       "       [0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.07058824, 0.85882353, 0.99215686,\n",
       "        0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.77647059,\n",
       "        0.71372549, 0.96862745, 0.94509804, 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        ],\n",
       "       [0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.31372549, 0.61176471,\n",
       "        0.41960784, 0.99215686, 0.99215686, 0.80392157, 0.04313725,\n",
       "        0.        , 0.16862745, 0.60392157, 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        ],\n",
       "       [0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.05490196,\n",
       "        0.00392157, 0.60392157, 0.99215686, 0.35294118, 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        ],\n",
       "       [0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.54509804, 0.99215686, 0.74509804, 0.00784314,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        ],\n",
       "       [0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.04313725, 0.74509804, 0.99215686, 0.2745098 ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        ],\n",
       "       [0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.1372549 , 0.94509804, 0.88235294,\n",
       "        0.62745098, 0.42352941, 0.00392157, 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        ],\n",
       "       [0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.31764706, 0.94117647,\n",
       "        0.99215686, 0.99215686, 0.46666667, 0.09803922, 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        ],\n",
       "       [0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.17647059,\n",
       "        0.72941176, 0.99215686, 0.99215686, 0.58823529, 0.10588235,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        ],\n",
       "       [0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.0627451 , 0.36470588, 0.98823529, 0.99215686, 0.73333333,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        ],\n",
       "       [0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.97647059, 0.99215686, 0.97647059,\n",
       "        0.25098039, 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        ],\n",
       "       [0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.18039216,\n",
       "        0.50980392, 0.71764706, 0.99215686, 0.99215686, 0.81176471,\n",
       "        0.00784314, 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        ],\n",
       "       [0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.15294118, 0.58039216, 0.89803922,\n",
       "        0.99215686, 0.99215686, 0.99215686, 0.98039216, 0.71372549,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        ],\n",
       "       [0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.09411765, 0.44705882, 0.86666667, 0.99215686, 0.99215686,\n",
       "        0.99215686, 0.99215686, 0.78823529, 0.30588235, 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        ],\n",
       "       [0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.09019608, 0.25882353,\n",
       "        0.83529412, 0.99215686, 0.99215686, 0.99215686, 0.99215686,\n",
       "        0.77647059, 0.31764706, 0.00784314, 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        ],\n",
       "       [0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.07058824, 0.67058824, 0.85882353, 0.99215686,\n",
       "        0.99215686, 0.99215686, 0.99215686, 0.76470588, 0.31372549,\n",
       "        0.03529412, 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        ],\n",
       "       [0.        , 0.        , 0.        , 0.        , 0.21568627,\n",
       "        0.6745098 , 0.88627451, 0.99215686, 0.99215686, 0.99215686,\n",
       "        0.99215686, 0.95686275, 0.52156863, 0.04313725, 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        ],\n",
       "       [0.        , 0.        , 0.        , 0.        , 0.53333333,\n",
       "        0.99215686, 0.99215686, 0.99215686, 0.83137255, 0.52941176,\n",
       "        0.51764706, 0.0627451 , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        ],\n",
       "       [0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        ],\n",
       "       [0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        ],\n",
       "       [0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "        0.        , 0.        , 0.        ]])"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "x_train[0]"
   ]
  },
  {
   "cell_type": "raw",
   "id": "8cc1845a",
   "metadata": {},
   "source": [
    "#Define the network architecture using Keras\n",
    "# # Creating the model\n",
    "# # The ReLU function is one of the most popular activation functions. \n",
    "# It stands for “rectified linear unit”. Mathematically this function is defined as:\n",
    "# y = max(0,x)The ReLU function returns “0” if the input is negative and is linear if \n",
    "# the input is positive.\n",
    "# # The softmax function is another activation function. \n",
    "# It changes input values into values that reach from 0 to 1."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "9871b4f0",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = keras.Sequential([\n",
    "    keras.layers.Flatten(input_shape=(28, 28)),\n",
    "    keras.layers.Dense(128, activation='relu'),\n",
    "    keras.layers.Dense(10, activation='softmax')\n",
    "])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "65b3f7df",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"sequential\"\n",
      "_________________________________________________________________\n",
      " Layer (type)                Output Shape              Param #   \n",
      "=================================================================\n",
      " flatten (Flatten)           (None, 784)               0         \n",
      "                                                                 \n",
      " dense (Dense)               (None, 128)               100480    \n",
      "                                                                 \n",
      " dense_1 (Dense)             (None, 10)                1290      \n",
      "                                                                 \n",
      "=================================================================\n",
      "Total params: 101770 (397.54 KB)\n",
      "Trainable params: 101770 (397.54 KB)\n",
      "Non-trainable params: 0 (0.00 Byte)\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "model.summary()\n",
    "\n",
    "# # Compile the model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "ccc2689e",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.compile(optimizer='sgd',\n",
    "              loss='sparse_categorical_crossentropy',\n",
    "              metrics=['accuracy'])\n",
    "\n",
    "# # Train the model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "2c05d9b2",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/10\n",
      "1875/1875 [==============================] - 13s 6ms/step - loss: 0.6530 - accuracy: 0.8328 - val_loss: 0.3547 - val_accuracy: 0.9032\n",
      "Epoch 2/10\n",
      "1875/1875 [==============================] - 12s 6ms/step - loss: 0.3358 - accuracy: 0.9049 - val_loss: 0.2920 - val_accuracy: 0.9172\n",
      "Epoch 3/10\n",
      "1875/1875 [==============================] - 14s 7ms/step - loss: 0.2880 - accuracy: 0.9183 - val_loss: 0.2593 - val_accuracy: 0.9274\n",
      "Epoch 4/10\n",
      "1875/1875 [==============================] - 13s 7ms/step - loss: 0.2583 - accuracy: 0.9275 - val_loss: 0.2370 - val_accuracy: 0.9323\n",
      "Epoch 5/10\n",
      "1875/1875 [==============================] - 26s 14ms/step - loss: 0.2355 - accuracy: 0.9342 - val_loss: 0.2199 - val_accuracy: 0.9374\n",
      "Epoch 6/10\n",
      "1875/1875 [==============================] - 25s 14ms/step - loss: 0.2174 - accuracy: 0.9395 - val_loss: 0.2059 - val_accuracy: 0.9413\n",
      "Epoch 7/10\n",
      "1875/1875 [==============================] - 25s 13ms/step - loss: 0.2021 - accuracy: 0.9437 - val_loss: 0.1936 - val_accuracy: 0.9449\n",
      "Epoch 8/10\n",
      "1875/1875 [==============================] - 24s 13ms/step - loss: 0.1888 - accuracy: 0.9470 - val_loss: 0.1837 - val_accuracy: 0.9469\n",
      "Epoch 9/10\n",
      "1875/1875 [==============================] - 24s 13ms/step - loss: 0.1774 - accuracy: 0.9505 - val_loss: 0.1736 - val_accuracy: 0.9493\n",
      "Epoch 10/10\n",
      "1875/1875 [==============================] - 33s 18ms/step - loss: 0.1673 - accuracy: 0.9531 - val_loss: 0.1633 - val_accuracy: 0.9513\n"
     ]
    }
   ],
   "source": [
    "history=model.fit(x_train, y_train,validation_data=(x_test,y_test),epochs=10)\n",
    "\n",
    "# # Evaluate the model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "24a09ef4",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "313/313 [==============================] - 3s 9ms/step - loss: 0.1633 - accuracy: 0.9513\n",
      "Loss=0.163\n",
      "Accuracy=0.951\n"
     ]
    }
   ],
   "source": [
    "test_loss,test_acc=model.evaluate(x_test,y_test)\n",
    "print(\"Loss=%.3f\" %test_loss)\n",
    "print(\"Accuracy=%.3f\" %test_acc)\n",
    "\n",
    "# # Making Prediction on New Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "3de71747",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "n=random.randint(0,9999)\n",
    "plt.imshow(x_test[n])\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "bb262586",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "313/313 [==============================] - 4s 9ms/step\n",
      "Handwritten number in the image is= 9\n"
     ]
    }
   ],
   "source": [
    "#we use predict() on new data\n",
    "predicted_value=model.predict(x_test)\n",
    "print(\"Handwritten number in the image is= %d\" %np.argmax(predicted_value[n]))\n",
    "\n",
    "# # Plot graph for Accuracy and Loss"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "fbd5316c",
   "metadata": {},
   "outputs": [],
   "source": [
    "get_ipython().run_line_magic('pinfo2', 'history.history')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "1df00a8c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "history.history.keys()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "34f171b4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAkAAAAHFCAYAAAAaD0bAAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8qNh9FAAAACXBIWXMAAA9hAAAPYQGoP6dpAABknElEQVR4nO3de1iUZf7H8fdwPoOAIAIC5llTE41ErdTSzEzLLWtLs6zdylKytjKzg5WW/bR2My3bzNyttOywbbkVeSjNPKHm+SyiCCIeAEFOM8/vj5HREY+IzMB8Xtc1l84zz8x8R8z5dN/f575NhmEYiIiIiLgQN0cXICIiIlLTFIBERETE5SgAiYiIiMtRABIRERGXowAkIiIiLkcBSERERFyOApCIiIi4HAUgERERcTkKQCIiIuJyFIBEpEalp6djMpmYOXPmRT930aJFmEwmFi1aVO11iYhrUQASERERl6MAJCLiYMePH0fbMorULAUgERfz0ksvYTKZWLduHXfccQfBwcGEhoYyatQoysvL2bp1KzfddBOBgYHEx8czceLESq+RkZHBvffeS0REBN7e3rRs2ZJJkyZhsVjsztu/fz933nkngYGBBAcHM2jQILKzs89Y16pVq7j11lsJDQ3Fx8eHq666is8//7xKn/HgwYM8+uijtGrVioCAACIiIujRoweLFy+udG5JSQnjxo2jZcuW+Pj4EBYWRvfu3Vm6dKntHIvFwjvvvEP79u3x9fUlJCSEa665hm+//dZ2jslk4qWXXqr0+vHx8QwdOtR2f+bMmZhMJn766SceeOAB6tevj5+fHyUlJezYsYP777+fpk2b4ufnR3R0NP369WP9+vWVXvfo0aM8+eSTNG7cGG9vbyIiIrj55pvZsmULhmHQtGlTevfuXel5x44dIzg4mOHDh1/kn6pI3eLh6AJExDHuvPNO7r33Xv7617+SmprKxIkTKSsr4+eff+bRRx/lqaee4tNPP+WZZ56hSZMm3H777YA1XCQnJ1NaWsorr7xCfHw83333HU899RQ7d+5k6tSpgHVU44YbbmD//v1MmDCBZs2a8f333zNo0KBKtSxcuJCbbrqJpKQk3nvvPYKDg5k9ezaDBg2iqKjILkBciMOHDwPw4osv0qBBA44dO8bXX3/N9ddfz/z587n++usBKC8vp0+fPixevJiUlBR69OhBeXk5y5YtIyMjg+TkZACGDh3Kv//9b4YNG8a4cePw8vJi9erVpKenV+0PH3jggQfo27cv//rXvygsLMTT05P9+/cTFhbG66+/Tv369Tl8+DAff/wxSUlJrFmzhubNmwNQUFBA165dSU9P55lnniEpKYljx47x66+/kpWVRYsWLXj88cdJSUlh+/btNG3a1Pa+s2bNIj8/XwFIxBARl/Liiy8agDFp0iS74+3btzcA46uvvrIdKysrM+rXr2/cfvvttmPPPvusARjLly+3e/4jjzximEwmY+vWrYZhGMa0adMMwPjPf/5jd95DDz1kAMZHH31kO9aiRQvjqquuMsrKyuzOveWWW4yoqCjDbDYbhmEYCxcuNABj4cKFF/WZy8vLjbKyMqNnz57GbbfdZjs+a9YsAzA++OCDsz73119/NQBjzJgx53wPwHjxxRcrHY+LizPuu+8+2/2PPvrIAIwhQ4ZcUN2lpaVG06ZNjSeeeMJ2fNy4cQZgpKamnvW5+fn5RmBgoDFy5Ei7461atTK6d+9+3vcWqes0BSbiom655Ra7+y1btsRkMtGnTx/bMQ8PD5o0acKePXtsxxYsWECrVq24+uqr7Z4/dOhQDMNgwYIFgHVUJzAwkFtvvdXuvD//+c9293fs2MGWLVu45557AOuoTMXt5ptvJisri61bt17053vvvffo0KEDPj4+eHh44Onpyfz589m8ebPtnP/973/4+PjwwAMPnPV1/ve//wFU+4jJwIEDKx0rLy9n/PjxtGrVCi8vLzw8PPDy8mL79u2V6m7WrBk33HDDWV8/MDCQ+++/n5kzZ1JYWAhYf3abNm3iscceq9bPIlIbKQCJuKjQ0FC7+15eXvj5+eHj41PpeHFxse3+oUOHiIqKqvR6DRs2tD1e8WtkZGSl8xo0aGB3/8CBAwA89dRTeHp62t0effRRAHJzcy/qs02ePJlHHnmEpKQkvvzyS5YtW8bKlSu56aabOH78uO28gwcP0rBhQ9zczv5P4cGDB3F3d69U96U605/hqFGjGDt2LAMGDOC///0vy5cvZ+XKlbRr165S3TExMed9j8cff5yCggI++eQTAKZMmUJMTAz9+/evvg8iUkupB0hELkpYWBhZWVmVju/fvx+A8PBw23krVqyodN7pTdAV548ePdrWZ3S6it6XC/Xvf/+b66+/nmnTptkdLygosLtfv359lixZgsViOWsIql+/Pmazmezs7DOGlgre3t6UlJRUOl4RCE9nMpnOWPeQIUMYP3683fHc3FxCQkLsatq3b99Za6nQpEkT+vTpw7vvvkufPn349ttvefnll3F3dz/vc0XqOo0AichF6dmzJ5s2bWL16tV2x2fNmoXJZKJ79+4AdO/enYKCArsrpQA+/fRTu/vNmzenadOm/PHHH3Ts2PGMt8DAwIuq0WQy4e3tbXds3bp1/P7773bH+vTpQ3Fx8TkXZayYEjw9TJ0uPj6edevW2R1bsGABx44du6S6v//+ezIzMyvVtG3bNtt047mMHDmSdevWcd999+Hu7s5DDz10wfWI1GUaARKRi/LEE08wa9Ys+vbty7hx44iLi+P7779n6tSpPPLIIzRr1gyAIUOG8NZbbzFkyBBee+01mjZtyrx58/jxxx8rveb7779Pnz596N27N0OHDiU6OprDhw+zefNmVq9ezRdffHFRNd5yyy288sorvPjii1x33XVs3bqVcePGkZCQQHl5ue28u+++m48++oiHH36YrVu30r17dywWC8uXL6dly5bcdddddOvWjcGDB/Pqq69y4MABbrnlFry9vVmzZg1+fn48/vjjAAwePJixY8fywgsvcN1117Fp0yamTJlCcHDwRdU9c+ZMWrRoQdu2bUlLS+PNN9+sNN2VkpLCnDlz6N+/P88++yxXX301x48f55dffuGWW26xhVCAG2+8kVatWrFw4ULb0gUigq4CE3E1FVeBHTx40O74fffdZ/j7+1c6/7rrrjNat25td2zPnj3Gn//8ZyMsLMzw9PQ0mjdvbrz55pu2q7Uq7Nu3zxg4cKAREBBgBAYGGgMHDjSWLl1a6SowwzCMP/74w7jzzjuNiIgIw9PT02jQoIHRo0cP47333rOdc6FXgZWUlBhPPfWUER0dbfj4+BgdOnQwvvnmG+O+++4z4uLi7M49fvy48cILLxhNmzY1vLy8jLCwMKNHjx7G0qVLbeeYzWbjrbfeMtq0aWN4eXkZwcHBRufOnY3//ve/du/59NNPG7GxsYavr69x3XXXGWvXrj3rVWArV66sVPeRI0eMYcOGGREREYafn5/RtWtXY/HixcZ1111nXHfddZXOHTlypNGoUSPD09PTiIiIMPr27Wts2bKl0uu+9NJLBmAsW7bsnH9uIq7EZBhaflREpC7r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pR6glStXsnz58krHly9fzqpVqy65KKk5hSXlrK/o/0lQ/4+IiLiGKgWg4cOHs3fv3krHMzMzGT58+CUXJTUnbc8Ryi0G0SG+xIaq/0dERFxDlQLQpk2b6NChQ6XjV111FZs2bbrkoqTmaPpLRERcUZUCkLe3NwcOHKh0PCsrCw+PKrcViQNoA1QREXFFVQpAN954I6NHjyYvL8927OjRozz33HPceOON1VacXF5FpSfX/9EIkIiIuJIqDddMmjSJa6+9lri4OK666ioA1q5dS2RkJP/617+qtUC5fNT/IyIirqpKASg6Opp169bxySef8Mcff+Dr68v999/P3XffjaenZ3XXKJdJxfSXVn8WERFXU+WGHX9/f7p27UqjRo0oLS0F4H//+x9gXShRnJ9tA1RNf4mIiIupUgDatWsXt912G+vXr8dkMmEYBiaTyfa42WyutgLl8igqLeePvUcB6KwAJCIiLqZKTdAjR44kISGBAwcO4Ofnx4YNG/jll1/o2LEjixYtquYS5XJYveeorf8npp6vo8sRERGpUVUaAfr9999ZsGAB9evXx83NDXd3d7p27cqECRMYMWIEa9asqe46pZrZ+n8SQu1G70RERFxBlUaAzGYzAQEBAISHh7N//34A4uLi2Lp1a/VVJ5eNFkAUERFXVqURoDZt2rBu3ToaN25MUlISEydOxMvLi+nTp9O4cePqrlGq2fFSM3/sOwooAImIiGuqUgB6/vnnKSwsBODVV1/llltuoVu3boSFhTFnzpxqLVCq3+qMI5SZDRoG+xAbqv4fERFxPVWaAuvduze33347AI0bN2bTpk3k5uaSk5NDjx49Luq1pk6dSkJCAj4+PiQmJrJ48eJznv/JJ5/Qrl07/Pz8iIqK4v777+fQoUN257z99ts0b94cX19fYmNjeeKJJyguLr64D1mHnVz/J0z9PyIi4pKqFIDOJDT04ptp58yZQ0pKCmPGjGHNmjV069aNPn36kJGRccbzlyxZwpAhQxg2bBgbN27kiy++YOXKlTz44IO2cz755BOeffZZXnzxRTZv3syHH37InDlzGD169CV9vrpE+3+JiIirq7YAVBWTJ09m2LBhPPjgg7Rs2ZK3336b2NhYpk2bdsbzly1bRnx8PCNGjCAhIYGuXbvy17/+lVWrVtnO+f333+nSpQt//vOfiY+Pp1evXtx9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BLBYLHTs2JHx48cDcNVVV7Fx40amTZumAOQAc+bM4d///jeffvoprVu3Zu3ataSkpNCwYUPuu+8+R5dXaykA1YDw8HDc3d0rjfbk5ORUGhWSmvX444/z7bff8uuvvxITE+PoclxSWloaOTk5JCYm2o6ZzWZ+/fVXpkyZQklJCe7u7g6s0PVERUXRqlUru2MtW7bkyy+/dFBFru1vf/sbzz77LHfddRcAV155JXv27GHChAkKQJdAPUA1wMvLi8TERFJTU+2Op6amkpyc7KCqXJthGDz22GN89dVXLFiwgISEBEeX5LJ69uzJ+vXrWbt2re3WsWNH7rnnHtauXavw4wBdunSptCzEtm3biIuLc1BFrq2oqAg3N/uva3d3d10Gf4k0AlRDRo0axeDBg+nYsSOdO3dm+vTpZGRk8PDDDzu6NJc0fPhwPv30U/7zn/8QGBhoG50LDg7G19fXwdW5lsDAwEq9V/7+/oSFhakny0GeeOIJkpOTGT9+PHfeeScrVqxg+vTpTJ8+3dGluaR+/frx2muv0ahRI1q3bs2aNWuYPHkyDzzwgKNLq9V0GXwNmjp1KhMnTiQrK4s2bdrw1ltv6ZJrBzlb79VHH33E0KFDa7YYqeT666/XZfAO9t133zF69Gi2b99OQkICo0aN4qGHHnJ0WS6poKCAsWPH8vXXX5OTk0PDhg25++67eeGFF/Dy8nJ0ebWWApCIiIi4HPUAiYiIiMtRABIRERGXowAkIiIiLkcBSERERFyOApCIiIi4HAUgERERcTkKQCIiIuJyFIBERC7AokWLMJlMHD161NGliEg1UAASERERl6MAJCIiIi5HAUhEagXDMJg4cSKNGzfG19eXdu3aMXfuXODk9NT3339Pu3bt8PHxISkpifXr19u9xpdffknr1q3x9vYmPj6eSZMm2T1eUlLC008/TWxsLN7e3jRt2pQPP/zQ7py0tDQ6duyIn58fycnJlXZNF5HaQQFIRGqF559/no8++ohp06axceNGnnjiCe69915++eUX2zl/+9vf+L//+z9WrlxJREQEt956K2VlZYA1uNx5553cddddrF+/npdeeomxY8cyc+ZM2/OHDBnC7Nmz+cc//sHmzZt57733CAgIsKtjzJgxTJo0iVWrVuHh4aEduUVqKW2GKiJOr7CwkPDwcBYsWEDnzp1txx988EGKior4y1/+Qvfu3Zk9ezaDBg0C4PDhw8TExDBz5kzuvPNO7rnnHg4ePMhPP/1ke/7TTz/N999/z8aNG9m2bRvNmzcnNTWVG264oVINixYtonv37vz888/07NkTgHnz5tG3b1+OHz+Oj4/PZf5TEJHqpBEgEXF6mzZtori4mBtvvJGAgADbbdasWezcudN23qnhKDQ0lObNm7N582YANm/eTJcuXexet0uXLmzfvh2z2czatWtxd3fnuuuuO2ctbdu2tf0+KioKgJycnEv+jCJSszwcXYCIyPlYLBYAvv/+e6Kjo+0e8/b2tgtBpzOZTIC1h6ji9xVOHQD39fW9oFo8PT0rvXZFfSJSe2gESEScXqtWrfD29iYjI4MmTZrY3WJjY23nLVu2zPb7I0eOsG3bNlq0aGF7jSVLlti97tKlS2nWrBnu7u5ceeWVWCwWu54iEam7NAIkIk4vMDCQp556iieeeAKLxULXrl3Jz89n6dKlBAQEEBcXB8C4ceMICwsjMjKSMWPGEB4ezoABAwB48skn6dSpE6+88gqDBg3i999/Z8qUKUydOhWA+Ph47rvvPh544AH+8Y9/0K5dO/bs2UNOTg533nmnoz66iFwmCkAiUiu88sorREREMGHCBHbt2kVISAgdOnTgueees01Bvf7664wcOZLt27fTrl07vv32W7y8vADo0KEDn3/+OS+88AKvvPIKUVFRjBs3jqFDh9reY9q0aTz33HM8+uijHDp0iEaNGvHcc8854uOKyGWmq8BEpNaruELryJEjhISEOLocEakF1AMkIiIiLkcBSERERFyOpsBERETE5WgESERERFyOApCIiIi4HAUgERERcTkKQCIiIuJyFIBERETE5SgAiYiIiMtRABIRERGXowAkIiIiLkcBSERERFzO/wOFHgdm+J3/NwAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot(history.history['accuracy'])\n",
    "plt.plot(history.history['val_accuracy'])\n",
    "plt.title('model accuracy')\n",
    "plt.ylabel('accuracy')\n",
    "plt.xlabel('epoch')\n",
    "plt.legend(['Train', 'Validation'], loc='upper left')\n",
    "plt.show()\n",
    "\n",
    "# graph representing the model’s accuracy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "475176af",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAjcAAAHFCAYAAAAOmtghAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8qNh9FAAAACXBIWXMAAA9hAAAPYQGoP6dpAABX7klEQVR4nO3deXhU5fnG8e9k3xOSkI1s7LssibK58VNRFJWqBRVBq1StS0Vq3W0VFyxqpa2CpS5U64KKtu6CCogiKDtC2JcESAhkX8g2c35/TDJhSBgDhJzJzP25rrlIzpw584xRcvue531fi2EYBiIiIiIewsfsAkRERERak8KNiIiIeBSFGxEREfEoCjciIiLiURRuRERExKMo3IiIiIhHUbgRERERj6JwIyIiIh5F4UZEREQ8isKNiLi93bt3Y7FYmDt37nG/dvHixVgsFhYvXtwq54mI+1O4EREREY+icCMiIiIeReFGRH7Ro48+isViYf369fz6178mMjKS6Ohopk6dSl1dHVu2bOGiiy4iPDyc9PR0ZsyY0eQa2dnZXHfddcTFxREYGEjv3r157rnnsNlsTuft37+fcePGER4eTmRkJOPHjycvL6/ZulauXMlll11GdHQ0QUFBDBo0iHfffbdVP/tHH33EsGHDCAkJITw8nAsuuIAffvjB6ZyDBw9y8803k5KSQmBgIB07dmTEiBF89dVXjnPWrFnDmDFjHJ8/KSmJSy65hL1797ZqvSICfmYXICLtx7hx47juuuu45ZZbWLhwITNmzKC2tpavvvqK2267jXvuuYe33nqL++67j27dunHFFVcA9l/+w4cPp6amhscff5z09HQ++eQT7rnnHnbs2MGsWbMAOHz4MOeffz779+9n+vTp9OjRg08//ZTx48c3qWXRokVcdNFFDBkyhJdeeonIyEjeeecdxo8fT2VlJTfccMNJf9633nqLCRMmMGrUKN5++22qq6uZMWMG5557Ll9//TVnnnkmABMnTmT16tU8+eST9OjRg+LiYlavXk1BQQEAFRUVXHDBBXTu3JkXX3yR+Ph48vLyWLRoEWVlZSddp4gcxRAR+QV//vOfDcB47rnnnI4PHDjQAIwPPvjAcay2ttbo2LGjccUVVziO3X///QZgrFixwun1v/vd7wyLxWJs2bLFMAzDmD17tgEY//vf/5zO++1vf2sAxmuvveY41qtXL2PQoEFGbW2t07ljxowxEhMTDavVahiGYSxatMgAjEWLFrn8jEefZ7VajaSkJKN///6OaxmGYZSVlRlxcXHG8OHDHcfCwsKMKVOmHPPaK1euNADjv//9r8saRKR16LaUiLTYmDFjnL7v3bs3FouF0aNHO475+fnRrVs39uzZ4zj2zTff0KdPH8444wyn199www0YhsE333wD2EdjwsPDueyyy5zOu/baa52+3759O5s3b2bChAkA1NXVOR4XX3wxubm5bNmy5aQ+65YtW9i/fz8TJ07Ex6fxr8qwsDCuvPJKli9fTmVlJQBnnHEGc+fO5YknnmD58uXU1tY6Xatbt2506NCB++67j5deeolNmzadVG0i4prCjYi0WHR0tNP3AQEBhISEEBQU1OR4VVWV4/uCggISExObXC8pKcnxfMOf8fHxTc5LSEhw+v7AgQMA3HPPPfj7+zs9brvtNgAOHTp0vB/PSUNNx6rbZrNRVFQEwLx587j++ut5+eWXGTZsGNHR0UyaNMnRKxQZGcmSJUsYOHAgDz74IH379iUpKYk///nPTYKQiJw89dyIyCkXExNDbm5uk+P79+8HIDY21nHejz/+2OS8oxuKG85/4IEHHH09R+vZs+dJ1wwcs24fHx86dOjgqGfmzJnMnDmT7OxsPvroI+6//37y8/P54osvAOjfvz/vvPMOhmGwfv165s6dy7Rp0wgODub+++8/qVpFxJlGbkTklDvvvPPYtGkTq1evdjr++uuvY7FYGDlyJAAjR46krKyMjz76yOm8t956y+n7nj170r17d9atW0dmZmazj/Dw8JOquWfPnnTq1Im33noLwzAcxysqKpg/f75jBtXRUlNTueOOO7jggguafF4Ai8XCgAEDeP7554mKimr2HBE5ORq5EZFT7u677+b111/nkksuYdq0aaSlpfHpp58ya9Ysfve739GjRw8AJk2axPPPP8+kSZN48skn6d69O5999hlffvllk2v+85//ZPTo0Vx44YXccMMNdOrUicLCQrKysli9ejXvvffeSdXs4+PDjBkzmDBhAmPGjOGWW26hurqaZ555huLiYp5++mkASkpKGDlyJNdeey29evUiPDycn376iS+++MIxqvTJJ58wa9Ysxo4dS5cuXTAMgw8++IDi4mIuuOCCk6pTRJpSuBGRU65jx44sW7aMBx54gAceeIDS0lK6dOnCjBkzmDp1quO8kJAQvvnmG+666y7uv/9+LBYLo0aN4p133mH48OFO1xw5ciQ//vgjTz75JFOmTKGoqIiYmBj69OnDuHHjWqXua6+9ltDQUKZPn8748ePx9fVl6NChLFq0yFFPUFAQQ4YM4Y033mD37t3U1taSmprKfffdx7333gtA9+7diYqKYsaMGezfv5+AgAB69uzJ3Llzuf7661ulVhFpZDGOHG8VERERaefUcyMiIiIeReFGREREPIrCjYiIiHgUhRsRERHxKAo3IiIi4lEUbkRERMSjeN06Nzabjf379xMeHo7FYjG7HBEREWkBwzAoKysjKSnJaTPb5nhduNm/fz8pKSlmlyEiIiInICcnh+TkZJfneF24adhvJicnh4iICJOrERERkZYoLS0lJSWlRfvGeV24abgVFRERoXAjIiLSzrSkpUQNxSIiIuJRFG5ERETEoyjciIiIiEfxup6blrJardTW1ppdhrQCf39/fH19zS5DRETaiMLNUQzDIC8vj+LiYrNLkVYUFRVFQkKC1jYSEfECCjdHaQg2cXFxhISE6JdhO2cYBpWVleTn5wOQmJhockUiInKqKdwcwWq1OoJNTEyM2eVIKwkODgYgPz+fuLg43aISEfFwaig+QkOPTUhIiMmVSGtr+Jmqj0pExPMp3DRDt6I8j36mIiLeQ+FGREREPIrCjRzTueeey5QpU8wuQ0RE5LioodgD/NItl+uvv565c+ce93U/+OAD/P39T7AqERERcyjctCKrzUZNnY3ggLb9x5qbm+v4et68efzpT39iy5YtjmMNs4Ua1NbWtii0REdHt16RIiIibUS3pVrJ4Zo6Nu4vZdehSgzDaNP3TkhIcDwiIyOxWCyO76uqqoiKiuLdd9/l3HPPJSgoiP/85z8UFBRwzTXXkJycTEhICP379+ftt992uu7Rt6XS09N56qmnuPHGGwkPDyc1NZU5c+a06WcVERH5JQo3v8AwDCpr6n7xYTUMqutslFfXUlRZ06LX/NKjNUPSfffdx+9//3uysrK48MILqaqqIiMjg08++YSff/6Zm2++mYkTJ7JixQqX13nuuefIzMxkzZo13Hbbbfzud79j8+bNrVaniIjIydJtqV9wuNZKnz99acp7b5p2ISGtdItrypQpXHHFFU7H7rnnHsfXd955J198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aRLr1q1j6dKlTiGm4T2ys7N555132LFjB3//+9/58MMPnc5JT09n165drF27lkOHDlFdXd3kvSZMmEBQUBDXX389P//8M4sWLeLOO+9k4sSJjltSIiIirijcnGJHbqJZ2QZbMdx0000UFRVx/vnnk5qaCsAjjzzC4MGDufDCCzn33HNJSEhg7NixLb6mj48PH374IdXV1ZxxxhlMnjyZJ5980umcyy+/nLvvvps77riDgQMHsmzZMh555BGnc6688kouuugiRo4cSceOHZudjh4SEsKXX35JYWEhp59+OldddRXnnXceL7zwwvH/wxAREa9kMZprqPBgpaWlREZGUlJS0qTZtaqqil27dtG5c2eCgoJa7T1ziw9zsLya6NAAkju0zcJ44uxU/WxFRKRtuPr9fTSN3LSBkEB7a9Op3oZBREREFG7aRMM2DFW17WMTTRERkfZM4aYN+Pv6EOhn/0et0RsREZFTy/RwM2vWLEcfREZGBkuXLnV5fnV1NQ899BBpaWkEBgbStWtXXn311Taq9sSFBOjWlIiISFswdZ2befPmMWXKFGbNmsWIESP45z//yejRo9m0aZNjps/Rxo0bx4EDB3jllVfo1q0b+fn51NW17q7bp6LHOiTQl6JKqDjFO4RL87ysb15ExKuZGm7++te/ctNNNzF58mQAZs6cyZdffsns2bOZPn16k/O/+OILlixZws6dO4mOjgbsa6e0loZVbysrKwkODm616wKE1o/cHK6xYjOM499EU05KZaV989OjVzYWERHPY1q4qampYdWqVdx///1Ox0eNGsWyZcuafc1HH31EZmYmM2bM4I033iA0NJTLLruMxx9//JhhpGEfpQalpaXHrMnX15eoqCjy8/MB+5orrbWisGEYWGy1WG0GJWUVBAdocei2YBgGlZWV5OfnExUV5bQflYiIeCbTfsMeOnQIq9XaZNXZ+Pj4Y66eu3PnTr777juCgoL48MMPOXToELfddhuFhYXH7LuZPn06jz32WIvratixuiHgtKaS8moO19qoKfInLEjhpi1FRUW53I1cREQ8h+m/YY8eGXG1caLNZsNisfDmm28SGRkJ2G9tXXXVVbz44ovNjt488MADTJ061fF9aWkpKSkpLutJTEwkLi6O2traE/lIx7RsxR5e/W4XZ/foyJ8v7d6q15Zj8/f314iNiIgXMS3c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o6jnRkRERDyKwo2IiIh4FIUbERER8SgKNyIiIuJRFG5ERETEoyjciIiIiEdRuBERERGPonAjIl5v8eLFWCwWiouLzS5FRFqBwo2IiIh4FIUbERER8SgKNyJiOsMwmDFjBl26dCE4OJgBAwbw/vvvA423jD799FMGDBhAUFAQQ4YMYcOGDU7XmD9/Pn379iUwMJD09HSee+45p+erq6u59957SUlJITAwkO7du/PKK684nbNq1SoyMzMJCQlh+PDhTXbPFpH2QeFGREz38MMP89prrzF79mw2btzI3XffzXXXXceSJUsc5/zxj3/k2Wef5aeffiIuLo7LLruM2tpawB5Kxo0bx9VXX82GDRt49NFHeeSRR5g7d67j9ZMmTeKdd97h73//O1lZWbz00kuEhYU51fHQQw/x3HPPsXLlSvz8/LQzs0g7pY0zRcRUFRUVxMbG8s033zBs2DDH8cmTJ1NZWcnNN9/MyJEjeeeddxg/fjwAhYWFJCcnM3fuXMaNG8eECRM4ePAgCxYscLz+3nvv5dNPP2Xjxo1s3bqVnj17snDhQs4///wmNSxevJiRI0fy1Vdfcd555wHw2Wefcckll3D48GGCgoJO8T8FEWlNGrkREVNt2rSJqqoqLrjgAsLCwhyP119/nR07djjOOzL4REdH07NnT7KysgDIyspixIgRTtcdMWIE27Ztw2q1snbtWnx9fTnnnHNc1nLaaac5vk5MTAQgPz//pD+jiLQtP7MLEBHvZrPZAPj000/p1KmT03OBgYFOAedoFosFsPfsNHzd4MhB6eDg4BbV4u/v3+TaDfWJSPuhkRsRMVWfPn0IDAwkOzubbt26OT1SUlIc5y1fvtzxdVFREVu3bqVXr16Oa3z33XdO1122bBk9evTA19eX/v37Y7PZnHp4RMRzaeRGREwVHh7OPffcw913343NZuPMM8+ktLSUZcuWERYWRlpaGgDTpk0jJiaG+Ph4HnroIWJjYxk7diwAf/jDHzj99NN5/PHHGT9+PD/88AMvvPACs2bNAiA9PZ3rr7+eG2+8kb///e8MGDCAPXv2kJ+fz7hx48z66CJyiijciIjpHn/8ceLi4pg+fTo7d+4kKiqKwYMH8+CDDzpuCz399NPcddddbNu2jQEDBvDRRx8REBAAwODBg3n33Xf505/+xOOPP05iYiLTpk3jhhtucLzH7NmzefDBB7ntttsoKCggNTWVBx980IyPKyKnmGZLiYhba5jJVFRURFRUlNnliEg7oJ4bERER8SgKNyIiIuJRdFtKREREPIpGbkRERMSjKNyIiIiIR1G4EREREY+icCMiIiIeReFGREREPIrCjYiIiHgUhRsRERHxKAo3IiIi4lEUbkRERMSj/D9h6Q0kBi7lAAAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot(history.history['loss'])\n",
    "plt.plot(history.history['val_loss'])\n",
    "plt.title('model loss')\n",
    "plt.ylabel('loss')\n",
    "plt.xlabel('epoch')\n",
    "plt.legend(['Train', 'Validation'], loc='upper left')\n",
    "plt.show()\n",
    "\n",
    "# graph represents the model’s loss"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "774526a5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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bdsWWq0WPXr0wIIFC1C/fv0Sx6hUKqhUqup/A0RERFQjidZyo1QqERISgvDwcJvt4eHhCA0NLfWYvLw8SKW2JctkMgCWFh8iIiIiUbulZsyYga+//hqrV69GZGQkpk+fjpiYGGs30+zZszF27Fjr/oMGDcIvv/yClStX4tq1azh06BCmTJmCzp07o0GDBmK9DSIiIqpBROuWAoBRo0YhNTUV8+fPR3x8PFq3bo2dO3fC398fABAfH28z58348eORnZ2N5cuXY+bMmXB1dUXv3r2xcOFCsd4CERER1TASoY7152RlZcHFxQWZmZlwdnYWuxwiIiKqgMp8f4t+tRQRERFRdWK4ISIiIrvCcENERER2heGGiIiI7ArDDREREdkVhhsiIiKyKww3REREZFcYboiIiMiuMNwQERGRXWG4ISIiIrvCcENERER2heGGiIiI7ArDDREREdkVhhsiIiKyKww3REREZFcYboiIiMiuMNwQERGRXWG4ISIiIrvCcENERER2heGGiIiI7ArDDREREdkVhhsiIiKyKww3REREZFcYboiIiMiuMNwQERGRXWG4ISIiIrvCcENERER2heGGiIiI7ArDDREREdkVhhsiIiKyKww3REREZFdEDzcrVqxAYGAg1Go1QkJCcODAgTL3HT9+PCQSSYlbq1at7mPFREREVJOJGm42btyIadOm4a233sLp06fRo0cP9O/fHzExMaXu/8knnyA+Pt56i42Nhbu7O0aMGHGfKyciIqKaSiIIgiDWi3fp0gUdO3bEypUrrduCg4MxdOhQhIWF3fH4LVu2YPjw4YiKioK/v3+FXjMrKwsuLi7IzMyEs7NzlWsnIiKi+6cy39+itdzo9XqcPHkSffv2tdnet29fHD58uELn+Oabb/DII49UONgQERGR/ZOL9cIpKSkwmUzw8fGx2e7j44OEhIQ7Hh8fH49du3Zh3bp15e6n0+mg0+msj7OysqpWMBEREdUKog8olkgkNo8FQSixrTTfffcdXF1dMXTo0HL3CwsLg4uLi/Xm6+t7N+USERFRDSdauPH09IRMJivRSpOUlFSiNed2giBg9erVGDNmDJRKZbn7zp49G5mZmdZbbGzsXddORERENZdo4UapVCIkJATh4eE228PDwxEaGlrusX/99ReuXLmCCRMm3PF1VCoVnJ2dbW5ERERkv0QbcwMA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pOSniFwNEVHd1bRpUwwZMgQvvPACDh48iDNnzuDZZ59Fw4YNMWSI5QrXadOmYffu3YiKisKpU6ewd+9ea/CZO3cutm7diitXruDcuXPYvn27TSii6lelMTddunTB6dOnq7sWKibIJQhtPNvAJJiw89pOscshIqrTvv32W4SEhODxxx9H165dIQgCdu7cCYVCAQAwmUx45ZVXEBwcjMceewzNmzfHihUrAABKpRKzZ89G27Zt0bNnT8hkMmzYsEHMt2P3JEIVJlP56aef8MYbb2D69OkICQmBVqu1eb5t27bVVmB1y8rKgouLCzIzM+Hs7Cx2OeXacGED3j/6Ppq7NcfPg38WuxwiokorKChAVFQUAgMDoVarxS6Harjy/l4q8/1dpQHFo0aNAmAZHV5EIpFYB1eZTKaqnJZu0z+wPxYdX4SL6RdxMe0imrs3F7skIiKiGq9K4SYqKqq666BSuKhc8JDvQwi/Ho6tV7dilvsssUsiIiKq8aoUbvz9/au7DirD4MaDEX49HDuu7cD0kOlQSBVil0RERFSjVSncrFmzptznx44dW6ViqKRuDbvBXe2OtII0HL55GL18e4ldEhERUY1WpXAzdepUm8cGgwF5eXlQKpXQaDSVCjcrVqzARx99hPj4eLRq1QrLli1Djx49St13//79ePjhh0tsj4yMRIsWLSr3JmoJhVSBAYED8GPkj9h6dSvDDRER0R1U6VLw9PR0m1tOTg4uXryI7t27Y/369RU+z8aNGzFt2jS89dZbOH36NHr06IH+/fsjJiam3OMuXryI+Ph4661p06ZVeRu1RtFK4ftj9yNTlyluMURERDVclcJNaZo2bYoPP/ywRKtOeZYsWYIJEyZg4sSJCA4OxrJly+Dr64uVK1eWe5y3tzfq1atnvclksrstv0Zr4d4CzdyawWA24Peo38Uuh4iIqEartnADADKZDHFxcRXaV6/X4+TJk+jb13bdpL59+95xUbEOHTqgfv366NOnD/bt21flemuTosU0uRwDERFR+ao05mbbNtsvWEEQEB8fj+XLl6Nbt24VOkdKSgpMJhN8fHxstvv4+CAhIaHUY+rXr49Vq1YhJCQEOp0OP/zwA/r06YP9+/ejZ8+epR6j0+lslorIysqqUH01zcCggVh6cin+TfkX1zKvIcglSOySiIiIaqQqhZuhQ4faPJZIJPDy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\n",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot(history.history['accuracy'])\n",
    "plt.plot(history.history['val_accuracy'])\n",
    "plt.plot(history.history['loss'])\n",
    "plt.plot(history.history['val_loss'])\n",
    "plt.title('Training Loss and accuracy')\n",
    "plt.ylabel('accuracy/Loss')\n",
    "plt.xlabel('epoch')\n",
    "plt.legend(['accuracy', 'val_accuracy','loss','val_loss'])\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "raw",
   "id": "87859c71",
   "metadata": {},
   "source": [
    "# Conclusion: With above code We can see, that throughout the epochs, our model accuracy \n",
    "#     increases and our model loss decreases,that is good since our model gains confidence\n",
    "#     with its predictions.\n",
    "#     \n",
    "#     1. The two losses (loss and val_loss) are decreasing and the accuracy \n",
    "#        (accuracy and val_accuracy)are increasing. \n",
    "#         So this indicates the model is trained in a good way.\n",
    "# \n",
    "#     2. The val_accuracy is the measure of how good the predictions of your model are. \n",
    "#        So In this case, it looks like the model is well trained after 10 epochs"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "3d2c8d1b",
   "metadata": {},
   "outputs": [
    {
     "ename": "SyntaxError",
     "evalue": "(unicode error) 'unicodeescape' codec can't decode bytes in position 2-3: truncated \\UXXXXXXXX escape (2561125329.py, line 4)",
     "output_type": "error",
     "traceback": [
      "\u001b[1;36m  File \u001b[1;32m\"C:\\Users\\ashis\\AppData\\Local\\Temp\\ipykernel_6688\\2561125329.py\"\u001b[1;36m, line \u001b[1;32m4\u001b[0m\n\u001b[1;33m    keras_model_path='C:\\Users\\ashis\\OneDrive\\Desktop\\Deep Learning'\u001b[0m\n\u001b[1;37m                                                                    ^\u001b[0m\n\u001b[1;31mSyntaxError\u001b[0m\u001b[1;31m:\u001b[0m (unicode error) 'unicodeescape' codec can't decode bytes in position 2-3: truncated \\UXXXXXXXX escape\n"
     ]
    }
   ],
   "source": [
    "#pwd\n",
    "\n",
    "# # Save the model\n",
    "keras_model_path='C:\\Users\\ashis\\OneDrive\\Desktop\\Deep Learning'\n",
    "model.save(keras_model_path)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6ea190c9",
   "metadata": {},
   "outputs": [],
   "source": [
    "#use the save model\n",
    "restored_keras_model = tf.keras.models.load_model(keras_model_path)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
