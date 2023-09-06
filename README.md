# Applied Machine Intelligence Project

It is the repository of the final project for Applied Machine Intelligence (SS23) at Technical University Munich.

## Introduction
This project aims to help clinicians to identify blood cell types. The main functions are segmentation, classification and active learning.

## Requirements

This project requires Python 3.9 or higher and uses the following Python packages:

- pytorch
- torchvision
- scikit-learn
- numpy
- matplotlib
- django
- openCV
- tqdm
- h5py

## User's Guide
> :computer: : work for **Windows**, **Linux** and **macOS**
>

To install the necessary packages, use pip:

```bash
pip install torch
pip install torchvision
pip install scikit-learn
pip install numpy
pip install matlabplot
pip install django
pip install opencv-python-headless
pip install tqdm
pip install h5py
```

**Local Version**

To run the Django server, switch to /CellAI, use the following command:
```bash
python manage.py runserver
```
Then open a web browser and navigate to http://127.0.0.1:8000/ to see the website in action.


In the homepage, upload a h5py file and wait for the prediction.
![homepage](assets/homepage.png)


In the results gallery, click an image and modify the prediction label. Click `UPDATE & TRAINING` to retrain the model and predict.
![homepage](assets/details.png)
![homepage](assets/update.png)



