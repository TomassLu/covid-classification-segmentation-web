# covid-classification-segmentation
Simple covid classification, lung and infection segmentation from image

# How to use
On cmd go to backend folder and use command:

```
python app.py
```

The go to frontend folder and use command:

```
npm start
```

Then the web interface will open
In it select "choose file" and choose a lung x-ray image (for testing image used - covid_1585.png
Then select "Upload and analyze" and wait for process to finish
The result will be displayed at the bottom

# Libraries used
Python: Tensorflow, keras, matplot, numpy, cv2, os, flask
Javascript: axios, react (the javascript frontend was built with create-react-app)
(Tensorflow and keras versions - 2.13)

# Used method info
The same as covid-classification-segmentation three models were trained: densenet201 for image classification and 2 unets for lung and infection segmentation (for densenet batch size used - 16, for unets - 8, optimizer - adam)
The handleFileChange on changes select image and handleUpload uses axios to transfer it to the backend (through upload_image '/upload') where the image is then saved to file
The models are used on the input image
THe results are set into matplot and saved as image 
('matplotlib.use('agg') was used to fix matplot instability when used in backend)
The generated result image is transfered to frontend (through get_result_image '/results/<filename>')

# Data used
The models and data for model training is the same as covid-classification-segmentation project from https://www.kaggle.com/datasets/anasmohammedtahir/covidqu (only the infection segmentation data)

# Model training info
Due to time constraints the models were trained with low amount of epochs (densenet - 5, unets - 10) and as such the results are not very accurate and serve more as proof of concept

# Image Preview of the web interface
![Interface image](https://github.com/TomassLu/covid-classification-segmentation-web/blob/main/Result.png)
