# Face_recognition_file_access

This Program is basically for file management uses face recognition feature.
you have follow some steps to use this.
1.Save some of your face images to images folder .


How it works:-
1.First It trains the images of images folder using classifiers.
Note:
Classifier defines the structure of face.

2.It saves the trained information in pickle.
3.After when we access camera using open-cv we just try to match the trained images with framed images.
if it matches then we try to access directory and thats  how this program works.
