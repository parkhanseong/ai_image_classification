from flask import Flask, render_template, request, redirect, url_for, jsonify
import tensorflow as tf
import os
import cv2
import PIL
from PIL import Image
import numpy as np
# import os
import matplotlib.pyplot as plt

app = Flask(__name__)

current_dir = os.path.dirname( os.path.abspath( __file__ ) )

def predict(path, img_path):
    classes = ['비행기', '자동차', '새', '고양이', '사슴', '개', '개구리', '말', '배', '트럭']
    model = tf.keras.models.load_model(path)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
    img = img.reshape(1, 32, 32, 3)
    img = tf.cast(img, tf.float16)

    print(' >>> img')
    print(img)

    prediction = model.predict_classes(img)
    
    print(' >> prediction')
    print(prediction)

    return classes[prediction[0]]

@app.route("/")
def index():
    # source = url_for('static', filename="img.png")
    source = url_for('static', filename="cat9.png")
    return render_template("home.html", img_src = source)

@app.route("/getClass", methods=['POST'])
def get_class():
    # model_path = current_dir + url_for('static', filename="cifar10.h5")
    
    # model_phs_cifar.h5 > accuracy : 0.6992
    model_path = current_dir + url_for('static', filename="model_phs_cifar.h5") 
    img_path = current_dir + url_for('static', filename="cat9.png")

    print(' >> model_path')
    print(model_path)
    print(' >> img_path')
    print(img_path)

    prediction = predict(model_path, img_path)
    return jsonify(resp=prediction)

# @app.route("/generateImg", methods=['GET'])
# def generateImg():
    # data directory
input = os.getcwd() + "/static/data"
output = os.getcwd() + "/static/data/data.bin"
imageSize = 32
imageDepth = 3
debugEncodedImage = False

# show given image on the window for debug
def showImage(r, g, b):
    temp = []
    for i in range(len(r)):
        temp.append(r[i])
        temp.append(g[i])
        temp.append(b[i])
    show = np.array(temp).reshape(imageSize, imageSize, imageDepth)
    plt.imshow(show, interpolation='nearest')
    plt.show()

# convert to binary bitmap given image and write to law output file
def writeBinaray(outputFile, imagePath, label):
    img = Image.open(imagePath)
    img = img.resize((imageSize, imageSize), PIL.Image.ANTIALIAS)
    img = (np.array(img))

    r = img[:,:,0].flatten()
    g = img[:,:,1].flatten()
    b = img[:,:,2].flatten()
    label = [label]

    out = np.array(list(label) + list(r) + list(g) + list(b), np.uint8)
    outputFile.write(out.tobytes())

    # if you want to show the encoded image. set up 'debugEncodedImage' flag
    if debugEncodedImage:
        showImage(r, g, b)

subDirs = os.listdir(input)
numberOfClasses = len(input)

try:
    os.remove(output)
except OSError:
    pass

outputFile = open(output, "ab")
label = -1
totalImageCount = 0
labelMap = []

for subDir in subDirs:
    subDirPath = os.path.join(input, subDir)

    # filter not directory
    if not os.path.isdir(subDirPath):
        continue

    imageFileList = os.listdir(subDirPath)
    label += 1

    print("writing %3d images, %s" % (len(imageFileList), subDirPath))
    totalImageCount += len(imageFileList)
    labelMap.append([label, subDir])

    for imageFile in imageFileList:
        imagePath = os.path.join(subDirPath, imageFile)
        writeBinaray(outputFile, imagePath, label)

outputFile.close()
print("Total image count: ", totalImageCount)
print("Succeed, Generate the Binary file")
print("You can find the binary file : ", output)
print("Label MAP: ", labelMap)


if __name__ == "__main__":
    app.run(port = 5000)