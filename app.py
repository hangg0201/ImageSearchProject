import time
import torch
import faiss
import os
import pathlib
from PIL import Image
from flask import Flask, render_template, request
from src.feature_extraction import MyVGG16, MyResnet50, RGBHistogram, LBP, MyViT
from src.dataloader import get_transformation

UPLOAD_FOLDER = './'

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

device = torch.device('cpu')

def get_image_list(image_root):
    image_root = pathlib.Path(image_root)
    image_list = list()
    for image_path in image_root.iterdir():
        if image_path.exists():
            image_list.append(image_path)
    image_list = sorted(image_list, key=lambda x: x.name)
    return image_list

def retrieve_image(img, feature_extractor, feature_root):
    if (feature_extractor == 'vgg16'):
        extractor = MyVGG16('cpu')
    elif (feature_extractor == 'resnet50'):
        extractor = MyResnet50(device)
    elif (feature_extractor == 'rgbhistogram'):
        extractor = RGBHistogram(device)
    elif (feature_extractor == 'lbp'):
        extractor = LBP(device)
    elif (feature_extractor == 'vit'):
        extractor = MyViT(device)
    transform = get_transformation()

    img = img.convert('RGB')
    image_tensor = transform(img)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    feat = extractor.extract_features(image_tensor)

    indexer = faiss.read_index(feature_root + '/' + feature_extractor + '.index.bin')

    _, indices = indexer.search(feat, k=11)

    return indices[0]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        option = request.form['feature_extractor']
        dataset_name = request.form['dataset']
        image_root = os.path.join('static', str(dataset_name), 'image')
        print(image_root)
        feature_root = os.path.join('static', str(dataset_name), 'feature')
        img_file = request.files['img_file']
        img_file.save(os.path.join(
        app.config['UPLOAD_FOLDER'],
        "query.jpg"
        ))
        img = Image.open(img_file)

        retriev = retrieve_image(img, option, feature_root)
        image_list = get_image_list(image_root)
        
        os.remove("query.jpg")
        
        res_path = []
        for i in range(11):
            path = str(image_list[retriev[i]])
            res_path.append(path)
        return render_template('result.html', PATHS=res_path)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
