import os
import time
import argparse
import pathlib
from PIL import Image

import torch
import faiss

from src.feature_extraction import MyResnet50, MyVGG16, RGBHistogram, LBP, MyViT
from src.dataloader import get_transformation

ACCEPTED_IMAGE_EXISTS = ['.jpg', '.png']

def get_image_list(image_root):
    image_root = pathlib.Path(image_root)
    image_list = list()
    for image_path in os.listdir(image_root):
        image_list.append(image_path[:-4])
    image_list = sorted(image_list, key=lambda x: x)
    return image_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', help='Name of dataset path', required=True, type=str)
    parser.add_argument('--feature_extractor', required=True, type=str, default='resnet50')
    parser.add_argument('--device', required=False, type=str, default='cpu')
    parser.add_argument('-k', '--top_k', type=int, default=10)
    parser.add_argument('--crop', required=False, type=bool, default=False)
    
    print('----------------- Start Ranking ----------------- ')
    start = time.time()
    args = parser.parse_args()
    device = torch.device(args.device)
    data_path = args.data_path
    image_root = os.path.join(data_path, 'image')
    feature_root = os.path.join(data_path, 'feature')
    evaluate_root = os.path.join(data_path, 'evaluation')
    query_root = os.path.join(data_path, 'groundtruth')
    crop_path = os.path.join(evaluate_root, 'crop')
    original_path = os.path.join(evaluate_root, 'original')
    
    os.makedirs(evaluate_root, exist_ok=True)
    os.makedirs(crop_path, exist_ok=True)
    os.makedirs(original_path, exist_ok=True)
    
    if args.feature_extractor == 'resnet50':
        extractor = MyResnet50(device)
    elif args.feature_extractor == 'vgg16':
        extractor = MyVGG16(device)
    elif args.feature_extractor == 'rgbhistogram':
        extractor = RGBHistogram(device)
    elif args.feature_extractor == 'lbp':
        extractor = LBP(device)
    elif args.feature_extractor == 'vit':
        extractor = MyViT(device)
    else:
        print("No matching model found!")
        return

    img_list = get_image_list(image_root)
    transform = get_transformation()
    
    for query_name in os.listdir(query_root):
        if query_name[-9:-4] == 'query':
            rank_list = []
            
            with open(query_root + '/' + query_name, 'r') as file:
                img_query, left, top, right, bottom = file.read().split()
            if data_path == 'static/oxford':
                img_query = img_query[5:]
            test_image_path = pathlib.Path(image_root + '/' + img_query + '.jpg')
            pil_image = Image.open(test_image_path)
            pil_image = pil_image.convert('RGB')
            
            path_crop = 'original'
            
            if args.crop:
                pil_image=pil_image.crop((float(left), float(top), float(right), float(bottom)))
                path_crop = 'crop'
                
            image_tensor = transform(pil_image)
            image_tensor = image_tensor.unsqueeze(0).to(device)
            feat = extractor.extract_features(image_tensor)

            indexer = faiss.read_index(feature_root + '/' + args.feature_extractor + '.index.bin')

            _, indices = indexer.search(feat, k=args.top_k)  

            for index in indices[0]:
                rank_list.append(str(img_list[index]))
            file_path = os.path.join(evaluate_root, path_crop, args.feature_extractor, query_name[:-10] + '.txt')
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(evaluate_root + '/' + path_crop + '/' + args.feature_extractor + '/' + query_name[:-10] + '.txt', "w") as file:
                file.write("\n".join(rank_list))
    end = time.time()
    print('Finish in ' + str(end - start) + ' seconds')


if __name__ == '__main__':
    main()