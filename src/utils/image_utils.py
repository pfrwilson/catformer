
import skimage
import requests
import io
from PIL import Image
from typing import Tuple
import numpy as np

def get_sample_image():
    cat_url = r'https://static01.nyt.com/images/2021/09/14/science/07CAT-STRIPES/07CAT-STRIPES-mediumSquareAt3X-v2.jpg'
    response = requests.get(cat_url)
    bytes = io.BytesIO(response.content)
    img = Image.open(bytes)
    array = np.array(img)
    
    return {
        'img': img, 
        'array': array
    }

def get_sample_mask(size=32):
    random = np.random.randn(size**2).reshape((size, size))
    img = Image.fromarray(random, mode='L')
    return {
        'img': img, 
        'array': random
    }
        
def resize_like(target_img, source_img):
    
    h = target_img.height
    w = target_img.width
    
    return source_img.resize((h, w))
    
def add_opacity_mask(image: Image, mask: Image):
    image.putalpha(mask)
    return image