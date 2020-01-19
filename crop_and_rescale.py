import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2
import os

HEIGHT = 137
WIDTH = 236
SIZE = 64


def series_to_img_reshape(df_row, height = 137, width = 236):
    '''
    DOCS
    Takes an pandas series
    
    returns numpy array
    '''    
    return np.resize(df_row.to_numpy(), (height, width)).astype(int)


def img_to_binary(img, threshold=200):
    '''
    DOCS
    '''
    img[img>threshold] = 255
    img[img<=threshold] = 0
    return img


def crop_symbol(img, pad=5):
    '''
    DOCS
    '''
    horizontal_where = np.any(img == 0, axis=1)
    x0 = np.searchsorted(horizontal_where, True)
    x1 = x0 + sum(horizontal_where)
    
    vertical_where = np.any(img == 0, axis=0)
    y0 = np.searchsorted(vertical_where, True)
    y1 = y0 + sum(vertical_where)
    
    img_crop = img[x0:x1, y0:y1]
    x, y = img_crop.shape
    
    # This could be simplified, but good for now
    if x < y:
        padding_y = np.ones((pad, np.size(img_crop, axis=1)))*255
        while x < y:
            img_crop = np.vstack((padding_y ,img_crop, padding_y))
            x, y = img_crop.shape
            
        padding_x = np.ones((np.size(img_crop, axis=0), pad))*255
        img_crop = np.hstack((padding_x, img_crop, padding_x))
        
    else:
        padding_x = np.ones((np.size(img_crop, axis=0), pad))*255
        while x > y:
            img_crop = np.hstack((padding_x ,img_crop, padding_x))
            x, y = img_crop.shape
            
        padding_y = np.ones((pad, np.size(img_crop, axis=1)))*255
        img_crop = np.vstack((padding_y, img_crop, padding_y))
        
    return img_crop

# Main script
root = '../parquet'

for file in os.listdir(root):
    path=os.path.join(root, file)
    if file == 'cropped':
        continue
    
    print(file)
    df = pd.read_parquet(path, 'fastparquet')
    
    N_images = len(df)
    
    images = np.zeros((N_images, SIZE**2))
    for idx in range(N_images):
        img = series_to_img_reshape(df.iloc[idx, 1:])
        img_binary = img_to_binary(img)
        img_crop = crop_symbol(img_binary)
        resized = cv2.resize(crop_img, (SIZE, SIZE), interpolation = cv2.INTER_AREA)
        images[idx, :] = resized.flatten()
        if idx%500 == 0:
            print(idx)

    df_to_write = pd.DataFrame(images, columns=[str(i) for i in range(SIZE**2)])
    if 'test' in file:
        file_type = 'Test'
    else:
        file_type = 'Train'
        
    df_to_write.insert(0, 'image_id', [file_type+'_'+str(i) for i in range(len(df_to_write))])

    df_to_write.to_parquet(os.path.join(*[root,'cropped', file]), engine='fastparquet')
