import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2

import sys
import os
sys.path.append(os.path.abspath('.'))

from crop_and_rescale import crop_and_resize, img_to_binary, crop_symbol


HEIGHT = 137
WIDTH = 236
SIZE = 64


# # Main script for writing all parquet crops to a seperate folder
root = 'parquet'

for file in os.listdir(root):
    path=os.path.join(root, file)
    if file == 'cropped':
        continue
    
    print(file)
    df = pd.read_parquet(path, 'fastparquet')
    print('File has been read with fast parquet')
    N_images = len(df)
    
    images = np.zeros((N_images, SIZE**2))
    for idx in range(N_images):
        img = series_to_img_reshape(df.iloc[idx, 1:])
        img_binary = img_to_binary(img)
        img_crop = crop_symbol(img_binary)
        try:
            resized = cv2.resize(img_crop, (SIZE, SIZE), interpolation = cv2.INTER_AREA)
            images[idx, :] = resized.flatten()
        except cv2.error:
            images[idx, :] = np.ones(SIZE**2)
            print(f'Error as index {idx}')

    df_to_write = pd.DataFrame(images, columns=[str(i) for i in range(SIZE**2)])
    
    if 'test' in file:
        file_type = 'Test'
    else:
        file_type = 'Train'
        
    df_to_write.insert(0, 'image_id', [file_type+'_'+str(i) for i in range(len(df_to_write))])

    df_to_write.to_parquet(os.path.join(*[root,'cropped', file]), engine='fastparquet')
