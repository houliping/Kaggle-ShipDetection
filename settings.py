import os

config={
    "raw_data_path":"../kaggle/train_v2/",
    "segmentation":"../kaggle/train_ship_segmentations.csv",
    "raw_test_path":"../kaggle/test_v2",
    "val_data_path":"/root/home/gujiang/ding/kaggle/validation/",
    "root":".~/kaggle",
    "shape":[768,768],
    'BAD_IMAGES' : ['6384c3e78.jpg','13703f040.jpg', '14715c06d.jpg',  '33e0ff2d5.jpg',
                '4d4e09f2a.jpg', '877691df8.jpg', '8b909bb20.jpg', 'a8d99130e.jpg',
                'ad55c3143.jpg', 'c8260c541.jpg', 'd6c7f17c7.jpg', 'dc3e7c901.jpg',
                'e44dffe88.jpg', 'ef87bad36.jpg', 'f083256d8.jpg'],
    "channels":3,
    "classes":1,
}
config_win={
    "raw_data_path":"./kaggle/train_v2/",
    "segmentation":"./kaggle/train_ship_segmentations_v2/train_ship_segmentations_v2.csv",
    "val_data_path":"./kaggle/validation/",
    "root":"./kaggle",
    "shape":[768,768],
    'BAD_IMAGES' : ['6384c3e78.jpg','13703f040.jpg', '14715c06d.jpg',  '33e0ff2d5.jpg',
                '4d4e09f2a.jpg', '877691df8.jpg', '8b909bb20.jpg', 'a8d99130e.jpg',
                'ad55c3143.jpg', 'c8260c541.jpg', 'd6c7f17c7.jpg', 'dc3e7c901.jpg',
                'e44dffe88.jpg', 'ef87bad36.jpg', 'f083256d8.jpg'],
    "channels":3,
    "classes":1
}
hyper_parameter={
    "alpha":10.0,
    "gamma":2,
    "batch_size":1,
    "num_workers":8
}