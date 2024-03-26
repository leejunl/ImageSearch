# 导入必要的库
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input as preprocess_input_vgg
import numpy as np
from numpy import linalg as LA
import os
import h5py


class VGGNet:
    def __init__(self):
        """
        初始化 VGGNet 类。
        
        参数:
        - self.input_shape: 输入图像的形状，格式为 (高度, 宽度, 通道数)。
        - self.weight: 使用的权重类型，此处为 'imagenet'。
        - self.pooling: 使用的池化方法，此处为 'max'。
        - self.model_vgg: VGG16 模型的实例。
        """
        self.input_shape = (224, 224, 3)
        self.weight = 'imagenet'
        self.pooling = 'max'
        self.model_vgg = VGG16(weights=self.weight, input_shape=(self.input_shape[0], self.input_shape[1], self.input_shape[2]), pooling=self.pooling, include_top=False)
        self.model_vgg.predict(np.zeros((1, 224, 224, 3)))

    def vgg_extract_feat(self, img_path):
        """
        提取图像特征的函数。
        
        参数:
        - img_path: 图像文件的路径。
        
        返回:
        - norm_feat: 经过处理的图像特征。
        """
        # 加载图像并进行预处理
        img = image.load_img(img_path, target_size=(self.input_shape[0], self.input_shape[1]))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input_vgg(img)
        # 提取特征并进行归一化处理
        feat = self.model_vgg.predict(img)
        norm_feat = feat[0] / LA.norm(feat[0])
        return norm_feat

def get_imlist(path):
    """
    获取指定路径下所有的图像文件列表。
    
    参数:
    - path: 文件路径。
    
    返回:
    - 文件列表，仅包含以 '.jpg' 结尾的文件。
    """
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]

def init_model():
    """
    初始化 VGG16 模型。
    
    返回:
    - model_vgg: VGG16 模型实例。
    - input_shape: 输入图像的形状。
    """
    input_shape = (224, 224, 3)
    weight = 'imagenet'
    pooling = 'max'
    model_vgg = VGG16(weights=weight, input_shape=(input_shape[0], input_shape[1], input_shape[2]), pooling=pooling, include_top=False)
    model_vgg.predict(np.zeros((1, 224, 224, 3)))
    return model_vgg, input_shape
    


def generate_h5_file(database_path, index_path):
    """
    Extract features from images in the database directory and index them into an H5 file.
    
    Args:
    - database_path (str): Path to the directory containing images.
    - index_path (str): Path to save the H5 index file.
    """
    img_list = [os.path.join(database_path, f) for f in os.listdir(database_path) if f.endswith('.jpg')]
    
    print("--------------------------------------------------")
    print("         feature extraction starts")
    print("--------------------------------------------------")
    
    feats = []
    names = []

    model = VGGNet()
    for i, img_path in enumerate(img_list):
        try:
            # 尝试加载图像
            img = image.load_img(img_path, target_size=(model.input_shape[0], model.input_shape[1]))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input_vgg(img)
            
            # 提取特征并进行归一化处理
            feat = model.model_vgg.predict(img)
            norm_feat = feat[0] / LA.norm(feat[0])
            
            # 添加有效图像的特征和名称
            feats.append(norm_feat)
            names.append(os.path.split(img_path)[1])
            
            print("extracting feature from image No. %d , %d images in total" %((i+1), len(img_list)))
        
        except Exception as e:
            print("Failed to extract feature from image:", img_path)
            print("Error:", e)
    
    feats = np.array(feats)
    
    print("--------------------------------------------------")
    print("      writing feature extraction results ...")
    print("--------------------------------------------------")

    with h5py.File(index_path, 'w') as h5f:
        h5f.create_dataset('dataset_1', data=feats)
        h5f.create_dataset('dataset_2', data=np.string_(names))
