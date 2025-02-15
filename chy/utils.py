# 导入必要的库
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input as preprocess_input_vgg
import numpy as np
from numpy import linalg as LA
import os
import h5py
import re
import requests
import time
import json

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
    从数据库目录中提取图像特征并将其索引到 H5 文件中。
    
    参数:
    - database_path (str): 包含图像的目录路径。
    - index_path (str): 保存 H5 索引文件的路径。
    """
    img_list = [os.path.join(database_path, f) for f in os.listdir(database_path) if f.endswith('.jpg')]
    
    print("--------------------------------------------------")
    print("         开始特征提取")
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
            
            print("提取图像特征中... (%d / %d)" %((i+1), len(img_list)))
        
        except Exception as e:
            print("无法从图像中提取特征:", img_path)
            print("错误:", e)
    
    feats = np.array(feats)
    
    print("--------------------------------------------------")
    print("      正在写入特征提取结果 ...")
    print("--------------------------------------------------")

    with h5py.File(index_path, 'w') as h5f:
        h5f.create_dataset('dataset_1', data=feats)
        h5f.create_dataset('dataset_2', data=np.array(names, dtype=h5py.string_dtype(encoding='utf-8')))

def save_settings_to_file(url, cookies):
    data = {'url': url, 'cookies': cookies}
    with open('spyder_settings.json', 'w') as f:
        json.dump(data, f)

def load_settings_from_file():
    try:
        with open('spyder_settings.json', 'r') as f:
            settings = json.load(f)
            return settings.get('url', ''), settings.get('cookies', '')
    except FileNotFoundError:
        return '', ''


def spyder(search_word, url, cookie, istext, error_count=0, data=None):
    if data is None:
        data = []  # 初始化结果数据

    num = 1  # 给图片名字加数字
    
    header = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.63 Safari/537.36',
        'Cookie': cookie,  # 这里需要根据自己的浏览器情况自行填写
        'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'zh-CN,zh;q=0.9'
    }  # 请求头
    # 图片页面的url
    urls = url + search_word
    # 通过requests库请求到了页面
    print('=====================  开始 爬取   ============================')
    try:
        html = requests.get(urls, headers=header, verify=False)
        # 防止乱码
        html.encoding = 'utf8'
        # 打印页面出来看看
        html = html.text

        picture_path = './data/picture'
        if not os.path.exists(picture_path):
            os.mkdir(picture_path)

         # 正则表达式，筛选出html页面中符合条件的图片源代码地址url
        res = re.findall('"middleURL": "(.*?)"', html) 
        print(res)
        # 过滤掉空值或只包含空白字符的匹配项  
        res = [match for match in res if match.strip()]  
        images = []
        print('检索到的url:',res)  # 看看有哪些url
        print('爬取到的URL数===========>:',len(res))
        print('循环次数================>:',5 if len(res)>=5 else len(res))
        
        if len(res)==0:
            print('=====================  爬取结束   ============================')
            return {'code': 500,'data':[]},0
        
        for i in res:  # 遍历
            print('当前：',num,i)
            if num >= (5 if len(res)>=5 else len(res)):  # 设置只返回10张正确的图片
                print('=====================  爬取结束   ============================')
                print('data的值为：',data)
                print('error_count的值为：',error_count)
                return {'code': 200, 'data': data}, error_count  # 返回成功的同时返回当前的 error_count 值和结果数据
            num = num + 1  # 数字加1，这样图片名字就不会重复了
            print('当前次数：',num,istext)
            if istext == 'yes':
                
                images.append({'filename': f'{search_word}{num}.jpg', 'source': i})  # 将图片URL添加到结果数据中
                data= images  # 将当前的图片数据添加到结果数据中
                continue  # 如果是文字模式，直接跳过保存图片的部分
            picture = requests.get(i, headers=header, verify=False)  # 得到每一张图片的大图
            file_name = f'./data/picture/{search_word}{num}.jpg'  # 给下载下来的图片命名。加数字，是为了名字不重复
            with open(file_name, "wb") as f:  # 以二进制写入的方式打开图片
                f.write(picture.content)  # 往图片里写入爬下来的图片内容，content是写入内容的意思
            
               
    except requests.exceptions.ConnectionError as e:
        print('连接错误次数:', error_count)
        error_count += 1
        if error_count >= 5:
            print('连接错误次数过多，爬取失败')
            return {'code': 500, 'data': data}, error_count  # 返回500错误并且返回新的 error_count 值和结果数据
        time.sleep(1)  # 等待1秒后继续尝试
        return spyder(search_word, url, cookie, istext, error_count, data)  # 递归调用自身，重新尝试爬取

    except Exception as e:
        print('发生异常:', e)
        import traceback
        traceback.print_exc()
        error_count += 1
        if error_count >= 5:
            print('爬取失败')
            return {'code': 500, 'data': data}, error_count  # 返回500错误并且返回新的 error_count 值和结果数据
        time.sleep(1)  # 等待1秒后继续尝试
        return spyder(search_word, url, cookie, istext, error_count, data)  # 递归调用自身，重新尝试爬取

    #return {'code': 200, 'data': data}, error_count  # 在发生异常后返回错误代码和结果数据
