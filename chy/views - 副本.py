import os
import base64
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .utils import init_model, VGGNet,generate_h5_file
import h5py

# 加载图像特征和图像名称
def load_indexed_data(index_path):
    with h5py.File(index_path, 'r') as h5f:
        feats = h5f['dataset_1'][:]
        img_names = h5f['dataset_2'][:]
    return feats, img_names

# 计算两个特征向量之间的余弦相似度
def compute_cosine_similarity(query_feat, feats):
    norm_query_feat = query_feat / np.linalg.norm(query_feat)
    norm_feats = feats / np.linalg.norm(feats, axis=1)[:, np.newaxis]
    scores = np.dot(norm_query_feat, norm_feats.T)
    return scores


# 图像搜索功能
def image_search_function(image_path, index_path):
    # 生成H5文件
    generate_h5_file('./data/picture', index_path)
    
    # 加载图像特征和图像名称
    feats, img_names = load_indexed_data(index_path)

    # 初始化模型
    model, input_shape = init_model()

    # 提取上传图片的特征
    norm_feat = VGGNet().vgg_extract_feat(image_path)

    # 计算与上传图片特征的余弦相似度
    scores = compute_cosine_similarity(norm_feat, feats)

    # 根据相似度排序
    rank_ids = np.argsort(scores)[::-1]
    rank_scores = scores[rank_ids]

    # 选择相似度较高的前几张图片
    max_res = 9999
    search_results = []
    for i in range(max_res):
        if rank_scores[i] > 0.5:  # 设定相似度阈值
            search_results.append({'filename': img_names[rank_ids[i]].decode('utf-8'), 'source':'','score':float(rank_scores[i])})
        print(img_names[rank_ids[i]].decode('utf-8'),float(rank_scores[i]))
    return search_results

@csrf_exempt
def image_search_view(request):
    if request.method == 'POST' and request.FILES.get('image'):
        uploaded_image = request.FILES['image']
        print(uploaded_image)
        # 确保临时文件夹存在，如果不存在则创建
        temp_folder_path = './data/search'
        os.makedirs(temp_folder_path, exist_ok=True)
        # 保存上传的图片到临时文件夹中
        temp_image_path = os.path.join(temp_folder_path, uploaded_image.name)
        with open(temp_image_path, 'wb+') as destination:
            for chunk in uploaded_image.chunks():
                destination.write(chunk)
        # 调用图像搜索功能，传入上传的图片
        index_path = 'vgg_featureCNN.h5'  # 图像索引文件路径
        search_results = image_search_function(temp_image_path, index_path)
        # 删除临时图片文件
        # os.remove(temp_image_path)

        # 将筛选出的源文件转换为Base64编码并发送给前端
        data = []
        for result in search_results:
            source_file_path = os.path.join('./data/picture', result['filename'])
            with open(source_file_path, 'rb') as file:
                source_file_content = file.read()
                source_file_base64 = base64.b64encode(source_file_content).decode('utf-8')
                result['source'] = source_file_base64
                data.append(result)
        return JsonResponse({'data': data})
    else:
        return JsonResponse({'error': '请上传图片文件！'})
