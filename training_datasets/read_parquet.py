import pandas as pd
from PIL import Image
import os
import io
import os
import shutil

def read_parquet(parquet_file, output_folder):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    # 读取 Parquet 文件
    df = pd.read_parquet(parquet_file)
    image_column = "image"  # 根据具体列名修改
    for index, row in df.iterrows():
        image_data = row[image_column]
        image = Image.open(io.BytesIO(image_data['bytes']))
        # 获取文件名，确保仅包含文件名而不是路径
        image_filename = os.path.basename(row['path'])
        image_path = os.path.join(output_folder, image_filename)
        # 确认路径不是目录
        if os.path.isdir(image_path):
            print(f"错误: {image_path} 被识别为目录而非文件路径。")
        else:
            # 保存图像
            print(f"正在保存图片到: {image_path}")
            image.save(image_path)
    print("图片保存完成！")




def sort_and_limit_files(folder_path, limit=10000):
    # 获取文件夹中所有文件的路径
    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    # 按文件名排序
    file_paths.sort()
    
    # 保留前limit个文件，删除其余文件
    for file_path in file_paths[limit:]:
        os.remove(file_path)
        print(f"Deleted: {file_path}")




if __name__=='__main__':
    # # 定义文件路径和保存目录
    # data_path = "/mlx_devbox/users/liyueying.flora/playground/arnold_workspace_root/SeeSR/training_datasets/LSDIR_parquet"
    # parquet_list = os.listdir(data_path)
    # # parquet_list = ['train-00000-of-00195']
    # sum = 0
    # for parquet in parquet_list:
    #     parquet_file = os.path.join(data_path, parquet)
    #     print(parquet)
    #     output_folder = os.path.join(data_path, parquet.split('.')[0])
    #     # sum += len(os.listdir(output_folder))
    #     # 对每个文件执行提取图片操作
    #     read_parquet(parquet_file, output_folder)
    # print(f"一共有图片{sum}张")
    sort_and_limit_files("/mlx_devbox/users/liyueying.flora/playground/arnold_workspace_root/SeeSR/training_datasets/LSDIR10K")
