import cv2
import numpy as np
import argparse

def magnify_region(image_path, x, y):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return None
    
    # 获取图像尺寸
    height, width = image.shape[:2]
    if height != 512:
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
        height, width = image.shape[:2]
    
    # 确保坐标在图像范围内
    x = max(0, min(x, width - 50))
    y = max(0, min(y, height - 50))
    
    # 提取要放大的区域
    region = image[y:y+50, x:x+50].copy()
    
    # 放大区域
    magnified = cv2.resize(region, (200, 200), interpolation=cv2.INTER_LINEAR)
    
    # 创建结果图像的副本
    result = image.copy()
    
    # 计算放大区域在右下角的位置
    mag_x = width - 200
    mag_y = height - 200
    
    # 确保放大区域不会超出图像范围
    mag_x = max(0, mag_x)
    mag_y = max(0, mag_y)
    
    # 将放大区域放置在右下角
    result[mag_y:mag_y+200, mag_x:mag_x+200] = magnified
    
    # 在原图上绘制红色矩形框标记要放大的区域
    cv2.rectangle(result, (x, y), (x+50, y+50), (0, 0, 255), 2)
    
    # 在放大区域上绘制红色矩形框
    cv2.rectangle(result, (mag_x, mag_y), (mag_x+200, mag_y+200), (0, 0, 255), 2)
    return result

def main():
    # parser = argparse.ArgumentParser(description='放大图像中的特定区域并将结果放在右下角')
    # parser.add_argument('image_path', type=str, help='输入图像路径')
    # parser.add_argument('x', type=int, help='区域左上角的X坐标')
    # parser.add_argument('y', type=int, help='区域左上角的Y坐标')
    # parser.add_argument('--output', type=str, default='output.jpg', help='输出图像路径')
    
    # args = parser.parse_args()
    
    files = ['Nikon_009_LR4.png','Nikon_023_LR4.png', 'Nikon_048_LR4.png']
    place = [[100,350], [110,300], [200,200]]
    # gen_path = "/mlx/users/liyueying.flora/playground/TFDSR/RB_Structure/gen"
    # base_path = "/mlx/users/liyueying.flora/playground/TFDSR/RB_Structure/base"
    # gt_path = "/mlx/users/liyueying.flora/playground/TFDSR/RB_Structure/gt"
    lr_path = "/mlx/users/liyueying.flora/playground/TFDSR/RB_Structure/lr"
    for i in range(len(files)):
        file = files[i]
        # result_base = magnify_region(f"{base_path}/{file}", place[i][0], place[i][1])
        # if result_base is not None:
        #     cv2.imwrite(f"base_{file}", result_base)

        # result_gen = magnify_region(f"{gen_path}/{file}", place[i][0], place[i][1])
        # if result_gen is not None:
        #     cv2.imwrite(f"gen_{file}", result_gen)

        # result_gt = magnify_region(f"{gt_path}/{file}", place[i][0], place[i][1])
        # if result_gt is not None:
        #     cv2.imwrite(f"gt_{file}", result_gt)
        result_lr = magnify_region(f"{lr_path}/{file}", place[i][0], place[i][1])
        if result_lr is not None:
            cv2.imwrite(f"lr_{file}", result_lr)
if __name__ == "__main__":
    main()