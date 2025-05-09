import argparse
import os
import numpy as np
import torch
from tqdm import tqdm
import pyiqa
from PIL import Image


def path2tensor(root, file):
    file_path = os.path.join(root, file)
    img = Image.open(file_path).convert('RGB')
    img = np.array(img).astype(np.float32) / 255.0

    img_tensor = torch.from_numpy(np.array(img).transpose(2, 0, 1)).float()
    img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor


class IQA:
    def __init__(self, path=None, ref_path=None):
        self.path = path
        self.ref_path = ref_path

        assert path is not None


    def calculate_full_metrics(self, model_name=None, batch_size=8, **kwargs):
        
        # 模型准备
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        iqa_metric = pyiqa.create_metric(model_name, device=device, **kwargs)

        # 数据准备
        count = 0
        batch_real_tensors,  batch_gen_tensors = [], []
        batch_iqa = []
        batch_real_imgs = []
        batch_gen_imgs = []

        file_names = os.listdir(self.path)
        for filename in tqdm(file_names, desc=model_name, unit="item"):
            if count == batch_size:
                # 计算当前batch指标
                batch_real_tensors = torch.cat(batch_real_imgs, dim=0).to(device)
                batch_gen_tensors = torch.cat(batch_gen_imgs, dim=0).to(device)
                batch_iqa.append(iqa_metric(batch_real_tensors, batch_gen_tensors).mean().item())

                # 更新迭代变量
                count -= batch_size
                batch_real_imgs = []
                batch_gen_imgs = []

            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                batch_real_imgs.append(path2tensor(self.ref_path, filename))
                batch_gen_imgs.append(path2tensor(self.path, filename))
                count += 1
            
        return f"{np.array(batch_iqa).mean():.4g}"
        
    
    def calculate_mse_loss(self, batch_size=8, **kwargs):
        
        # 模型准备
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # iqa_metric = pyiqa.create_metric(model_name, device=device, **kwargs)

        # 数据准备
        count = 0
        batch_real_tensors,  batch_gen_tensors = [], []
        batch_iqa = []
        batch_real_imgs = []
        batch_gen_imgs = []

        file_names = os.listdir(self.path)
        for filename in tqdm(file_names, desc="mse_loss", unit="item"):
            if count == batch_size:
                # 计算当前batch指标
                batch_real_tensors = torch.cat(batch_real_imgs, dim=0).to(device)
                batch_gen_tensors = torch.cat(batch_gen_imgs, dim=0).to(device)
                batch_iqa.append(pyiqa.losses.losses.mse_loss(batch_real_tensors, batch_gen_tensors).mean().item())

                # 更新迭代变量
                count -= batch_size
                batch_real_imgs = []
                batch_gen_imgs = []

            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                batch_real_imgs.append(path2tensor(self.ref_path, filename))
                batch_gen_imgs.append(path2tensor(self.path, filename))
                count += 1
            
        return f"{np.array(batch_iqa).mean():.4g}"
        # return f"{np.array(batch_iqa).mean()}"
        


    def calculate_no_metrics(self, model_name=None, batch_size=8):
        # 模型准备
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        iqa_metric = pyiqa.create_metric(model_name, device=device)

        # 数据准备
        count = 0
        batch_imgs = []
        batch_iqa = []

        file_names = os.listdir(self.path)
        for filename in tqdm(file_names, desc=model_name, unit="item"):
            if count == batch_size:
                # 计算当前batch指标
                batch_tensors = torch.cat(batch_imgs, dim=0).to(device)
                batch_iqa.append(iqa_metric(batch_tensors).mean().item())

                # 更新迭代变量
                count -= batch_size
                batch_imgs = []
            
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                batch_imgs.append(path2tensor(self.path, filename))
                count += 1
            
        return f"{np.array(batch_iqa).mean():.4g}"
    



    def IQA_calculate(self):
        
        psnr = self.calculate_full_metrics(model_name='psnr', batch_size=64, test_y_channel=True, color_space='ycbcr')
        ssim = self.calculate_full_metrics(model_name='ssim', batch_size=64, test_y_channel=True, color_space='ycbcr')
        lpips = self.calculate_full_metrics(model_name='lpips', batch_size=64)
        # dists = self.calculate_full_metrics(model_name='dists', batch_size=64)
        clip_iqa = self.calculate_no_metrics(model_name='clipiqa', batch_size=64)
        musiq = self.calculate_no_metrics(model_name='musiq', batch_size=64)
        maniqa = self.calculate_no_metrics(model_name='maniqa', batch_size=8)
        # niqe = self.calculate_no_metrics(model_name='niqe', batch_size=64)
        # # qalign = self.calculate_no_metrics(model_name='qalign', batch_size=64)

        # mse_loss = self.calculate_mse_loss(batch_size=64)

        # 格式化输出
        print(f'PSNR value: {psnr}')
        print(f'SSIM value: {ssim}')
        print(f'LPIPS value: {lpips}')
        # print(f'DISTS value: {dists}')

        print(f'MUSIQ value: {musiq}')
        # print(f'NIQE value: {niqe}')
        print(f'CLIP-IQA value: {clip_iqa}')
        print(f'MANIQA value: {maniqa}')

        # print(f'MSE Loss value: {mse_loss}')
        # return mse_loss

if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="Run IQA evaluation")
    parser.add_argument("--gen_path", type=str, required=True, help="Path to the generated images")
    parser.add_argument("--ref_path", type=str, required=True, help="Path to the reference images")
    
    # 解析命令行参数
    args = parser.parse_args()

    # 获取命令行参数
    gen_path = args.gen_path
    ref_path = args.ref_path


    # path="/mlx/users/liyueying.flora/playground/DiffBIR/timesteps/GT_L"
    # ref_path="/mlx_devbox/users/liyueying.flora/playground/arnold_workspace_root/SeeSR/val_datasets/gt"

    # ref_path="/mlx_devbox/users/liyueying.flora/playground/arnold_workspace_root/SeeSR/testing_datasets/RealSRVal_crop128/test_HR"
    # gen_path="/mlx_devbox/users/liyueying.flora/playground/arnold_workspace_root/SeeSR/test_outputs/RealSR/AEM_FSDA_APHL_t_v6/sample00"

    # ref_path="/mlx_devbox/users/liyueying.flora/playground/arnold_workspace_root/SeeSR/val_datasets/gt"
    # gen_path="/mlx_devbox/users/liyueying.flora/playground/arnold_workspace_root/SeeSR/params_adjust/l_4_h_1/sample00"

    
    # for item in sorted(os.listdir(path)):
    #     print(item)
    #     gen_path = os.path.join(path, item)
    iqa = IQA(gen_path, ref_path)
    iqa.IQA_calculate()