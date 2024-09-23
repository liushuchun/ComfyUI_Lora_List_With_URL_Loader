import comfy.sd
import torch
import os
import sys
import folder_paths
import wget  # 引入 wget 模块
from pathlib import Path  # 用于处理文件路径

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))
import json
from typing import Dict
from nodes import LoraLoader

class AVLoraLoader(LoraLoader):
    @classmethod
    def INPUT_TYPES(s):
        inputs = LoraLoader.INPUT_TYPES()
        inputs["optional"] = {
            "lora_override": ("STRING", {"default": "None"}),
            "enabled": ("BOOLEAN", {"default": True}),
        }
        return inputs

    CATEGORY = "Liushuchun /Loaders"

    def download_file(self, url, save_path):
        """使用 wget 下载文件"""
        try:
            wget.download(url, save_path)
            print(f"\nDownloaded: {save_path}")
        except Exception as e:
            print(f"Failed to download {url}: {e}")
            return False
        return True

    def get_lora_filename(self, lora_name):
        """从URL提取文件名"""
        return os.path.basename(lora_name)

    def check_and_download_lora(self, lora_name):
        """如果lora_name是URL，检测并下载模型文件"""
        if lora_name.startswith("http"):
            filename = self.get_lora_filename(lora_name)
            local_path = os.path.join(folder_paths.get_folder("loras"), filename)

            if not os.path.exists(local_path):
                print(f"File {filename} not found locally. Downloading...")
                if not self.download_file(lora_name, local_path):
                    raise Exception(f"Failed to download Lora model from {lora_name}")
            else:
                print(f"File {filename} already exists locally.")
            return filename
        return lora_name

    def load_lora(self, model, clip, lora_name, *args, lora_override="None", enabled=True, **kwargs):
        if not enabled:
            return (model, clip)

        if lora_override != "None":
            if lora_override not in folder_paths.get_filename_list("loras"):
                print(f"Warning: Not found Lora model {lora_override}. Using {lora_name} instead.")
            else:
                lora_name = lora_override

        # 检查并下载 Lora 模型文件（如果需要）
        lora_name = self.check_and_download_lora(lora_name)

        return super().load_lora(model, clip, lora_name, *args, **kwargs)

class LoraListUrlLoader(AVLoraLoader):
    @classmethod
    def INPUT_TYPES(s):
        lora_files = ["None"] + folder_paths.get_filename_list("loras")
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name1": (lora_files,),
                "model_strength_1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "clip_strength_1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lora_name2": (lora_files,),
                "model_strength_2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "clip_strength_2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lora_name3": (lora_files,),
                "model_strength_3": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "clip_strength_3": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")

    def load_list_lora(self, model, clip, lora_name1, model_strength_1, clip_strength_1,
                       lora_name2, model_strength_2, clip_strength_2,
                       lora_name3, model_strength_3, clip_strength_3):
        loras = []
        if lora_name1 != "None":
            lora_name1 = self.check_and_download_lora(lora_name1)
            loras.append((lora_name1, model_strength_1, clip_strength_1))

        if lora_name2 != "None":
            lora_name2 = self.check_and_download_lora(lora_name2)
            loras.append((lora_name2, model_strength_2, clip_strength_2))

        if lora_name3 != "None":
            lora_name3 = self.check_and_download_lora(lora_name3)
            loras.append((lora_name3, model_strength_3, clip_strength_3))

        if len(loras) == 0:
            return (model, clip)

        def load_loras(lora_params, model, clip):
            for lora_name, strength_model, strength_clip in lora_params:
                lora_path = folder_paths.get_full_path("loras", lora_name)
                lora_file = comfy.utils.load_torch_file(lora_path)
                model, clip = comfy.sd.load_lora_for_models(model, clip, lora_file, strength_model, strength_clip)
            return model, clip

        return load_loras(loras, model, clip)

NODE_CLASS_MAPPINGS = {
    "LoraListUrlLoader": LoraListUrlLoader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoraListUrlLoader": "Lora URL List Loader",
}
