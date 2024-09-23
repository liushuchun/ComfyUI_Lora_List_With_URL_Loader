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


    
class LoraListStacker:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "data": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False}),
            },
            "optional": {"lora_stack": ("LORA_STACK",)},
        }

    RETURN_TYPES = ("LORA_STACK",)
    FUNCTION = "load_list_lora"
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
            local_path =   folder_paths.get_full_path("loras", filename)
            print("local path",local_path)
            
            if local_path!="NoneType" or local_path is None or not os.path.exists(local_path):
                print(f"File {filename} not found locally. Downloading...")
                if not self.download_file(lora_name, local_path):
                    raise Exception(f"Failed to download Lora model from {lora_name}")
            else:
                print(f"File {filename} already exists locally.")
            return filename
        return lora_name

    def parse_lora_list(self, data: str):
        # data is a list of lora model (lora_name, strength_model, strength_clip, url) in json format
        # trim data
        data = data.strip()
        if data == "" or data == "[]" or data is None:
            return []

        print(f"Loading lora list: {data}")

        lora_list = json.loads(data)
        if len(lora_list) == 0:
            return []

        available_loras = folder_paths.get_filename_list("loras")

        lora_params = []
        for lora in lora_list:
            lora_name = lora["name"]
            strength_model = lora["strength"]
            strength_clip = lora["strength"]

            if strength_model == 0 and strength_clip == 0:
                continue

            if lora_name not in available_loras:
                print(f"Not found lora {lora_name}, skipping")
                continue

            lora_params.append((lora_name, strength_model, strength_clip))

        return lora_params

    def load_list_lora(self, data, lora_stack=None):
        loras = self.parse_lora_list(data)

        if lora_stack is not None:
            loras.extend([l for l in lora_stack if l[0] != "None"])

        return (loras,)

class LoraListUrlLoader(LoraListStacker):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name1": ("STRING", {"forceInput": False}),
                "model_strength_1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "clip_strength_1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lora_name2": ("STRING", {"forceInput": False}),
                "model_strength_2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "clip_strength_2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lora_name3": ("STRING", {"forceInput": False}),
                "model_strength_3": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "clip_strength_3": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")

    def load_list_lora(self, model, clip, lora_name1, model_strength_1, clip_strength_1,
                       lora_name2, model_strength_2, clip_strength_2,
                       lora_name3, model_strength_3, clip_strength_3):
        loras = []
        if lora_name1 != "":
            lora_name1 = self.check_and_download_lora(lora_name1)
            loras.append((lora_name1, model_strength_1, clip_strength_1))

        if lora_name2 != "":
            lora_name2 = self.check_and_download_lora(lora_name2)
            loras.append((lora_name2, model_strength_2, clip_strength_2))

        if lora_name3 != "":
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
