# Object detection
# Semantic seg
# Instance seg
# Human keypoint detection
# Depth estimation
# Surface normal estimation
# Low-light enhancement
# OCR

"""
fastapi dev image_augmentations/main/api_server.py --port 7777
"""

import os
import random
from PIL import Image, ImageEnhance
from typing import Annotated, Literal, List, Tuple, Dict, Optional, Union
from enum import Enum
import enum
from PIL import Image
from io import BytesIO
import base64
import cv2
from fastapi import FastAPI, Body
import subprocess
import asyncio
import numpy as np
import torch
import sys
import logging
import torch.nn.functional as F
 
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)

class ImageAugmentationType(Enum):
    LOW_LIGHT = enum.auto()
    OBJ_DETECT = enum.auto()
    SEMA_SEG = enum.auto()
    INST_SEG = enum.auto()
    HUMAN_KEYPOINT = enum.auto()
    DEPTH_EST = enum.auto()
    SURFACE_NORM = enum.auto()
    OCR = enum.auto()
    
class ImageAugmentationProvider:
    async def __call__(self, image: Image.Image) -> Union[dict, Image.Image]:
        raise NotImplementedError()

class LowLightProvider(ImageAugmentationProvider):
    async def __call__(self, image: Image.Image) -> Union[dict, Image.Image]:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(2.0)
        return image

class SegAnythingProvider(ImageAugmentationProvider):
    def __init__(self):
        self.device = 0
        self.lock = asyncio.Lock()
        try:
            self.load_model()
        except (ModuleNotFoundError, ImportError) as ex:
            logger.error(ex)
            self.install()
            self.load_model()
    
    def install(self):
        exit_code = subprocess.call([
            'pip', 'install', 'git+https://github.com/facebookresearch/segment-anything.git'
        ])
        assert exit_code == 0
        
        os.makedirs('./.cache/sam', exist_ok=True)
        exit_code = subprocess.call([
            'wget',
            '-r',
            '-O', './.cache/sam/sam_vit_h_4b8939.pth',
            'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'
        ])
        assert exit_code == 0
    
    def load_model(self):
        from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
        
        if not os.path.exists('./.cache/sam/sam_vit_h_4b8939.pth'):
            os.makedirs('./.cache/sam', exist_ok=True)
            exit_code = subprocess.call([
                'wget',
                '-r',
                '-O', './.cache/sam/sam_vit_h_4b8939.pth',
                'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'
            ])
            assert exit_code == 0
        
        logger.info('SAM loading... ./.cache/sam/sam_vit_h_4b8939.pth')
        self.sam = sam_model_registry["vit_h"](checkpoint="./.cache/sam/sam_vit_h_4b8939.pth").to(self.device).eval()
        # self.predictor = SamPredictor(self.sam)
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)
        logger.info('SAM loaded')
    
    async def __call__(self, image: Image.Image) -> Dict | Image.Image:
        async with self.lock:
            with torch.inference_mode(), torch.autocast('cuda', torch.float16):
               masks = self.mask_generator.generate(np.asarray(image.convert('RGB')))
            
            H, W = masks[0]['segmentation'].shape
            img = torch.zeros((H, W, 3), dtype=torch.uint8)
            
            for i, mask_data in enumerate(masks):
                mask = torch.tensor(mask_data["segmentation"])
                img.view(-1, 3)[mask.reshape(-1), :] = torch.tensor([
                    random.randint(0, 255), 
                    random.randint(0, 255), 
                    random.randint(0, 255),
                ], dtype=img.dtype)
            
            img = img.cpu().numpy()
            
            image = Image.fromarray(img)
             
            return image

class DepthAnythingProvider(ImageAugmentationProvider):
    def __init__(self):
        self.device = 0
        self.lock = asyncio.Lock()
        try:
            self.load_model()
        except (ModuleNotFoundError, ImportError) as ex:
            logger.error(ex)
            self.install()
            self.load_model()
    
    def install(self):
        os.makedirs('./.cache/depth_anything', exist_ok=True)
        
        if not os.path.exists('./.cache/depth_anything/depth_anything/.git/HEAD'):
            assert subprocess.call('git clone https://github.com/LiheYoung/Depth-Anything ./.cache/depth_anything/depth_anything'.split()) == 0
            assert subprocess.call('pip install -r ./.cache/depth_anything/depth_anything/requirements.txt'.split()) == 0
            assert subprocess.call('mv ./.cache/depth_anything/depth_anything/torchhub .'.split())
    
    def load_model(self):
        if './.cache/depth_anything/depth_anything' not in sys.path:
            sys.path.append('./.cache/depth_anything/depth_anything')
            os.environ['PYTHONPATH'] = f'./.cache/depth_anything/depth_anything:{os.getenv("PYTHONPATH", "")}'
        from depth_anything.dpt import DepthAnything
        logging.info('DAM loading ... vitl')
        encoder = 'vitl'
        self.depth_anything = DepthAnything.from_pretrained(
            'LiheYoung/depth_anything_{:}14'.format(encoder)
        ).to(self.device)
        logging.info('DAM loaded')
    
    async def __call__(self, pil_image: Image.Image) -> Dict | Image.Image:
        from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
        from torchvision.transforms import Compose

        transform = Compose([
            Resize(
                width=518,
                height=518,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
        
        async with self.lock:
            image = np.asarray(pil_image.convert('RGB')) / 255.0
            
            h, w = image.shape[:2]
            
            image = transform({'image': image})['image']
            image = torch.from_numpy(image).unsqueeze(0).to(self.device)
            
            with torch.inference_mode(), torch.autocast('cuda', torch.float16):
                depth = self.depth_anything(image)
            
            depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.cpu().numpy().astype(np.uint8)
            depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
            depth = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)
            
            image = Image.fromarray(depth)
            
            return image

class GroundingDINOProvider(ImageAugmentationProvider):
    def __init__(self):
        self.device = 0
        self.lock = asyncio.Lock()
        try:
            self.load_model()
        except (ModuleNotFoundError, ImportError) as ex:
            logger.error(ex)
            self.install()
            self.load_model()
    
    def load_model(self):
        from groundingdino.util.inference import load_model, load_image, predict, annotate
        import groundingdino.datasets.transforms as T

        self.model = load_model("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")
        self.IMAGE_PATH = "weights/dog-3.jpeg"
        self.TEXT_PROMPT = "chair . person . dog ."
        self.BOX_TRESHOLD = 0.35
        self.TEXT_TRESHOLD = 0.25
        
        self.predict = predict
        self.annotate = annotate
        
        self.transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    
    def install(self):
        MSG = """
        Installation:

        1.Clone the GroundingDINO repository from GitHub.
        ```
            git clone https://github.com/IDEA-Research/GroundingDINO.git
            cd GroundingDINO/
            pip install -e .
        ```
        
        2.Download pre-trained model weights.
        ```
            mkdir weights
            cd weights
            wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
            cd .. 
        ```
        """
        print(MSG)
        input('After install, press enter >>>')
        
        assert os.path.exists("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
        assert os.path.exists("weights/groundingdino_swint_ogc.pth")
    
    async def __call__(self, image: Image.Image) -> Dict | Image.Image:
        image_source = np.asarray(image.convert('RGB'))
        
        image_transformed, _ = self.transform(image.convert('RGB'), None)

        boxes, logits, phrases = self.predict(
            model=self.model,
            image=image_transformed,
            caption=self.TEXT_PROMPT,
            box_threshold=self.BOX_TRESHOLD,
            text_threshold=self.TEXT_TRESHOLD
        )

        annotated_frame = self.annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        image = Image.fromarray(annotated_frame)
        
        return image

class ImageAugmentationService:
    def __init__(self):
        self.providers = {
            ImageAugmentationType.LOW_LIGHT: LowLightProvider(),
            ImageAugmentationType.SEMA_SEG: SegAnythingProvider(),
            ImageAugmentationType.DEPTH_EST: DepthAnythingProvider(),
            ImageAugmentationType.OBJ_DETECT: GroundingDINOProvider(),
        }
    
    async def augmentation(self, img: Image.Image, type: ImageAugmentationType):
        if type in self.providers:
            return await self.providers[type](img)
        else:
            raise Exception()

app = FastAPI()
service: ImageAugmentationService = None

def get_service():
    global service
    if service is None:
        service = ImageAugmentationService()
    return service

@app.get("/list")
async def list_api():
    return list(map(lambda x: str(x).split('.')[-1], get_service().providers.keys()))

@app.post("/augmentation")
async def augmentation(
    image_base64: Annotated[str, Body()],
    type: Annotated[str, Body()],
):
    img = Image.open(BytesIO(base64.b64decode(image_base64)))
    result = await get_service().augmentation(img, ImageAugmentationType[type])
    
    if isinstance(result, Image.Image):
        buffer = BytesIO()
        result.save(buffer, format="PNG")
        result_base64 = base64.b64encode(buffer.getvalue())
        
        return {
            'type': 'base64',
            'content': result_base64
        }
    else:
        raise Exception('not supported return type')