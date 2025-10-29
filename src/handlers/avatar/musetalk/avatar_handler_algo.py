import copy
import glob
import json
import os
import pickle
import queue
import shutil
import sys
import threading
import time

import cv2
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm
from transformers import WhisperModel

# 添加MuseTalk模块路径
current_dir = os.path.dirname(os.path.abspath(__file__))
musetalk_module_path = os.path.join(current_dir, "MuseTalk")
if musetalk_module_path not in sys.path:
    sys.path.append(musetalk_module_path)

handlers_dir = os.getcwd()
handlers_dir = os.path.join(handlers_dir, "src")
if handlers_dir not in sys.path:
    sys.path.append(handlers_dir)

from src.handlers.avatar.musetalk.MuseTalk.musetalk.utils.audio_processor import AudioProcessor
from src.handlers.avatar.musetalk.MuseTalk.musetalk.utils.blending import get_image_prepare_material, get_image_blending
from src.handlers.avatar.musetalk.MuseTalk.musetalk.utils.face_parsing import FaceParsing
from src.handlers.avatar.musetalk.MuseTalk.musetalk.utils.utils import load_all_model, datagen
from src.handlers.avatar.musetalk.utils.preprocessing import read_imgs, get_landmark_and_bbox


def osmakedirs(path_list):
    for path in path_list:
        os.makedirs(path) if not os.path.exists(path) else None


def video2imgs(vid_path, save_path, ext='.png', cut_frame=10000000):
    cap = cv2.VideoCapture(vid_path)
    count = 0
    while True:
        if count > cut_frame:
            break
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f"{save_path}/{count:08d}.png", frame)
            count += 1
        else:
            break


class Avatar:
    def __init__(self, avatar_id, video_path, bbox_shift, batch_size, preparation=False,
                 parsing_mode='jaw', extra_margin=10, fps=25,
                 gpu_id=0, version="v15",
                 audio_padding_length_left=2, audio_padding_length_right=2,
                 left_cheek_width=90, right_cheek_width=90,
                 vae_type="sd-vae", unet_model_path=None, unet_config=None,
                 result_dir="./results", whisper_dir=None):
        self.avatar_id = avatar_id
        self.video_path = video_path
        self.bbox_shift = bbox_shift
        self.batch_size = batch_size
        self.preparation = preparation
        self.parsing_mode = parsing_mode
        self.extra_margin = extra_margin
        self.fps = fps
        self.gpu_id = gpu_id
        self.version = version
        self.audio_padding_length_left = audio_padding_length_left
        self.audio_padding_length_right = audio_padding_length_right
        self.left_cheek_width = left_cheek_width
        self.right_cheek_width = right_cheek_width
        self.vae_type = vae_type
        self.unet_model_path = unet_model_path
        self.unet_config = unet_config
        self.result_dir = result_dir
        self.whisper_dir = whisper_dir

        if self.version == "v15":
            self.base_path = f"{self.result_dir}/{self.version}/avatars/{avatar_id}"
        else:  # v1
            self.base_path = f"{self.result_dir}/avatars/{avatar_id}"

        self.avatar_path = self.base_path
        self.full_imgs_path = f"{self.avatar_path}/full_imgs"
        self.coords_path = f"{self.avatar_path}/coords.pkl"
        self.latents_out_path = f"{self.avatar_path}/latents.pt"
        self.video_out_path = f"{self.avatar_path}/vid_output/"
        self.mask_out_path = f"{self.avatar_path}/mask"
        self.mask_coords_path = f"{self.avatar_path}/mask_coords.pkl"
        self.avatar_info_path = f"{self.avatar_path}/avatar_info.json"
        self.avatar_info = {
            "avatar_id": avatar_id,
            "video_path": video_path,
            "bbox_shift": bbox_shift,
            "version": self.version
        }

        # Model related
        self.device = None
        self.timesteps = None
        self.vae = None
        self.unet = None
        self.pe = None
        self.weight_dtype = None
        self.audio_processor = None
        self.whisper = None
        self.fp = None

        # Data related
        self.input_latent_list_cycle = None
        self.coord_list_cycle = None
        self.frame_list_cycle = None
        self.mask_coords_list_cycle = None
        self.mask_list_cycle = None

        # Initialization
        self.idx = 0
        self.init()

    def init(self):
        """
        根据 preparation 参数决定是否重新准备素材
        若已存在且用户选择重建，则删除旧目录并调用 prepare_material
        否则加载已保存的预处理数据（潜在向量、坐标、遮罩等）
        处理版本变更情况，若bbox_shift改变需重新创建
        """
        # 1. Check if data preparation is needed
        required_files = [
            self.latents_out_path,      # latent features file
            self.coords_path,           # face coordinates file
            self.mask_coords_path,      # mask coordinates file
        ]

        # Check if data needs to be generated
        need_preparation = self.preparation  # If force regeneration, set to True

        if not need_preparation and os.path.exists(self.avatar_path):
            # Check if all required files exist
            for file_path in required_files:
                if not os.path.exists(file_path):
                    need_preparation = True
                    break

            # If config file exists, check if bbox_shift has changed
            if os.path.exists(self.avatar_info_path):
                with open(self.avatar_info_path, "r") as f:
                    avatar_info = json.load(f)
                if avatar_info['bbox_shift'] != self.avatar_info['bbox_shift']:
                    logger.error(f"bbox_shift changed from {avatar_info['bbox_shift']} to {self.avatar_info['bbox_shift']}, need re-preparation")
                    need_preparation = True
        else:
            need_preparation = True

        # 2. Initialize device and models
        self.device = torch.device(f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu")
        self.timesteps = torch.tensor([0], device=self.device)

        # Load model weights
        self.vae, self.unet, self.pe = load_all_model(
            unet_model_path=self.unet_model_path,
            vae_type=self.vae_type,
            unet_config=self.unet_config,
            device=self.device
        )

        # Convert to half precision
        self.pe = self.pe.half().to(self.device)
        self.vae.vae = self.vae.vae.half().to(self.device)
        self.unet.model = self.unet.model.half().to(self.device)
        self.weight_dtype = self.unet.model.dtype

        # Initialize audio processor and Whisper model
        self.audio_processor = AudioProcessor(feature_extractor_path=self.whisper_dir)
        self.whisper = WhisperModel.from_pretrained(self.whisper_dir)
        self.whisper = self.whisper.to(device=self.device, dtype=self.weight_dtype).eval()
        self.whisper.requires_grad_(False)

        # Initialize face parser with configurable parameters based on version
        if self.version == "v15":
            self.fp = FaceParsing(
                left_cheek_width=self.left_cheek_width,
                right_cheek_width=self.right_cheek_width
            )
        else:
            self.fp = FaceParsing()

        # 3. Prepare or load data
        if need_preparation:
            print("*********************************")
            print(f"  creating avator: {self.avatar_id}")
            print("*********************************")
            if os.path.exists(self.avatar_path):
                shutil.rmtree(self.avatar_path)
            osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
            self.prepare_material()
        else:
            logger.info(f"Avatar {self.avatar_id} exists and is complete, loading existing data...")
            self.input_latent_list_cycle = torch.load(self.latents_out_path)
            with open(self.coords_path, 'rb') as f:
                self.coord_list_cycle = pickle.load(f)
            input_img_list = glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
            input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            self.frame_list_cycle = read_imgs(input_img_list)
            with open(self.mask_coords_path, 'rb') as f:
                self.mask_coords_list_cycle = pickle.load(f)
            input_mask_list = glob.glob(os.path.join(self.mask_out_path, '*.[jpJP][pnPN]*[gG]'))
            input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            self.mask_list_cycle = read_imgs(input_mask_list)

    def prepare_material(self):
        """
        准备推理所需的所有素材数据：
            1.保存头像信息到JSON文件
            2.处理输入视频/图像序列：
                若为视频文件则调用 video2imgs 转换为图像
                若为图像目录则复制文件
            3.提取人脸关键点和边界框：
                调用 get_landmark_and_bbox 获取面部特征
                对v15版本增加额外边距处理
            4.编码潜在向量：
                裁剪并缩放面部区域到256x256
                使用 vae.get_latents_for_unet 编码为潜在向量
            5.构建循环数据：
                将帧列表、坐标列表、潜在向量列表与其反转版本拼接
            6.生成遮罩数据：
                为每帧调用 get_image_prepare_material 生成面部遮罩
                保存遮罩图像和坐标信息
            7.保存所有预处理数据：
                使用pickle保存坐标和遮罩坐标
                使用torch.save保存潜在向量
        """
        logger.info("preparing data materials ... ...")

        # Step 1: Save basic avatar config info
        with open(self.avatar_info_path, "w") as f:
            json.dump(self.avatar_info, f)

        # Step 2: Process input source (support video file or image sequence)
        if os.path.isfile(self.video_path):
            # If input is a video file, use video2imgs to extract frames
            video2imgs(self.video_path, self.full_imgs_path, ext='png')
        else:
            # If input is an image directory, copy all png images directly
            logger.info(f"copy files in {self.video_path}")
            files = os.listdir(self.video_path)
            files.sort()
            files = [file for file in files if file.split(".")[-1] == "png"]
            for filename in files:
                shutil.copyfile(f"{self.video_path}/{filename}", f"{self.full_imgs_path}/{filename}")

        # Get all input image paths and sort
        input_img_list = sorted(glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]')))

        # Step 3: Extract face landmarks and bounding boxes
        logger.info("extracting landmarks...")
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, self.bbox_shift)

        # Step 4: Extract latent features
        input_latent_list = []
        idx = -1
        # maker if the bbox is not sufficient
        coord_placeholder = (0.0, 0.0, 0.0, 0.0)
        for bbox, frame in zip(coord_list, frame_list):
            idx = idx + 1
            if bbox == coord_placeholder:
                continue
            x1, y1, x2, y2 = bbox
            if self.version == "v15":
                y2 = y2 + self.extra_margin
                y2 = min(y2, frame.shape[0])
                coord_list[idx] = [x1, y1, x2, y2]  # 更新coord_list中的bbox
            crop_frame = frame[y1:y2, x1:x2]
            resized_crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            latents = self.vae.get_latents_for_unet(resized_crop_frame)
            input_latent_list.append(latents)

        # Step 5: Build cycle sequence (by forward + reverse order)
        self.frame_list_cycle = frame_list + frame_list[::-1]
        self.coord_list_cycle = coord_list + coord_list[::-1]
        self.input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        self.mask_coords_list_cycle = []
        self.mask_list_cycle = []

        # Step 6: Generate and save masks
        for i, frame in enumerate(tqdm(self.frame_list_cycle)):
            cv2.imwrite(f"{self.full_imgs_path}/{str(i).zfill(8)}.png", frame)

            x1, y1, x2, y2 = self.coord_list_cycle[i]
            if self.version == "v15":
                mode = self.parsing_mode
            else:
                mode = "raw"
            mask, crop_box = get_image_prepare_material(frame, [x1, y1, x2, y2], fp=self.fp, mode=mode)

            cv2.imwrite(f"{self.mask_out_path}/{str(i).zfill(8)}.png", mask)
            self.mask_coords_list_cycle += [crop_box]
            self.mask_list_cycle.append(mask)

        # Step 7: Save all processed data
        with open(self.mask_coords_path, 'wb') as f:
            pickle.dump(self.mask_coords_list_cycle, f)

        with open(self.coords_path, 'wb') as f:
            pickle.dump(self.coord_list_cycle, f)

        torch.save(self.input_latent_list_cycle, os.path.join(self.latents_out_path))

    def process_frames(self, res_frame_queue, video_len, skip_save_images):
        """
        处理生成的帧并进行融合：
            循环处理直到达到视频长度
            从结果队列获取生成的帧
            调整帧尺寸以匹配原始面部区域大小
            使用 get_image_blending 将生成的面部与原始帧融合
            根据 skip_save_images 决定是否保存结果图像
            更新索引计数器
        args:
            res_frame_queue：生成帧的队列
            video_len：要处理的总帧数
            skip_save_images：是否跳过保存中间帧图像
        """
        logger.info(video_len)
        while True:
            if self.idx >= video_len - 1:
                break
            try:
                res_frame = res_frame_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue

            bbox = self.coord_list_cycle[self.idx % (len(self.coord_list_cycle))]
            ori_frame = copy.deepcopy(self.frame_list_cycle[self.idx % (len(self.frame_list_cycle))])
            x1, y1, x2, y2 = bbox
            try:
                res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
            except:
                continue
            mask = self.mask_list_cycle[self.idx % (len(self.mask_list_cycle))]
            mask_crop_box = self.mask_coords_list_cycle[self.idx % (len(self.mask_coords_list_cycle))]
            combine_frame = get_image_blending(ori_frame,res_frame,bbox,mask,mask_crop_box)

            if skip_save_images is False:
                cv2.imwrite(f"{self.avatar_path}/tmp/{str(self.idx).zfill(8)}.png", combine_frame)
            self.idx = self.idx + 1

    @torch.no_grad()
    def generate_frames(self, whisper_chunks: torch.Tensor, start_idx: int, batch_size: int) -> list:
        t0 = time.time()
        # Ensure whisper_chunks shape is (B, 50, 384)
        if whisper_chunks.ndim == 2:
            whisper_chunks = whisper_chunks.unsqueeze(0)
        elif whisper_chunks.ndim == 3 and whisper_chunks.shape[0] == 1:
            pass
        B = whisper_chunks.shape[0]
        assert B == batch_size, f"whisper_chunks.shape[0] ({B}) != batch_size ({batch_size})"
        idx_list = [start_idx + i for i in range(batch_size)]
        latent_list = []
        t1 = time.time()
        for idx in idx_list:
            latent = self.input_latent_list_cycle[idx % len(self.input_latent_list_cycle)]
            if latent.dim() == 3:
                latent = latent.unsqueeze(0)
            latent_list.append(latent)
        latent_batch = torch.cat(latent_list, dim=0)  # [B, ...]
        t2 = time.time()
        audio_feature = self.pe(whisper_chunks.to(self.device))
        t3 = time.time()
        latent_batch = latent_batch.to(device=self.device, dtype=self.unet.model.dtype)
        t4 = time.time()
        pred_latents = self.unet.model(
            latent_batch,
            self.timesteps,
            encoder_hidden_states=audio_feature
        ).sample
        # # Force set pred_latents to all nan for debugging： unet get nan value
        # pred_latents[:] = float('nan')
        t5 = time.time()
        pred_latents = pred_latents.to(device=self.device, dtype=self.vae.vae.dtype)
        recon = self.vae.decode_latents(pred_latents)
        t6 = time.time()
        avg_time = (t6 - t0) / B if B > 0 else 0.0
        fps = 1.0 / avg_time if avg_time > 0 else 0.0
        logger.info(
            f"[PROFILE] generate_frames: start_idx={start_idx}, batch_size={batch_size}, "
            f"prep_whisper={t1-t0:.4f}s, prep_latent={t2-t1:.4f}s, pe={t3-t2:.4f}s, "
            f"latent_to={t4-t3:.4f}s, unet={t5-t4:.4f}s, vae={t6-t5:.4f}s, total={t6-t0:.4f}s, total_per_frame={avg_time:.4f}s, fps={fps:.2f}"
        )
        logger.info(f"latent_batch stats: min={latent_batch.min().item()}, max={latent_batch.max().item()}, mean={latent_batch.mean().item()}, nan_count={(torch.isnan(latent_batch).sum().item() if torch.isnan(latent_batch).any() else 0)}")
        logger.info(f"pred_latents stats: min={pred_latents.min().item()}, max={pred_latents.max().item()}, mean={pred_latents.mean().item()}, nan_count={(torch.isnan(pred_latents).sum().item() if torch.isnan(pred_latents).any() else 0)}")
        if isinstance(recon, np.ndarray):
            logger.info(f"recon stats: min={recon.min()}, max={recon.max()}, mean={recon.mean()}, nan_count={np.isnan(recon).sum()}")
        elif isinstance(recon, torch.Tensor):
            logger.info(f"recon stats: min={recon.min().item()}, max={recon.max().item()}, mean={recon.mean().item()}, nan_count={(torch.isnan(recon).sum().item() if torch.isnan(recon).is_floating_point() else 0)}")
        else:
            logger.info(f"recon type: {type(recon)}")
        return [(recon[i], idx_list[i]) for i in range(B)]

    @torch.no_grad()
    def generate_frames_unet(self, whisper_chunks: torch.Tensor, start_idx: int, batch_size: int) -> list:
        """
        批量音频特征unet推理生成帧的潜在特征
        whisper_chunks: 音频特征张量，形状 [B, 50, 384]
        start_idx: 起始帧索引
        batch_size: 批处理大小
        Return: [pred_latents, idx_list] 包含预测潜在向量和索引列表的列表
        """
        t0 = time.time()
        """
        确保输入张量维度正确，如果是2D张量则添加批次维度
        验证批次大小与输入张量的第一个维度匹配
        """
        if whisper_chunks.ndim == 2:
            whisper_chunks = whisper_chunks.unsqueeze(0)
        elif whisper_chunks.ndim == 3 and whisper_chunks.shape[0] == 1:
            pass
        B = whisper_chunks.shape[0]
        assert B == batch_size, f"whisper_chunks.shape[0] ({B}) != batch_size ({batch_size})"
        # 生成帧索引列表
        idx_list = [start_idx + i for i in range(batch_size)]
        """
        从循环潜在向量列表中获取对应索引的潜在向量
        处理维度不一致的情况
        将所有潜在向量拼接成批次张量
        """
        latent_list = []
        t1 = time.time()
        for idx in idx_list:
            latent = self.input_latent_list_cycle[idx % len(self.input_latent_list_cycle)]
            if latent.dim() == 3:
                latent = latent.unsqueeze(0)
            latent_list.append(latent)
        latent_batch = torch.cat(latent_list, dim=0)  # [B, ...]
        t2 = time.time()
        # 使用姿态编码器(position encoder, self.pe)处理音频特征
        audio_feature = self.pe(whisper_chunks.to(self.device))
        t3 = time.time()
        latent_batch = latent_batch.to(device=self.device, dtype=self.unet.model.dtype)
        t4 = time.time()
        # self.unet.model: UNet2DConditionModel
        # -> input: latent_batch: torch.Size([B, 8, 32, 32],torch.float32
        # -> input: timesteps: torch.Size([1],torch.int64
        # -> input: audio_feature: torch.Size([B, 50, 384],torch.float32
        # <- output: pred_latents: torch.Size([B, 4, 32, 32],torch.float32
        # UNet模型推理
        pred_latents = self.unet.model(
            latent_batch,
            self.timesteps,
            encoder_hidden_states=audio_feature
        ).sample
        t5 = time.time()
        avg_time = (t5 - t0) / B if B > 0 else 0.0
        logger.info(
            f"[PROFILE] generate_frames_unet: start_idx={start_idx}, batch_size={batch_size}, "
            f"prep_whisper={t1-t0:.4f}s, prep_latent={t2-t1:.4f}s, pe={t3-t2:.4f}s, "
            f"latent_to={t4-t3:.4f}s, unet={t5-t4:.4f}s, total={t5-t0:.4f}s, total_per_frame={avg_time:.4f}s"
        )
        logger.info(f"latent_batch stats: "
                    f"min={latent_batch.min().item()}, "
                    f"max={latent_batch.max().item()}, "
                    f"mean={latent_batch.mean().item()}, "
                    f"nan_count={(torch.isnan(latent_batch).sum().item() if torch.isnan(latent_batch).any() else 0)}")
        logger.info(f"pred_latents stats: "
                    f"min={pred_latents.min().item()}, "
                    f"max={pred_latents.max().item()}, "
                    f"mean={pred_latents.mean().item()}, "
                    f"nan_count={(torch.isnan(pred_latents).sum().item() if torch.isnan(pred_latents).any() else 0)}")
        return [pred_latents, idx_list]

    @torch.no_grad()
    def generate_frames_vae(self, pred_latents: torch.Tensor, idx_list: list, batch_size: int) -> list:
        """
        将UNet模型生成的潜在特征向量通过VAE解码器转换为可视化的图像帧
        pred_latents: UNet模型输出的潜在向量，形状为 [B, 4, 32, 32]
        idx_list: frame index list 帧索引列表
        batch_size: batch size 批处理大小
        Return: List of (recon, idx) tuples, length is batch_size
        """
        t0 = time.time()
        B = pred_latents.shape[0]
        assert B == batch_size, f"pred_latents.shape[0] ({B}) != batch_size ({batch_size})"
        pred_latents = pred_latents.to(device=self.device, dtype=self.vae.vae.dtype)
        # ----- self.vae.decode_latents: AutoencoderKL -----
        # -> input: pred_latents: torch.Size([B, 4, 32, 32],torch.float32
        # <- output: recon: numpy.ndarray: (B, 256, 256, 3),np.float32
        recon = self.vae.decode_latents(pred_latents)
        t1 = time.time()
        avg_time = (t1 - t0) / B if B > 0 else 0.0
        logger.info(
            f"[PROFILE] generate_frames: start_idx={idx_list[0]}, batch_size={batch_size}, "
            f"vae={t1-t0:.4f}s, total={t1-t0:.4f}s, total_per_frame={avg_time:.4f}s"
        )
        logger.info(f"pred_latents stats: min={pred_latents.min().item()}, max={pred_latents.max().item()}, mean={pred_latents.mean().item()}, nan_count={(torch.isnan(pred_latents).sum().item() if torch.isnan(pred_latents).any() else 0)}")
        if isinstance(recon, np.ndarray):
            logger.info(f"recon stats: min={recon.min()}, max={recon.max()}, mean={recon.mean()}, nan_count={np.isnan(recon).sum()}")
        elif isinstance(recon, torch.Tensor):
            logger.info(f"recon stats: min={recon.min().item()}, max={recon.max().item()}, mean={recon.mean().item()}, nan_count={(torch.isnan(recon).sum().item() if torch.isnan(recon).is_floating_point() else 0)}")
        else:
            logger.info(f"recon type: {type(recon)}")
        return [(recon[i], idx_list[i]) for i in range(B)]

    def generate_frames_avatar(self, res_frame, idx):
        """ 融合面部帧与原始视频帧生成视频帧 """
        t0 = time.time()
        """
        从循环列表中获取当前帧对应的人脸边界框坐标
        深拷贝原始帧作为背景基础
        """
        bbox = self.coord_list_cycle[idx % len(self.coord_list_cycle)]
        ori_frame = copy.deepcopy(self.frame_list_cycle[idx % len(self.frame_list_cycle)])
        t1 = time.time()
        """
        将生成的面部图像调整到与原始人脸区域相同的尺寸
        如果调整失败，则返回原始帧
        """
        x1, y1, x2, y2 = bbox
        try:
            res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
        except Exception as e:
            logger.opt(exception=True).error(f"generate_frames_avatar error: {str(e)}")
            return ori_frame
        t2 = time.time()
        if np.all(res_frame == 0):
            logger.warning(f"generate_frames_avatar: res_frame is all zero, return ori_frame, idx={idx}")
            return ori_frame
        """ 获取用于融合的面部遮罩及其坐标信息 """
        mask = self.mask_list_cycle[idx % len(self.mask_list_cycle)]
        mask_crop_box = self.mask_coords_list_cycle[idx % len(self.mask_coords_list_cycle)]
        t3 = time.time()
        """ 面部与原始帧融合 """
        # combine_frame = self.acc_get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)
        combine_frame = get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)
        t4 = time.time()
        total_time = t4 - t0
        fps = 1.0 / total_time if total_time > 0 else 0
        if fps < self.fps:
            logger.warning(f"[PROFILE] generate_frames_avatar fps is not enough, fps={fps:.2f}, self.fps={self.fps}")
        logger.info(
            f"[PROFILE] generate_frames_avatar: idx={idx}, ori_copy={t1-t0:.4f}s, "
            f"resize={t2-t1:.4f}s, mask_fetch={t3-t2:.4f}s, "
            f"blend={t4-t3:.4f}s, total={total_time:.4f}s, fps={fps:.2f}"
        )
        return combine_frame

    def generate_idle_frame(self, idx: int) -> np.ndarray:
        """ 生成空闲帧 保持自然的静态画面 """
        # Directly return a frame from the original frame cycle
        return self.frame_list_cycle[idx % len(self.frame_list_cycle)]

    @torch.no_grad()
    def inference(self, audio_path, out_vid_name, fps, skip_save_images):
        """
        推理生成avatar video
            audio_path：输入音频文件路径
            out_vid_name：输出视频名称（基于音频文件名）
            fps：视频帧率
            skip_save_images：是否跳过保存中间帧图像
        """
        """ 创建临时目录用于存储中间结果 """
        os.makedirs(self.avatar_path + '/tmp', exist_ok=True)
        logger.info("start inference")
        """
        音频特征提取：
            使用 audio_processor.get_audio_feature 提取音频特征
            使用 audio_processor.get_whisper_chunk 将音频分块
        """
        start_time = time.time()
        # Extract audio features
        whisper_input_features, librosa_length = self.audio_processor.get_audio_feature(
            audio_path,
            weight_dtype=self.weight_dtype
        )
        whisper_chunks = self.audio_processor.get_whisper_chunk(
            whisper_input_features,
            self.device,
            self.weight_dtype,
            self.whisper,
            librosa_length,
            fps=fps,
            audio_padding_length_left=self.audio_padding_length_left,
            audio_padding_length_right=self.audio_padding_length_right,
        )
        logger.info(f"processing audio:{audio_path} costs {(time.time() - start_time) * 1000}ms")
        """
        设置并行处理：
            创建结果帧队列
            启动处理线程执行 process_frames
        """
        video_num = len(whisper_chunks)
        res_frame_queue = queue.Queue()
        self.idx = 0
        # Create a sub-thread and start it
        process_thread = threading.Thread(
            target=self.process_frames,
            args=(res_frame_queue, video_num, skip_save_images)
        )
        process_thread.start()

        """
        批处理推理：
            使用 datagen 生成器按批次处理音频块和潜在向量
            通过姿态编码器(pe)处理音频特征
            使用UNet模型生成预测潜在向量
            通过VAE解码器将潜在向量解码为图像帧
            将结果帧放入队列供处理线程使用
        """
        gen = datagen(whisper_chunks,
                      self.input_latent_list_cycle,
                      self.batch_size)
        start_time = time.time()

        for i, (whisper_batch, latent_batch) in enumerate(tqdm(gen, total=int(np.ceil(float(video_num) / self.batch_size)))):
            audio_feature_batch = self.pe(whisper_batch.to(self.device))
            latent_batch = latent_batch.to(device=self.device, dtype=self.unet.model.dtype)

            pred_latents = self.unet.model(
                latent_batch,
                self.timesteps,
                encoder_hidden_states=audio_feature_batch
            ).sample
            pred_latents = pred_latents.to(device=self.device, dtype=self.vae.vae.dtype)
            recon = self.vae.decode_latents(pred_latents)
            for res_frame in recon:
                res_frame_queue.put(res_frame)
        # Close the queue and sub-thread after all tasks are completed
        process_thread.join()

        if skip_save_images is True:
            logger.info('Total process time of {} frames without saving images = {}s'.format(
                video_num,
                time.time() - start_time))
        else:
            logger.info('Total process time of {} frames including saving images = {}s'.format(
                video_num,
                time.time() - start_time))

        """
        视频合成：
            使用ffmpeg将图像序列合成视频
            合并音频和视频生成最终输出文件
            清理临时文件
        """
        if out_vid_name is not None and skip_save_images is False:
            # 1. 图片序列转视频
            cmd_img2video = f"ffmpeg -y -v warning -r {fps} -f image2 -i {self.avatar_path}/tmp/%08d.png -vcodec libx264 -vf format=yuv420p -crf 18 {self.avatar_path}/temp.mp4"
            logger.info(cmd_img2video)
            os.system(cmd_img2video)

            # 2. 将音频合并到视频中
            output_vid = os.path.join(self.video_out_path, out_vid_name + ".mp4")
            cmd_combine_audio = f"ffmpeg -y -v warning -i {audio_path} -i {self.avatar_path}/temp.mp4 {output_vid}"
            logger.info(cmd_combine_audio)
            os.system(cmd_combine_audio)

            # 3. 清理临时文件
            os.remove(f"{self.avatar_path}/temp.mp4")
            shutil.rmtree(f"{self.avatar_path}/tmp")
            logger.info(f"result is save to {output_vid}")
        logger.info("\n")

    @torch.no_grad()
    def extract_whisper_feature(self, segment: np.ndarray, sampling_rate: int) -> torch.Tensor:
        """ 提取单个音频片段的特征 """
        t0 = time.time()
        audio_feature = self.audio_processor.feature_extractor(
            segment,
            return_tensors="pt",
            sampling_rate=sampling_rate
        ).input_features
        if self.weight_dtype is not None:
            audio_feature = audio_feature.to(dtype=self.weight_dtype)
        whisper_chunks = self.audio_processor.get_whisper_chunk(
            [audio_feature],
            self.device,
            self.weight_dtype,
            self.whisper,
            len(segment),
            fps=self.fps,
            audio_padding_length_left=self.audio_padding_length_left,
            audio_padding_length_right=self.audio_padding_length_right,
        )
        t1 = time.time()
        logger.info(f"[PROFILE] extract_whisper_feature: duration={t1-t0:.4f}s, segment_len={len(segment)}, "
                    f"sampling_rate={sampling_rate}")
        return whisper_chunks
