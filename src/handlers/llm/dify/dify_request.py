import json

import PIL
import numpy as np
import requests
from loguru import logger

from src.handlers.llm.dify.llm_handler_base import DifyContext


class DifyRequest:

    @staticmethod
    def file_upload(context: DifyContext, image_data):
        """
        上传图像
        """
        upload_url = f"{context.config.api_url}/files/upload"
        headers = {
            "Authorization": f"Bearer {context.config.api_key}"
        }

        # 假设 image_data 是 PIL 图像或 numpy 数组，需先保存为临时文件
        from io import BytesIO
        buffered = BytesIO()

        image = PIL.Image.fromarray(np.squeeze(image_data)[..., ::-1])
        image.save(buffered, format="JPEG")  # 可根据需要调整格式
        img_bytes = buffered.getvalue()

        files = {
            'file': ('image.jpg', img_bytes, 'image/jpeg'),
        }

        response = requests.post(upload_url, headers=headers, files=files, data={'user': context.session_id})
        if response.status_code < 400:
            result = response.json()
            return result.get("id")
        else:
            logger.error(f"Failed to upload image:{response.status_code} {response.text}")
            return None

    @staticmethod
    def chat_messages(context: DifyContext, query: str, images=None):
        """
        发送请求到 Dify API
        """
        url = f"{context.config.api_url}/chat-messages"
        headers = {
            "Authorization": f"Bearer {context.config.api_key}",
            "Content-Type": "application/json"
        }

        # 如果有图片，需要特殊处理
        files = []
        if images and len(images) > 0:
            for img in images:
                if img is not None:
                    file_id = DifyRequest.file_upload(context, img)
                    if file_id:
                        files.append({
                            "type": "image",
                            "transfer_method": "local_file",
                            "upload_file_id": file_id
                        })

        payload = {
            "inputs": {},
            "query": query,
            "response_mode": context.config.response_mode,
            "conversation_id": context.conversation_id,
            "user": context.session_id
        }

        if files and len(files) > 0:
            payload["files"] = files

        try:
            if context.config.response_mode == "streaming":
                with requests.post(url, headers=headers, json=payload, stream=True,
                                   timeout=context.config.timeout) as response:
                    if response.status_code != 200:
                        error_text = response.text
                        logger.error(f"Dify API request failed with: {response.status_code} - {error_text}")
                        yield f"Error: {response.status_code} - {error_text}"
                        return

                    for line in response.iter_lines():
                        if line:
                            line_str = line.decode('utf-8')
                            if line_str.startswith("data: "):
                                data = line_str[6:]
                                if data.strip() == "[DONE]":
                                    break
                                try:
                                    json_data = json.loads(data)
                                    if "answer" in json_data:
                                        yield json_data["answer"]
                                    elif "conversation_id" in json_data and json_data["conversation_id"]:
                                        context.conversation_id = json_data["conversation_id"]
                                except json.JSONDecodeError:
                                    logger.warning(f"Failed to parse JSON: {data}")
            else:
                response = requests.post(url, headers=headers, json=payload, timeout=context.config.timeout)
                if response.status_code != 200:
                    error_text = response.text
                    logger.error(f"Dify API request failed with: {response.status_code} - {error_text}")
                    yield f"Error: {response.status_code} - {error_text}"
                    return

                json_data = response.json()
                if "answer" in json_data:
                    yield json_data["answer"]
                if "conversation_id" in json_data and json_data["conversation_id"]:
                    context.conversation_id = json_data["conversation_id"]

        except requests.exceptions.Timeout:
            logger.error("Dify API request timeout")
            yield "Error: Dify API request timed out"
        except requests.exceptions.RequestException as e:
            logger.error(f"Dify API request error: {str(e)}")
            yield f"Error: Dify API request failed with: {str(e)}"
        except Exception as e:
            logger.error(f"Dify API Unexpected error when calling: {str(e)}")
            yield f"Error: Dify API Unexpected error occurred: {str(e)}"
