from openai import OpenAI
import os
import base64
import mimetypes
import requests
import json


client = OpenAI(
    api_key="None",
    base_url="http://localhost:33335/v1",
)
image_path = './weigui.jpeg'


completion = client.chat.completions.create(
    model="VQA",
    messages=[{
            "role": "user",
            "content": [{
                "type": "text",
                "text": "这是一个问题上报的场景，请从城市治理的角度，描述一下发生了什么违规现象，以市民问题上报的口吻，以发现什么现象为开头，要求语气严肃描述严谨，内容不超过50字。"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": "http://p7.itc.cn/images03/20200514/3ea2f5f7908a4ee082103aee14f3fb78.jpeg"
                }
            }]
        }],
)

print(completion)
