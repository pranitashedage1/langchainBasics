# I have API key for openAi but do not have credits so this will give 429 error.
# this file is to get the description of the image using gpt-4.1-mini model 
# which is multimodal and can process images as well.

from dotenv import load_dotenv
# from base64 import b64encode
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage

load_dotenv()

model = init_chat_model("gpt-4.1-mini")

message = HumanMessage(
    content=[
            {
                "type": "text", 
                "text": "Describe the content of this image in detail."
            },
            {
                "type": "image", 
                "url": "https://media.licdn.com/dms/image/v2/D4E22AQFFFmUsOoZyag/feedshare-shrink_1280/feedshare-shrink_1280/0/1715552071442?e=1775692800&v=beta&t=rYafA-yuMAj_zGvWEcHiLbBhLFOEl0cmQtrTFQuriVw",
                # "base64": b64encode(open("image.jpg", "rb").read()).decode("utf-8")
                # "mime_type": "image/jpeg"
            }
        ]
    )


response = model.invoke([message])

print(response.content)

