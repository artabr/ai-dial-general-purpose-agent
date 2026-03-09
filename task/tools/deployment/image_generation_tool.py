from typing import Any

from aidial_sdk.chat_completion import Message
from pydantic import StrictStr

from task.tools.deployment.base import DeploymentTool
from task.tools.models import ToolCallParams


class ImageGenerationTool(DeploymentTool):

    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        result = await super()._execute(tool_call_params)

        if result.custom_content and result.custom_content.attachments:
            for attachment in result.custom_content.attachments:
                if attachment.type in ("image/png", "image/jpeg"):
                    tool_call_params.choice.append_content(f"\n\r![image]({attachment.url})\n\r")

            if not result.content:
                result.content = StrictStr(
                    "The image has been successfully generated according to request and shown to user!"
                )

        return result

    @property
    def deployment_name(self) -> str:
        return "dall-e-3"

    @property
    def name(self) -> str:
        return "image_generation_tool"

    @property
    def description(self) -> str:
        return (
            "Generates an image using DALL-E 3 based on a text description. "
            "Use this tool when the user requests image creation, illustration, or visual content generation. "
            "The generated image will be displayed directly in the chat. "
            "Provide a detailed, descriptive prompt for best results. "
            "Note: DALL-E 3 does not support editing existing images, only generating new ones."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Extensive description of the image that should be generated.",
                },
                "size": {
                    "type": "string",
                    "description": "The size of the generated image.",
                    "enum": ["1024x1024", "1792x1024", "1024x1792"],
                    "default": "1024x1024"
                },
                "quality": {
                    "type": "string",
                    "description": "The quality of the generated image.",
                    "enum": ["standard", "hd"],
                    "default": "standard"
                },
                "style": {
                    "type": "string",
                    "description": "The style of the generated image.",
                    "enum": ["vivid", "natural"],
                    "default": "natural"
                },
            },
            "required": ["prompt"],
        }
