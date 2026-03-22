import os

import uvicorn
from aidial_sdk import DIALApp
from aidial_sdk.chat_completion import ChatCompletion, Request, Response

from task.agent import GeneralPurposeAgent
from task.prompts import SYSTEM_PROMPT
from task.tools.base import BaseTool
from task.tools.deployment.image_generation_tool import ImageGenerationTool
from task.tools.files.file_content_extraction_tool import FileContentExtractionTool
from task.tools.mcp.mcp_client import MCPClient
from task.tools.mcp.mcp_tool import MCPTool
from task.tools.rag.document_cache import DocumentCache
from task.tools.rag.rag_tool import RagTool

DIAL_ENDPOINT = os.getenv("DIAL_ENDPOINT", "http://localhost:8080")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME", "gpt-4o")
# DEPLOYMENT_NAME = os.getenv('DEPLOYMENT_NAME', 'claude-haiku-4-5')


class GeneralPurposeAgentApplication(ChatCompletion):

    def __init__(self):
        self.tools: list[BaseTool] = []

    async def _get_mcp_tools(self, url: str) -> list[BaseTool]:
        # 1. Create list of BaseTool
        tools: list[BaseTool] = []
        # 2. Create MCPClient
        mcp_client = await MCPClient.create(url)
        # 3. Get tools, wrap each as MCPTool
        mcp_tool_models = await mcp_client.get_tools()
        for mcp_tool_model in mcp_tool_models:
            tools.append(MCPTool(mcp_client, mcp_tool_model))
        # 4. Return tool list
        return tools

    async def _create_tools(self) -> list[BaseTool]:
        # 1. Create list of BaseTool
        tools: list[BaseTool] = []
        # 2. Add ImageGenerationTool
        tools.append(ImageGenerationTool(endpoint=DIAL_ENDPOINT))
        # 3. Add FileContentExtractionTool
        tools.append(FileContentExtractionTool(DIAL_ENDPOINT))
        # 4. Add RagTool
        tools.append(RagTool(DIAL_ENDPOINT, DEPLOYMENT_NAME, DocumentCache.create()))
        # 5. Add PythonCodeInterpreterTool
        # 6. Extend with MCP tools from localhost:8051
        mcp_tools = await self._get_mcp_tools("http://localhost:8051/mcp/")
        tools.extend(mcp_tools)
        return tools

    async def chat_completion(self, request: Request, response: Response) -> None:
        # 1. Lazily initialise tools
        if not self.tools:
            self.tools = await self._create_tools()
        # 2. Create single choice and run agent
        with response.create_single_choice() as choice:
            agent = GeneralPurposeAgent(
                endpoint=DIAL_ENDPOINT,
                system_prompt=SYSTEM_PROMPT,
                tools=self.tools,
            )
            await agent.handle_request(
                choice=choice,
                deployment_name=DEPLOYMENT_NAME,
                request=request,
                response=response,
            )


# 1. Create DIALApp
app = DIALApp()
# 2. Create GeneralPurposeAgentApplication
agent_app = GeneralPurposeAgentApplication()
# 3. Register with deployment name
app.add_chat_completion(deployment_name="general-purpose-agent", impl=agent_app)
# 4. Run with uvicorn
if __name__ == "__main__":
    uvicorn.run(app, port=5030, host="0.0.0.0")
