import base64
import json
from pathlib import PurePosixPath
from typing import Any, Optional

from aidial_client import AsyncDial
from aidial_sdk.chat_completion import Message, Attachment
from pydantic import StrictStr, AnyUrl

from task.tools.base import BaseTool
from task.tools.py_interpreter._response import _ExecutionResult
from task.tools.mcp.mcp_client import MCPClient
from task.tools.mcp.mcp_tool_model import MCPToolModel
from task.tools.models import ToolCallParams


class PythonCodeInterpreterTool(BaseTool):
    """
    Uses https://github.com/khshanovskyi/mcp-python-code-interpreter PyInterpreter MCP Server.

    ⚠️ Pay attention that this tool will wrap all the work with PyInterpreter MCP Server.
    """

    def __init__(
            self,
            mcp_client: MCPClient,
            mcp_tool_models: list[MCPToolModel],
            tool_name: str,
            dial_endpoint: str,
    ):
        """
        :param tool_name: it must be actual name of tool that executes code. It is 'execute_code'.
            https://github.com/khshanovskyi/mcp-python-code-interpreter/blob/main/interpreter/server.py#L303
        """
        self.dial_endpoint = dial_endpoint
        self.mcp_client = mcp_client
        self._code_execute_tool: Optional[MCPToolModel] = None
        
        for tool_model in mcp_tool_models:
            if tool_model.name == tool_name:
                self._code_execute_tool = tool_model
                break
        
        if self._code_execute_tool is None:
            raise ValueError(f"Cannot set up PythonCodeInterpreterTool without tool '{tool_name}' that executes code")

    @classmethod
    async def create(
            cls,
            mcp_url: str,
            tool_name: str,
            dial_endpoint: str,
    ) -> 'PythonCodeInterpreterTool':
        """Async factory method to create PythonCodeInterpreterTool"""
        mcp_client = await MCPClient.create(mcp_url)
        mcp_tool_models = await mcp_client.get_tools()
        return cls(mcp_client, mcp_tool_models, tool_name, dial_endpoint)

    @property
    def show_in_stage(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return self._code_execute_tool.name

    @property
    def description(self) -> str:
        return self._code_execute_tool.description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._code_execute_tool.parameters

    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        arguments = json.loads(tool_call_params.tool_call.function.arguments)
        code = arguments["code"]
        session_id = arguments.get("session_id")
        stage = tool_call_params.stage
        
        stage.append_content("## Request arguments: \n")
        stage.append_content(f"```python\n\r{code}\n\r```\n\r")
        
        if session_id is not None and session_id != 0:
            stage.append_content(f"**session_id**: {session_id}\n\r")
        else:
            stage.append_content("New session will be created\n\r")
        
        response = await self.mcp_client.call_tool(self._code_execute_tool.name, arguments)
        
        response_json = json.loads(response)
        execution_result = _ExecutionResult.model_validate(response_json)
        
        if execution_result.files:
            dial_client = AsyncDial(
                base_url=self.dial_endpoint,
                api_key=tool_call_params.api_key
            )
            files_home = await dial_client.my_appdata_home()
            
            for file_ref in execution_result.files:
                file_name = file_ref.name
                mime_type = file_ref.mime_type
                
                resource_content = await self.mcp_client.get_resource(AnyUrl(file_ref.uri))
                
                if mime_type.startswith('text/') or mime_type in ['application/json', 'application/xml']:
                    if isinstance(resource_content, str):
                        file_bytes = resource_content.encode('utf-8')
                    else:
                        file_bytes = resource_content
                else:
                    if isinstance(resource_content, str):
                        file_bytes = base64.b64decode(resource_content)
                    else:
                        file_bytes = resource_content
                
                upload_url = f"files/{(files_home / file_name).as_posix()}"
                await dial_client.files.upload(upload_url, file_bytes)
                
                attachment = Attachment(
                    url=StrictStr(upload_url),
                    type=StrictStr(mime_type),
                    title=StrictStr(file_name)
                )
                
                stage.add_attachment(
                    type=mime_type,
                    title=file_name,
                    url=upload_url
                )
                tool_call_params.choice.append_content(f"\n\r[{file_name}]({upload_url})\n\r")
        
        if execution_result.output:
            execution_result.output = [
                output[:1000] if len(output) > 1000 else output
                for output in execution_result.output
            ]
        
        stage.append_content(f"```json\n\r{execution_result.model_dump_json(indent=2)}\n\r```\n\r")
        
        return execution_result.model_dump_json()
