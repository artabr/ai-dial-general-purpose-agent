import asyncio
import json
from typing import Any

from aidial_client import AsyncDial
from aidial_client.types.chat.legacy.chat_completion import ToolCall
from aidial_sdk.chat_completion import Message, Role, Choice, Request, Response

from task.tools.base import BaseTool
from task.tools.models import ToolCallParams
from task.utils.constants import TOOL_CALL_HISTORY_KEY
from task.utils.history import unpack_messages
from task.utils.stage import StageProcessor


class GeneralPurposeAgent:
    def __init__(
        self,
        endpoint: str,
        system_prompt: str,
        tools: list[BaseTool],
    ):
        self.endpoint = endpoint
        self.system_prompt = system_prompt
        self.tools = tools
        self._tools_dict: dict[str, BaseTool] = {tool.name: tool for tool in tools}
        self.state: dict[str, Any] = {TOOL_CALL_HISTORY_KEY: []}

    async def handle_request(
        self, deployment_name: str, choice: Choice, request: Request, response: Response
    ) -> Message:
        client = AsyncDial(
            base_url=self.endpoint,
            api_key=request.api_key,
            api_version=request.api_version,
        )

        chunks = await client.chat.completions.create(
            deployment_name=deployment_name,
            messages=self._prepare_messages(request.messages),
            tools=[tool.schema for tool in self.tools],
            stream=True,
        )

        # key: tool_call index -> accumulated tool_call data dict
        tool_call_index_map: dict[int, dict[str, Any]] = {}
        content = ""

        async for chunk in chunks:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta:
                    if delta.content:
                        choice.append_content(delta.content)
                        content += delta.content
                    if delta.tool_calls:
                        for tool_call_delta in delta.tool_calls:
                            if tool_call_delta.id:
                                # First chunk of this tool call — initialise entry
                                tool_call_index_map[tool_call_delta.index] = {
                                    "index": tool_call_delta.index,
                                    "id": tool_call_delta.id,
                                    "type": "function",
                                    "function": {
                                        "name": tool_call_delta.function.name
                                        if tool_call_delta.function
                                        else "",
                                        "arguments": tool_call_delta.function.arguments
                                        or ""
                                        if tool_call_delta.function
                                        else "",
                                    },
                                }
                            else:
                                # Subsequent chunks — accumulate arguments
                                existing = tool_call_index_map[tool_call_delta.index]
                                if tool_call_delta.function:
                                    argument_chunk = (
                                        tool_call_delta.function.arguments or ""
                                    )
                                    existing["function"]["arguments"] += argument_chunk

        tool_calls: list[ToolCall] | None = (
            [ToolCall.validate(tc) for tc in tool_call_index_map.values()]
            if tool_call_index_map
            else None
        )

        assistant_message = Message(
            role=Role.ASSISTANT,
            content=content,
            tool_calls=tool_calls,  # type: ignore[arg-type]
        )

        if tool_calls:
            conversation_id = request.headers.get("x-conversation-id", "")
            tasks = [
                self._process_tool_call(
                    tool_call=tc,
                    choice=choice,
                    api_key=request.api_key,
                    conversation_id=conversation_id,
                )
                for tc in tool_calls
            ]
            tool_messages = await asyncio.gather(*tasks)

            self.state[TOOL_CALL_HISTORY_KEY].append(
                assistant_message.dict(exclude_none=True)
            )
            self.state[TOOL_CALL_HISTORY_KEY].extend(tool_messages)

            return await self.handle_request(deployment_name, choice, request, response)

        choice.set_state(self.state)
        return assistant_message

    def _prepare_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        unpacked = unpack_messages(messages, self.state[TOOL_CALL_HISTORY_KEY])

        unpacked.insert(
            0,
            {
                "role": Role.SYSTEM.value,
                "content": self.system_prompt,
            },
        )

        for msg in unpacked:
            print(json.dumps(msg, default=str))

        return unpacked

    async def _process_tool_call(
        self, tool_call: ToolCall, choice: Choice, api_key: str, conversation_id: str
    ) -> dict[str, Any]:
        tool_name = tool_call.function.name

        stage = StageProcessor.open_stage(choice, tool_name)

        tool = self._tools_dict[tool_name]

        if tool.show_in_stage:
            stage.append_content("## Request arguments: \n")
            stage.append_content(
                f"```json\n\r{json.dumps(json.loads(tool_call.function.arguments), indent=2)}\n\r```\n\r"
            )
            stage.append_content("## Response: \n")

        tool_call_params = ToolCallParams(
            tool_call=tool_call,
            stage=stage,
            choice=choice,
            api_key=api_key,
            conversation_id=conversation_id,
        )

        tool_message = await tool.execute(tool_call_params)

        StageProcessor.close_stage_safely(stage)

        return tool_message.dict(exclude_none=True)
