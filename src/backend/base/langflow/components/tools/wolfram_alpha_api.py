from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper

from langflow.custom import Component
from langflow.field_typing import Tool
from langflow.inputs import MultilineInput, SecretStrInput
from langflow.schema import Data
from langflow.io import Output
from langflow.schema.message import Message


class WolframAlphaAPIComponent(Component):
    display_name = "WolframAlpha API"
    description = """Enables queries to Wolfram Alpha for computational data, facts, and calculations across various \
topics, delivering structured responses."""
    name = "WolframAlphaAPI"

    inputs = [
        MultilineInput(
            name="input_value", display_name="Input Query", info="Example query: 'What is the population of France?'", tool_mode=True
        ),
        SecretStrInput(name="app_id", display_name="App ID", required=True),
    ]

    outputs = [
        Output(display_name="Data", name="data", method="fetch_content"),
        Output(display_name="Text", name="text", method="fetch_content_text"),
    ]

    icon = "WolframAlphaAPI"

    def run_model(self) -> list[Data]:
        return self.fetch_content()

    def fetch_content(self) -> list[Data]:
        wrapper = self._build_wrapper()
        result_str = wrapper.run(self.input_value)
        data = [Data(text=result_str, data={"result": result_str})]
        self.status = data
        return data

    def fetch_content_text(self) -> Message:
        data = self.fetch_content()
        result_string = ""
        for item in data:
            result_string += item.text + "\n"
        self.status = result_string
        return Message(text=result_string)

    def _build_wrapper(self) -> WolframAlphaAPIWrapper:
        return WolframAlphaAPIWrapper(wolfram_alpha_appid=self.app_id)
