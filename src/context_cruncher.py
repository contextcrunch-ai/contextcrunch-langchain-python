from typing import Callable, Dict, List
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, root_validator
from langchain_core.utils.utils import build_extra_kwargs, convert_to_secret_str
from langchain_core.utils import (
    check_package_version,
    get_from_env,
    get_pydantic_field_names,
)


class ContextCruncher(RunnableLambda):
    
    def __init__(self, compression_ratio=0.9):
        super().__init__(self.call)
        if compression_ratio <= 0.5 or compression_ratio >= 1 :
            raise Exception("Compression ratio must be between 0.5 and 1 (exclusive)")
        self.compression_ratio = compression_ratio
        print('validate_environment')
        self.contextcrunch_api_key = convert_to_secret_str(
            get_from_env( "contextcrunch_api_key", "CONTEXTCRUNCH_API_KEY")
        )
        # Get custom api url from environment.
        self.contextcrunch_api_url = get_from_env(
            "contextcrunch_api_url",
            "CONTEXTCRUNCH_API_URL",
            default="https://contextcrunch.com/api",
        )
        try:
            import contextcrunch

            check_package_version("contextcrunch", gte_version="1.0.0")
            self.client = contextcrunch.ContextCrunchClient(
                api_key=self.contextcrunch_api_key.get_secret_value(),
                url=self.contextcrunch_api_url
            )

        except ImportError:
            raise ImportError(
                "Could not import contextcrunch python package. "
                "Please it install it with `pip install contextcrunch`."
            )
    
    
    def call(self, input: Dict) -> Dict:
        context = input["context"]
        prompt = input["question"]
        compressed_context = self.client.compress(context, prompt)
        return {"context": compressed_context, "question": prompt}