from google import genai
from google.genai import types
from termcolor import cprint
from .base_model import BaseModel, ModelResponse

class GeminiModel(BaseModel):
    """Implementation for Google's Gemini models using the new google.genai SDK"""

    AVAILABLE_MODELS = {
        "gemini-2.5-pro": "Most advanced Gemini 2.5 model with superior capabilities",
        "gemini-2.5-flash": "Fast Gemini 2.5 model for quick responses",
        "gemini-2.5-flash-lite": "Ultra-fast lightweight Gemini 2.5 model"
    }

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash", **kwargs):
        self.model_name = model_name
        super().__init__(api_key, **kwargs)

    def initialize_client(self, **kwargs) -> None:
        """Initialize the Gemini client using the new google.genai SDK"""
        try:
            self.client = genai.Client(api_key=self.api_key)
            cprint(f"[+] Initialized Gemini model: {self.model_name}", "green")
        except Exception as e:
            cprint(f"[X] Failed to initialize Gemini model: {str(e)}", "red")
            self.client = None

    def generate_response(self,
        system_prompt: str,
        user_content: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> ModelResponse:
        """Generate a response using Gemini"""
        try:
            # Build config with safety settings
            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                system_instruction=system_prompt,
                safety_settings=[
                    types.SafetySetting(
                        category="HARM_CATEGORY_HARASSMENT",
                        threshold="BLOCK_ONLY_HIGH",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_HATE_SPEECH",
                        threshold="BLOCK_ONLY_HIGH",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        threshold="BLOCK_ONLY_HIGH",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="BLOCK_ONLY_HIGH",
                    ),
                ],
            )

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_content,
                config=config,
            )

            # Check if response was blocked or empty
            if not response.candidates or not response.candidates[0].content or not response.candidates[0].content.parts:
                finish_reason = None
                if response.candidates and len(response.candidates) > 0:
                    finish_reason = getattr(response.candidates[0], 'finish_reason', None)

                error_msg = "Empty response from Gemini"
                if finish_reason:
                    error_msg += f", finish_reason={finish_reason}"
                raise Exception(error_msg)

            return ModelResponse(
                content=response.text.strip(),
                raw_response=response,
                model_name=self.model_name,
                usage=None
            )

        except Exception as e:
            cprint(f"[X] Gemini generation error: {str(e)}", "red")
            raise

    def is_available(self) -> bool:
        """Check if Gemini is available"""
        return self.client is not None

    @property
    def model_type(self) -> str:
        return "gemini"
