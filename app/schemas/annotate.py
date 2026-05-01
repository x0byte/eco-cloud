from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class AnnotateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uuid: UUID = Field(
        ...,
        description="Unique request identifier supplied by the client.",
    )
    image: str = Field(
        ...,
        min_length=1,
        description="Base64-encoded image string.",
    )


class AnnotateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uuid: UUID = Field(..., description="Echo of the request identifier.")
    image: str = Field(..., description="Base64-encoded annotated image.")
