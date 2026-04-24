"""Pydantic request and response schemas."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., description="用户的文字咨询，必填")
    images: List[str] = Field(default_factory=list, description="Base64 格式图片列表，可选，支持 0-3 张")
    session_id: Optional[str] = Field(None, description="会话ID，可选，若传入则沿用")
    stream: bool = Field(default=False, description="是否流式输出，默认 False")

    class Config:
        json_schema_extra = {
            "example": {
                "question": "请问电钻指示灯闪烁是什么意思？",
                "images": [],
                "session_id": "test-session-001",
                "stream": False,
            }
        }


class ChatData(BaseModel):
    answer: str = Field(..., description="核心输出的字符串")
    session_id: str = Field(..., description="关联的会话ID")
    timestamp: int = Field(..., description="当前秒级时间戳")


class ChatResponse(BaseModel):
    code: int = Field(default=0, description="状态码，0 表示成功")
    msg: str = Field(default="success", description="状态信息")
    data: ChatData = Field(..., description="响应数据")

