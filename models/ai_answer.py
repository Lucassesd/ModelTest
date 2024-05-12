from datetime import datetime
from typing import Literal, List

from sqlalchemy import Column, DateTime, Integer, String, Text,JSON
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class AI_answer(Base):
    __tablename__ = "ai_answer"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    type: Literal["juejin"] = Column(String(255), nullable=False)
    page_id: str = Column(String(255), nullable=False, unique=True)
    processing_at: datetime = Column(DateTime, default=datetime.now, nullable=False)
    keywords: List[str] = Column(JSON, nullable=False)
    content: str = Column(Text, nullable=False)
