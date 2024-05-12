from datetime import datetime
from typing import List, Literal

from sqlalchemy import JSON, Column, DateTime, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Passage(Base):
    __tablename__ = "passage"

    id = Column(Integer, primary_key=True, autoincrement=True)
    type = Column(String(255), nullable=True)
    page_id = Column(String(255), nullable=True, unique=True)
    crawled_at = Column(DateTime, default=datetime.now, nullable=True)
    published_at = Column(DateTime, default=None, nullable=True)
    views_count = Column(Integer, default=None, nullable=True)
    tags = Column(JSON, nullable=True)
    title = Column(String(255), nullable=True)
    content = Column(Text, nullable=False)
    image_num = Column(Integer, nullable=True)
    next_page_ids = Column(JSON, nullable=True)
