from .minirag_base import SQLALCHEMY_BASE
from sqlalchemy import Column, String, DateTime, Integer, func, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship
from sqlalchemy import Index
from pydantic import BaseModel
import uuid


class DataChunk(SQLALCHEMY_BASE):

    __tablename__ = "chunks"

    chunk_id = Column(Integer, primary_key=True, autoincrement=True)
    chunk_uuid = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True, nullable=False)

    chunk_text = Column(String, nullable=False)
    chunk_order = Column(Integer, nullable=False)
    chunk_metadata = Column(JSONB, nullable=True)

    chunk_project_id = Column(Integer, ForeignKey("projects.project_id"), nullable=False)
    chunk_asset_id = Column(Integer, ForeignKey("assets.asset_id"), nullable=False)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), nullable=True)

    project = relationship("Project", back_populates="chunks")
    asset = relationship("Asset", back_populates="chunks")

    __table_args__ = (
        Index("ix_chunk_project_id", chunk_project_id),
        Index("ix_chunk_asset_id", chunk_asset_id),
    )


class RetrievedDocument(BaseModel):
    text: str
    score: float