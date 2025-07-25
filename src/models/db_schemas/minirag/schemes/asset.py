from .minirag_base import SQLALCHEMY_BASE
from sqlalchemy import Column, String, DateTime, Integer, func, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship
from sqlalchemy import Index
import uuid

class Asset(SQLALCHEMY_BASE):

    __tablename__ = "assets"
    asset_id = Column(Integer, primary_key=True, autoincrement=True)
    asset_uuid = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True, nullable=False)

    asset_type = Column(String, nullable=False)
    asset_name = Column(String, nullable=False)
    asset_size = Column(Integer, nullable=False)    
    asset_config = Column(JSONB, nullable=True)

    asset_project_id = Column(Integer, ForeignKey("projects.project_id"), nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), nullable=True)

    project = relationship("Project", back_populates="assets")
    chunks = relationship("DataChunk", back_populates="asset")

    __table_args__ = (
        Index("ix_asset_type", asset_type),
        Index("ix_asset_project_id", asset_project_id),
    )
