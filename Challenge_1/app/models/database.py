"""
SQLite database models using SQLAlchemy.
Why SQLAlchemy: Industry standard ORM, type-safe, migrations support.
"""

from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import uuid

from app.core.config import settings

Base = declarative_base()


class ImageModel(Base):
    """
    Store uploaded image metadata.
    Why: Track uploaded images and link to detected coins.
    """

    __tablename__ = "images"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String)
    file_path = Column(String, nullable=False)
    upload_time = Column(DateTime, default=datetime.utcnow)
    width = Column(Integer)
    height = Column(Integer)
    coin_count = Column(Integer, default=0)


class CoinModel(Base):
    """
    Store detected coin information.
    Why: Persist detection results for retrieval and evaluation.
    """

    __tablename__ = "coins"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    image_id = Column(String, nullable=False, index=True)
    # Bounding box stored as separate fields for easy querying
    bbox_x = Column(Integer, nullable=False)
    bbox_y = Column(Integer, nullable=False)
    bbox_width = Column(Integer, nullable=False)
    bbox_height = Column(Integer, nullable=False)
    # Circle parameters
    centroid_x = Column(Integer, nullable=False)
    centroid_y = Column(Integer, nullable=False)
    radius = Column(Integer, nullable=False)
    confidence = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


# Create database engine
# check_same_thread=False needed for SQLite with FastAPI
engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """Dependency injection for database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all database tables."""
    Base.metadata.create_all(bind=engine)
