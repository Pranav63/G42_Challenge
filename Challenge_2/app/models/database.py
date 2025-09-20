from sqlalchemy import (
    create_engine,
    Column,
    Float,
    Integer,
    LargeBinary,
    String,
    DateTime,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import numpy as np
from datetime import datetime

from app.core.config import settings

Base = declarative_base()


class ImageFrame(Base):
    __tablename__ = "image_frames"

    id = Column(Integer, primary_key=True, autoincrement=True)
    depth = Column(Float, nullable=False, index=True, unique=True)
    image_data = Column(LargeBinary, nullable=False)
    width = Column(Integer, default=150)
    height = Column(Integer, default=1)
    original_width = Column(Integer, default=200)
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_array(self) -> np.ndarray:
        return np.frombuffer(self.image_data, dtype=np.uint8).reshape(1, self.width)

    @classmethod
    def from_array(cls, depth: float, array: np.ndarray, original_width: int = 200):
        return cls(
            depth=depth,
            image_data=array.astype(np.uint8).tobytes(),
            width=array.shape[1] if len(array.shape) > 1 else len(array),
            height=1,
            original_width=original_width,
        )


engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    Base.metadata.create_all(bind=engine)
