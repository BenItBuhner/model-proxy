from sqlalchemy import Column, Integer, String, JSON
from .database import Base


class Log(Base):
    __tablename__ = "logs"

    id = Column(Integer, primary_key=True, index=True)
    service = Column(String, index=True)
    request = Column(JSON)
    response = Column(JSON)
