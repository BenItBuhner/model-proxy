from sqlalchemy.orm import Session
from . import models

def create_log(db: Session, service: str, request: dict, response: dict):
    db_log = models.Log(service=service, request=request, response=response)
    db.add(db_log)
    db.commit()
    db.refresh(db_log)
    return db_log
