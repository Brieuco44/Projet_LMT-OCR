from sqlalchemy.orm import Mapped, mapped_column
from app import db

class Typechamps(db.Model):
    id: Mapped[int] = mapped_column(primary_key=True)
    nom: Mapped[str]
