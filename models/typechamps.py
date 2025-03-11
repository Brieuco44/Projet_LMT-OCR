from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import db

class Typechamps(db.Model):
    id: Mapped[int] = mapped_column(primary_key=True)
    nom: Mapped[str]
