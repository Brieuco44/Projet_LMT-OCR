from sqlalchemy.orm import Mapped, mapped_column
from app import db

from models.champs import Champs
from models.documents import Documents


class Controles(db.Model):
    id: Mapped[int] = mapped_column(primary_key=True)
    resultat : Mapped[bool]

    db.Column("iddocument", db.ForeignKey(Documents.id))  # primary_key=True

    db.Column("idchamps", db.ForeignKey(Champs.id))