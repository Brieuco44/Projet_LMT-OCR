from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import db, Column, ForeignKey

from models.champs import Champs
from models.documents import Documents


class Controles(db.Model):
    id: Mapped[int] = mapped_column(primary_key=True)
    resultat : Mapped[bool]

    Column("iddocument", ForeignKey(Documents.id))  # primary_key=True

    Column("idchamps", ForeignKey(Champs.id))