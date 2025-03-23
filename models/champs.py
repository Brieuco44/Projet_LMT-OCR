from typing import TypedDict
from sqlalchemy.orm import Mapped, mapped_column
from app import db


from models.typechamps import Typechamps
from models.typelivrables import Typelivrables


class ZoneDict(TypedDict):
    x1: int
    y1: int
    x2: int
    y2: int


class Champs(db.Model):
    id: Mapped[int] = mapped_column(primary_key=True)

    nom: Mapped[str]
    zone: Mapped[ZoneDict] = mapped_column(db.JSON, nullable=False) # Coordonnées de type ZoneDict
    page: Mapped[int] # Numéro de page

    db.Column("idtypechamps", db.ForeignKey(Typechamps.id)) #primary_key=True

    db.Column("idtypelivrable", db.ForeignKey(Typelivrables.id))