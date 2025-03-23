from typing import TypedDict
from sqlalchemy.orm import Mapped, mapped_column
from app import db
from sqlalchemy import JSON


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
    zone: Mapped[ZoneDict] = mapped_column(JSON, nullable=False) # {'x1': 1356, 'x2': 2330, 'y1': 2860, 'y2': 3048}
    page: Mapped[int] # Numéro de page

    type_champs_id = db.Column(db.ForeignKey(Typechamps.id)) #primary_key=True

    type_livrable_id = db.Column(db.ForeignKey(Typelivrables.id))