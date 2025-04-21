from sqlalchemy.orm import Mapped, mapped_column
from app import db

from models.typechamps import Typechamps
from models.zone import Zone


class Champs(db.Model):
    id: Mapped[int] = mapped_column(primary_key=True)

    nom: Mapped[str]

    zoneid = db.Column(db.ForeignKey(Zone.id))

    question: Mapped[str]  # Numéro de page

    type_champs_id = db.Column(db.ForeignKey(Typechamps.id)) #primary_key=True
