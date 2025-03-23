from sqlalchemy.orm import Mapped, mapped_column
from app import db

from models.champs import Champs
from models.documents import Documents


class Controles(db.Model):
    id: Mapped[int] = mapped_column(primary_key=True)
    resultat : Mapped[bool]

    document_id = db.Column(db.ForeignKey(Documents.id))  # primary_key=True

    champs_id = db.Column(db.ForeignKey(Champs.id))