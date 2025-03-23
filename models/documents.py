from datetime import datetime

from sqlalchemy.orm import Mapped, mapped_column
from app import db

from models.typelivrables import Typelivrables


class Documents(db.Model):
    id: Mapped[int] = mapped_column(primary_key=True)

    date: Mapped[datetime]

    db.Column("idtypelivrable", db.ForeignKey(Typelivrables.id))
