from datetime import datetime

from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import db, Column, ForeignKey

from models.typelivrables import Typelivrables


class Documents(db.Model):
    id: Mapped[int] = mapped_column(primary_key=True)

    date: Mapped[datetime]

    Column("idtypelivrable", ForeignKey(Typelivrables.id))
