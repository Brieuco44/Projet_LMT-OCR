from dataclasses import dataclass
from typing import Optional
from datetime import date

@dataclass
class DossierPrestation:
    iddossier: int
    prestation: str
    numanpe: str
    numlettrecde: str
    nom_benef: str
    prenom_benef: str
    cp_benef: str
    datecreation: date
    datedebutcde: date
    datedebut: date
    seance_datedebut: Optional[date]
    datefin: date
    datefincde: date
    dossieretat: str
    convention: str
    type_population: str
    conseiller: str
    instinom: str
    ville_instinom: str
    insticp: str
    departement_instinom: str
    region_instinom: str
    mois_debut: int
    mois_fin: int
    dureeprevi: int
    dureereel: int
    dureeprodprevi: int
    dureeprodreel: int
    dureetotaleprevi: int
    dureetotalereel: int
    duree_jour: int
    sortie: str
    type_sortie: str
    statut_primo: str
    statut_adh: str
    statut_fin: str
    abandon: bool
    fin_presta: bool
    en_retard: bool
    suspension_LE: bool
    motif_suspension_LE: Optional[str]
    psp_LE: Optional[str]
    point_etape_LE: Optional[str]
    modalite_contact: str
    prestasousfamille: str
    prestafamille: str
    commentaire: Optional[str]
    typecontact: str
    gestionnaire: str
    financeur: str
    nb_heures_ateliers: int
    nb_ateliers: int
