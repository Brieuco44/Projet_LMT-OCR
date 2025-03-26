import pytest
from services.recognition_service import Recognition_service
from app import app, db  # Import the app object as well

path = "../data/"

ca_signe_csp = path + "ca_signe_csp/"

def test_ca_signe_CSP_20_new():
    # pdf "22 CA Signé CSP"
    pdf = ca_signe_csp + "20_CA_Signé_CSP.pdf"
    # Ensure you're in the app context
    with (app.app_context()):
        rgntn_serv = Recognition_service(
            pdf,
            6,
            db
        )
        res = rgntn_serv.process(True)
        rgntn_serv.draw_boxes_on_pdf()

    assert res["Beneficiare"]["Identifiant N Beneficiaire"] == "60387042"
    assert res["Consultant"]["Nom/prenom Consultant"] == "RIZZI Amélie"
    assert res["Organisme Prestataire"]["Adresse mail Organisme"] == "clisson@catalys-conseil.fr"
    assert res["Information"]["Date adhesion CSP"] == "08/03/2024"
    assert res["Information"]["Date Demarrer Accompagnement"] == "19/04/2024"
    assert res["Information"]["Date Fin Accompagnement"] == "07/03/2025"

def test_ca_signe_CSP_21_PasOk():
    # pdf "22 CA Signé CSP"
    pdf = ca_signe_csp + "21_CA_Signé_CSP.pdf"
    # Ensure you're in the app context
    with (app.app_context()):
        rgntn_serv = Recognition_service(
            pdf,
            6,
            db
        )
        res = rgntn_serv.process(True)
        print(res)
        rgntn_serv.draw_boxes_on_pdf()

    assert res["Beneficiare"]["Identifiant N Beneficiaire"] == "66816037"
    assert res["Consultant"]["Nom/prenom Consultant"] == "GUILLARD Elodie"
    assert res["Organisme Prestataire"]["Adresse mail Organisme"] == "trignac@catalys-conseil.fr"
    assert res["Information"]["Date adhesion CSP"] ==  "09/03/2024"
    assert res["Information"]["Date Demarrer Accompagnement"] ==  "17/05/2024"
    assert res["Information"]["Date Fin Accompagnement"] == "07/03/2025"

def test_ca_signe_CSP_22_Ok():
    pdf = ca_signe_csp + "22_CA_Signé_CSP.pdf"
    with (app.app_context()):
        rgntn_serv = Recognition_service(
            pdf,
            7,
            db
        )
        res = rgntn_serv.process(True)

    assert res["Beneficiare"]["Identifiant N Beneficiaire"] == "82979546"
    assert res["Referent"]["Nom/prenom Referent"] == "PAPIN Elise"
    assert res["Organisme Prestataire"]["Adresse mail Organisme"] == "Irsy@catalys-conseil.fr"
    assert res["Information"]["Date adhesion CSP"] == "07/08/2024"
    assert res["Information"]["Date Demarrer Accompagnement"] == "01/10/2024"
    assert res["Information"]["Date Fin Accompagnement"] ==  "06/08/2025"


def test_ca_signe_CSP_22_Ok():
    pdf = ca_signe_csp + "2_CA_Signé_CSP.pdf"
    with (app.app_context()):
        rgntn_serv = Recognition_service(
            pdf,
            7,
            db
        )
        res = rgntn_serv.process(True)

    assert res["Beneficiare"]["Identifiant N Beneficiaire"] == "1747403B"
    assert res["Referent"]["Nom/prenom Referent"] == "COLAS Pauline"
    assert res["Referent"]["Adresse mail Referent"] == "pauline.colas@catalys-conseil.fr"
    assert res["Information"]["Date adhesion CSP"] == "20/08/2024"
    assert res["Information"]["Date Demarrer Accompagnement"] == "23/10/2024"
    assert res["Information"]["Date Fin Accompagnement"] == "19/08/2025"

