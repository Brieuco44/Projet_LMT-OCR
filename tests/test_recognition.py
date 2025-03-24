import pytest
from services.recognition_service import Recognition_service
from app import app, db  # Import the app object as well

path = "../data/"

ca_signe_csp = path + "ca_signe_csp/"

def test_ca_signe_CSP_20_Ok():
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
        rgntn_serv.draw_boxes_on_pdf(output_pdf_path="first.pdf")

    assert res["Identifiant N Beneficiaire"] == "60387042"
    assert res["Nom/prenom Consultant"] == "RIZZI Amélie"
    assert res["Adresse mail Organisme"] == "clisson@catalys-conseil.fr"

def test_ca_signe_CSP_21_Ok():
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
        rgntn_serv.draw_boxes_on_pdf()

    assert res["Identifiant N Beneficiaire"] == "66816037"
    # assert res["Nom/prenom Consultant"] == "GUILLARD Elodie"
    # assert res["Adresse mail Organisme"] == "elodie.guillard@catalys-conseil.fr"

def test_ca_signe_CSP_22_Ok():
    pdf = ca_signe_csp + "22_CA_Signé_CSP.pdf"
    with (app.app_context()):
        rgntn_serv = Recognition_service(
            pdf,
            7,
            db
        )
        res = rgntn_serv.process(True)

    assert res["Identifiant N Beneficiaire"] == "82979546"
    assert res["Nom/prenom Referent"] == "PAPIN Elise"
    assert res["Telephone Portable Referent"] == "+33679097204"
    assert res["Adresse mail Referent"] == "elise.papin@catalys-conseil.fr"
    assert res["Date adhesion CSP"] == "07/08/2024"
    assert res["Date Demarrer Accompagnement"] == "01/10/2024"
    assert res["Date Fin Accompagnement"] == "06/08/2025"


def test_ca_signe_CSP_22_Ok():
    pdf = ca_signe_csp + "2_CA_Signé_CSP.pdf"
    with (app.app_context()):
        rgntn_serv = Recognition_service(
            pdf,
            7,
            db
        )
        res = rgntn_serv.process(True)

    assert res["Identifiant N Beneficiaire"] == "1747403B"
    assert res["Nom/prenom Referent"] == "COLAS Pauline"
    assert res["Telephone Portable Referent"] == "+33627064948"
    assert res["Adresse mail Referent"] == "pauline.colas@catalys-conseil.fr"
    assert res["Date adhesion CSP"] == "20/08/2024"
    assert res["Date Demarrer Accompagnement"] == "23/10/2024"
    assert res["Date Fin Accompagnement"] == "19/08/2025"

