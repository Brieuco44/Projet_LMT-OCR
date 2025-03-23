import pytest
from services.recognition_service import Recognition_service
from app import app, db  # Import the app object as well

path = "../data/"

ca_signe_csp = path + "ca_signe_csp/"

def test_ca_signe_CSP_Ok():
    # pdf "22 CA Signé CSP"
    pdf = ca_signe_csp + "20_CA_Signé_CSP.pdf"
    # Ensure you're in the app context
    with (app.app_context()):
        rgntn_serv = Recognition_service(
            pdf,
            6,
            db
        )
        res = rgntn_serv.process()
    assert res["Identifiant N Beneficiaire"] == "60387042"
    assert res["Nom/prenom Consultant"] == "RIZZI Amélie"
    assert res["Adresse mail Organisme"] == "clisson@catalys-conseil.fr"
    # This will now run inside the application context

