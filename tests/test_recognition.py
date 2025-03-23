#import pytest
from services.recognition_service import Recognition_service
from app import app, db  # Import the app object as well

path = "../data/"

ca_signe_csp = path + "ca_signe_csp/"

def cidentifiant_benificiaire_Ok():
    # pdf "22 CA Signé CSP"
    pdf = ca_signe_csp + "20_CA_Signé_CSP.pdf"
    # Ensure you're in the app context
    with (app.app_context()):
        rgntn_serv = Recognition_service(
            pdf,
            6,
            db
        )
        #print(rgntn_serv.process(True))
        rgntn_serv.draw_boxes_on_pdf()
    # This will now run inside the application context


cidentifiant_benificiaire_Ok()