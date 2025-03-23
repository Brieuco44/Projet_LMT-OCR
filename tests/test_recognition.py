import pytest

from services.recognition_service import Recognition_service

path = "../data/"

ca_signe_csp = path + "ca_signe_csp/"

def cidentifiant_benificiaire_Ok():
    # pdf "22 CA Signé CSP"
    pdf = ca_signe_csp + "20_CA_Signé_CSP.pdf"
    rgntn_serv = Recognition_service(
        pdf,
        6
    )
    # 82979546
    print(rgntn_serv)