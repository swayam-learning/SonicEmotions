import streamlit as st
import pymysql
import tempfile

# Load database credentials from Streamlit Secrets
MYSQL_HOST = st.secrets["MYSQL_HOST"]
MYSQL_USER = st.secrets["MYSQL_USER"]
MYSQL_PASSWORD = st.secrets["MYSQL_PASSWORD"]
MYSQL_DATABASE = st.secrets["MYSQL_DATABASE"]
MYSQL_CA_CERT = st.secrets["MYSQL_CA_CERT"]  # CA certificate content from secrets

# Store CA Cert as a temporary file
with tempfile.NamedTemporaryFile(delete=False, suffix=".pem") as temp_cert:
    temp_cert.write(MYSQL_CA_CERT.encode("utf-8"))
    temp_cert_path = temp_cert.name  # Get the path of the temporary certificate file

# Establish MySQL connection with SSL
def get_db_connection():
    return pymysql.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
        ssl_ca=temp_cert_path  # Correct way to use SSL in pymysql
    )
