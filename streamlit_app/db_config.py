import streamlit as st
import pymysql
import tempfile

# Load MySQL credentials from Streamlit secrets
MYSQL_HOST = st.secrets["MYSQL_HOST"]
MYSQL_USER = st.secrets["MYSQL_USER"]
MYSQL_PASSWORD = st.secrets["MYSQL_PASSWORD"]
MYSQL_DATABASE = st.secrets["MYSQL_DATABASE"]
MYSQL_CA_CERT = st.secrets["MYSQL_CA_CERT"]  # Get certificate from secrets

# Create a temporary file to store the certificate
with tempfile.NamedTemporaryFile(delete=False, suffix=".pem") as temp_cert:
    temp_cert.write(MYSQL_CA_CERT.encode("utf-8"))
    temp_cert_path = temp_cert.name  # Get the file path

# Function to establish a secure MySQL connection using SSL
def get_db_connection():
    return pymysql.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
        ssl={"ca": temp_cert_path}  # Use the temporary certificate file
    )
