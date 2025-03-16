import streamlit as st
import pymysql
import tempfile

# Load MySQL credentials from Streamlit secrets
MYSQL_CONFIG = st.secrets["MYSQL"]

# Store CA certificate temporarily for secure MySQL connection
with tempfile.NamedTemporaryFile(delete=False, suffix=".pem") as temp_cert:
    temp_cert.write(MYSQL_CONFIG["CA_CERT"].encode("utf-8"))
    temp_cert_path = temp_cert.name  # This is the correct variable name

# Function to connect to MySQL
def get_db_connection():
    return pymysql.connect(
        host=MYSQL_CONFIG["HOST"],
        user=MYSQL_CONFIG["USER"],
        password=MYSQL_CONFIG["PASSWORD"],
        database=MYSQL_CONFIG["DATABASE"],
        ssl={"ca": temp_cert_path}  # Corrected CA cert file path
    )
