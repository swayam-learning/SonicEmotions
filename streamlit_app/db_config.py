import streamlit as st
import pymysql
import tempfile

# Load MySQL credentials from Streamlit secrets
MYSQL_HOST = st.secrets["MYSQL_HOST"]
MYSQL_USER = st.secrets["MYSQL_USER"]
MYSQL_PASSWORD = st.secrets["MYSQL_PASSWORD"]
MYSQL_DATABASE = st.secrets["MYSQL_DATABASE"]
MYSQL_CA_CERT = st.secrets["MYSQL_CA_CERT"]  # Get the SSL certificate from secrets

# Check if MYSQL_CA_CERT is correctly loaded
if not MYSQL_CA_CERT:
    raise ValueError("MYSQL_CA_CERT is missing in Streamlit secrets.")

# Create a temporary file to store the SSL certificate
with tempfile.NamedTemporaryFile(delete=False, suffix=".pem") as temp_cert:
    temp_cert.write(MYSQL_CA_CERT.encode("utf-8"))
    temp_cert.flush()  # Ensure data is written before using the file
    MYSQL_CA_CERT_PATH = temp_cert.name  # Store the path of the temporary cert

# Function to establish a secure MySQL connection
def get_db_connection():
    return pymysql.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
        ssl={"ca": MYSQL_CA_CERT_PATH}  # Use the correct variable name
    )
