import streamlit as st
import pymysql
import tempfile
import os

# Load MySQL credentials from Streamlit secrets
MYSQL_HOST = st.secrets["MYSQL_HOST"]
MYSQL_USER = st.secrets["MYSQL_USER"]
MYSQL_PASSWORD = st.secrets["MYSQL_PASSWORD"]
MYSQL_CA_CERT = st.secrets["MYSQL_CA_CERT"]  # SSL certificate content from secrets

# Function to establish a secure MySQL connection
def get_db_connection(database=None):
    if not MYSQL_CA_CERT:
        raise ValueError("MYSQL_CA_CERT is missing in Streamlit secrets.")
    
    # Create a temporary file for the SSL certificate for each connection
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pem") as temp_cert:
        temp_cert.write(MYSQL_CA_CERT.encode("utf-8"))
        temp_cert.flush()
        ca_cert_path = temp_cert.name
        
        try:
            conn = pymysql.connect(
                host=MYSQL_HOST,
                user=MYSQL_USER,
                password=MYSQL_PASSWORD,
                database=database if database else None,  # Dynamic database support
                ssl={"ca": ca_cert_path},
                port=4000  # Default TiDB Serverless port
            )
            return conn
        except pymysql.MySQLError as e:
            st.error(f"Database Connection Error: {e}")
            return None
        finally:
            # Clean up the temporary file after connection attempt
            if os.path.exists(ca_cert_path):
                os.unlink(ca_cert_path)