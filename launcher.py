# launcher.py
import sys
from streamlit.web import cli as stcli

if __name__ == "__main__":
    sys.argv = ["streamlit", "run", "code/albert_query_app.py", "--server.port=8501"]
    sys.exit(stcli.main())
    