from ibm_watsonx_ai import APIClient

# --- INSERISCI I TUOI DATI QUI ---
API_KEY = input("inserisci chiave API server IBM Cloud: ")
PROJECT_ID = input("inserisci ID del progetto: ")
#'https://us-south.ml.cloud.ibm.com' per Dallas 
REGION_URL = "https://us-south.ml.cloud.ibm.com" 

credentials = {
    "url": REGION_URL,
    "apikey": API_KEY
}

client = APIClient(credentials)

try:
    # Imposta il progetto di default
    client.set.default_project(PROJECT_ID)
    print("✅ CONNESSIONE RIUSCITA!")
    
    # Recupera i dettagli del progetto (Sintassi corretta per versione 1.5.x)
    project_details = client.projects.get_details(PROJECT_ID)
    nome_progetto = project_details.get('entity', {}).get('name', 'N/A')
    print(f"Progetto attivo: {nome_progetto}")

except Exception as e:
    print(f"❌ ERRORE DI CONNESSIONE: {e}")