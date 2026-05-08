#Questo script serve a generare il datatset completo da dare a watson
#permette anche di mescolare i file jsonl

import random

#Nomi dei file JSONL
file_da_unire = [
    'tracciato_completo.json',
    'data_recupero_sbandata.json',
    'data_recupero_testacoda.json',
    'data_recupero_erba.json'
]

def crea_dataset_finale(lista_file, file_uscita):
    dataset_totale = []
    
    for nome_file in lista_file:
        try:
            with open(nome_file, 'r', encoding='utf-8') as f:
                righe = f.readlines()
                dataset_totale.extend(righe)
                print(f"Caricato {nome_file}: {len(righe)} righe")
        except FileNotFoundError:
            print(f"Attenzione: file {nome_file} non trovato. Salto...")

    #Shuffle (mescolamento) dei dati
    random.shuffle(dataset_totale)

    with open(file_uscita, 'w', encoding='utf-8') as f_out:
        for riga in dataset_totale:
            f_out.write(riga.strip() + "\n")
            
    print(f"\nDataset finale '{file_uscita}' creato con {len(dataset_totale)} righe mescolate.")

# Eseguo il merge
file_output=input("Dai un nome al file che unirà altri file JSON: ")
crea_dataset_finale(file_da_unire, file_output)