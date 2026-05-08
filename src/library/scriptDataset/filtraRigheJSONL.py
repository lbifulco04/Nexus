#Questo script recupera una riga su 5 di ogni file JSONL

def filter_jsonl(input_file, output_file, step=5):
    """
    Legge un file JSONL e ne salva una riga ogni 'step'.
    """
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        count = 0
        saved_count = 0
        
        for line in infile:
            # Salva la riga solo se il resto della divisione per step è 0
            if count % step == 0:
                outfile.write(line)
                saved_count += 1
            count += 1
            
    print(f"Filtraggio completato!")
    print(f"Righe totali analizzate: {count}")
    print(f"Righe salvate nel nuovo file: {saved_count}")

# Utilizzo dello script
input_path = input("Inserisci nome del file che vuoi ridurre: ")
step = input("\nInserisci il numero di righe da saltare: ")  
output_path = input("\nInserisci il nome del file che vuoi salvare :")

filter_jsonl(input_path, output_path, step)