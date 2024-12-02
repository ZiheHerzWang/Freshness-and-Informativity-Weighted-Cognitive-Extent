import csv

def extract_year_title(bib_file, csv_file):
    with open(bib_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    csv_data = []
    current_entry = {}
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("@"): 
            if current_entry:  
                if 'title' in current_entry and 'year' in current_entry:
                    csv_data.append([current_entry['year'], current_entry['title'], current_entry['url']])
            current_entry = {}  
        elif '=' in stripped:  
            key, value = stripped.split('=', 1)
            key = key.strip().lower()
            value = value.strip().strip('{},').replace('"', '').replace("'", "")
            current_entry[key] = value

    if current_entry and 'title' in current_entry and 'year' in current_entry:
        csv_data.append([current_entry['year'], current_entry['title'], current_entry['url']])

    with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["year", "title","url"])
        writer.writerows(csv_data)

    print(f"Extracted {len(csv_data)} entries from {bib_file} to {csv_file}")


bib_file = "new_pipeline/test.bib" 
csv_file = "new_pipeline/references.csv" 
extract_year_title(bib_file, csv_file)
