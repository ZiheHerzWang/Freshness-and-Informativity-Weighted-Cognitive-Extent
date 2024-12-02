import requests
import time
import csv
import pandas as pd


api_key = "your own ACL api key"  

def get_paper_id_from_url(url):
    parts = url.rsplit('/', 1)
    return parts[-1]

def get_citations(acl_id, offset):
    response = requests.get(
        f"https://api.semanticscholar.org/graph/v1/paper/ACL:{acl_id}/citations?limit=100&offset={offset}",
        headers={"x-api-key": api_key}
    )
    if response.status_code == 200:
        data = response.json()
        if 'message' in data:
            time.sleep(2)
            return get_citations(acl_id, offset)
        else:
            return data.get("data", [])
    else:
        print(f"Error retrieving citations for ACL:{acl_id}, ", response.content)
        return None

def get_citing_papers(acl_id):
    citing_papers = []
    offset = 0
    initial_response = requests.get(
        f"https://api.semanticscholar.org/graph/v1/paper/ACL:{acl_id}?fields=citationCount",
        headers={"x-api-key": api_key}
    )
    if initial_response.status_code == 200:
        data = initial_response.json()
        if 'message' in data:
            time.sleep(2)
            return get_citing_papers(acl_id)
        citation_count = data.get("citationCount", 0)
        print(citation_count)

        while True:
            citations = get_citations(acl_id, offset)
            if not citations:
                break
    
            for citation in citations:
                paper_info = citation.get("citingPaper")
                paper_id = paper_info.get("paperId", "")
                print(paper_id)
                if paper_info and paper_id:
                    citing_papers.append(paper_id)
            offset += len(citations)

    else:
        print(f"Error retrieving citation count for ACL:{acl_id}, ", initial_response.content)

    return citing_papers

def get_paper_publication_year(paper_id):
    response = requests.get(
        f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}?fields=year",
        headers={"x-api-key": api_key}
    )
    data = response.json()
    if 'message' in data:
        time.sleep(2)
        return get_paper_publication_year(paper_id)
    else:
        print(data.get("year"))
        return data.get("year")

def process_papers(csv_file_path, start_year, end_year):
    
    papers_df = pd.read_csv(csv_file_path)
    results = []
    paper_track_num =0
    for _, row in papers_df.iterrows():
        acl_id = get_paper_id_from_url(row['url'])
        print(acl_id)
        paper_track_num += 1
        print("which paper: " + str(paper_track_num))
        all_citing_papers = get_citing_papers(acl_id)
        

        citation_count = 0
        for citing_paper_id in all_citing_papers:
            publication_year = get_paper_publication_year(citing_paper_id)
            if publication_year and start_year <= publication_year <= end_year:
                citation_count += 1

        results.append({
            'title': row['title'],
            'url': row['url'],
            'year': row['year'],
            'citations': citation_count
        })

    return results
def main():
    csv_file_path = rf"FICE_prove_2015_paper/filtered_2015.csv"  
    start_year = 2015
    end_year = 2020
    
    results = process_papers(csv_file_path, start_year, end_year)
    with open('FICE_prove_2015_paper/references_with_citations.csv', 'a', newline='', encoding='utf-8') as csvfile:  
        fieldnames = ['Title', 'url', 'Year', 'citations']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow(result)

if __name__ == "__main__":
    main()
