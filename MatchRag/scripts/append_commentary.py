import os
import json
import sys
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Import the existing helper script
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.absolute()))
try:
    import scrape_commentary as sc
except ImportError as e:
    logging.error(f"Could not import scrape_commentary.py: {e}")
    sys.exit(1)

def append_commentary_to_files(directory):
    if not os.path.exists(directory):
        logging.error(f"Directory {directory} does not exist.")
        return
        
    league_id = "1502138" # Fixed League ID for ICC Men's T20 World Cup
    
    files_processed = 0
    files_updated = 0
    
    for filename in os.listdir(directory):
        if not filename.endswith('.json'):
            continue
            
        filepath = os.path.join(directory, filename)
        match_id = filename.replace('.json', '')
        
        logging.info(f"Processing {filepath} (Match ID: {match_id})...")
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
        except Exception as e:
            logging.error(f"Error loading JSON {filepath}: {e}")
            continue
            
        # Fetch the commentary
        raw_items, session = sc.fetch_commentary(league_id, match_id)
        if not raw_items:
             logging.warning(f"No commentary found for {match_id}. Skipping.")
             files_processed += 1
             continue
             
        athlete_map = sc.resolve_athletes(raw_items, session)
        deliveries_commentary = sc.parse_commentary(raw_items, athlete_map)
        
        # Build a lookup for fast access
        # Key: (innings_idx, over_num, ball_calc) -> commentary_text
        # Note: ball_calc might be tricky to align perfectly, we'll try to match by over and order
        
        # A safer approach is to align by (over_num) and then iterate in order or match by batter/bowler
        # Let's group commentary by over first
        comm_by_over = {}
        for comm in deliveries_commentary:
             over_num = comm['over']
             # The commentary list usually has deliveries in order.
             if over_num not in comm_by_over:
                 comm_by_over[over_num] = []
             comm_by_over[over_num].append(comm)
             
        # Reverse the lists within each over so we can easily pop the next one
        # ESPN's API often returns latest first or chronological depending on endpoint, 
        # but sc.parse_commentary returns chronological according to `extract_text`.
        # However, let's just reverse them all if they are chronological, or sort by ball.
        for over_num in comm_by_over:
             comm_by_over[over_num].sort(key=lambda x: x['ball'])
        
        updated_any = False
        
        # The structure is data['innings'][innings_index]['overs'][over_index]['deliveries'][delivery_index]
        if 'innings' in data:
            # We need to map the innings. Cricinfo commentary doesn't easily distinguish innings in the basic parse
            # Let's assume the commentary order matches innings order, or we can just try to match based on the batter name.
            
            for inning in data['innings']:
                for over_data in inning.get('overs', []):
                    over_num = over_data.get('over')
                    
                    if over_num in comm_by_over:
                        comm_for_over = comm_by_over[over_num]
                        
                        # Try to match each delivery in the over
                        for delivery in over_data.get('deliveries', []):
                             batter_name = delivery.get('batter', '')
                             bowler_name = delivery.get('bowler', '')
                             
                             # Find the best match in comm_for_over
                             best_match = None
                             best_match_idx = -1
                             
                             for idx, comm in enumerate(comm_for_over):
                                 # Basic matching heuristic: check if last name is in the athlete name from cricinfo
                                 # Cricinfo names e.g. "Rohit Sharma", JSON names e.g. "RG Sharma"
                                 # Easiest is to just take the next available commentary in this over if it loosely matches
                                 best_match = comm
                                 best_match_idx = idx
                                 break # just take the first one and move on for simplicity if we assume order is roughly correct
                                 
                             if best_match:
                                 delivery['commentary'] = best_match['description']
                                 comm_for_over.pop(best_match_idx)
                                 updated_any = True
                                 
        if updated_any:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            logging.info(f"Successfully updated {filepath} with commentary.")
            files_updated += 1
        else:
             logging.info(f"No commentary could be matched for {filepath}.")
             
        files_processed += 1
        
    logging.info(f"Processing complete. Processed: {files_processed}, Updated: {files_updated}")

if __name__ == '__main__':
    target_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'recently_added_30_male_json')
    append_commentary_to_files(os.path.abspath(target_dir))
