import requests
import json
import re
import sys
import os
from concurrent.futures import ThreadPoolExecutor

def extract_match_id(url):
    """
    Extract the match ID automatically from the URL.
    Returns: (league_id, match_id)
    """
    match_id = None
    league_id = None
    
    # Try capturing league id from URL like "series/icc-men-s-t20...-1502138/"
    league_match = re.search(r"series/[a-zA-Z0-9\-]+-(\d+)/", url)
    if league_match:
        league_id = league_match.group(1)
        
    match_match = re.search(r"-(\d+)/(?:ball-by-ball-commentary|live-cricket-score|full-scoreboard)", url)
    if match_match:
        match_id = match_match.group(1)
    else:
        # Fallback to finding the last sequence of digits followed by / or end
        match_match = re.search(r"/(\d+)(?:/|$)", url)
        if match_match:
            match_id = match_match.group(1)
            
    return league_id, match_id

def determine_event(text, runs):
    """
    Event detection rules:
    - if text contains "SIX" -> event = "six"
    - if text contains "FOUR" -> event = "four"
    - if text contains "OUT" -> event = "wicket"
    - if runs == 0 -> event = "dot"
    - otherwise -> event = "run"
    """
    text_upper = text.upper()
    if "SIX" in text_upper:
        return "six"
    elif "FOUR" in text_upper:
        return "four"
    elif "OUT" in text_upper or "WICKET" in text_upper:
        return "wicket"
    elif runs == 0:
        return "dot"
    else:
        return "run"

def group_by_over(commentary_list):
    """
    Helper function that groups deliveries by over.
    This will be used later for RAG chunking.
    """
    grouped = {}
    for item in commentary_list:
        over_num = item.get("over", 0)
        if over_num not in grouped:
            grouped[over_num] = []
        grouped[over_num].append(item)
        
    result = []
    # Sort them so they are ordered by over number
    for over_num in sorted(grouped.keys(), reverse=True):
        result.append({
            "over": over_num,
            "deliveries": grouped[over_num]
        })
    return result

def fetch_commentary(league_id, match_id):
    """
    Use the ESPN core mobile API to bypass the Akamai firewall.
    API: http://core.espnuk.org/v2/sports/cricket/leagues/{LEAGUE_ID}/events/{MATCH_ID}/competitions/{MATCH_ID}/plays
    """
    print(f"Fetching commentary for match {match_id} (league {league_id})...")
    
    if not league_id:
        print("League ID is missing. The mobile API requires a League ID. Try providing the full ESPN Cricinfo URL.")
        return []

    # First fetch the pointers to all plays
    base_url = f"http://core.espnuk.org/v2/sports/cricket/leagues/{league_id}/events/{match_id}/competitions/{match_id}/plays"
    
    headers = {
        "User-Agent": "ESPN/6.12.1 (iPhone; iOS 15.0; Scale/3.00)",
        "Accept": "application/json"
    }
    
    session = requests.Session()
    session.headers.update(headers)
    
    items = []
    try:
        # Fetching all plays pointers in one pagination limits (usually max 1000)
        resp = session.get(base_url, params={"limit": 1000}, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            
            # This API returns a list of references ($ref) instead of the actual data
            refs = data.get("items", [])
            print(f"Found {len(refs)} ball-by-ball references. Fetching details...")
            
            # Fetch the actual ball data from the references concurrently
            def fetch_ref(ref_obj):
                try:
                    url = ref_obj["$ref"]
                    # Do not force https as core.espnuk.org doesn't support it well on this endpoint
                    r = session.get(url, timeout=10)
                    if r.status_code == 200:
                        return r.json()
                except Exception:
                    pass
                return None
                
            with ThreadPoolExecutor(max_workers=20) as executor:
                results = list(executor.map(fetch_ref, refs))
                
            items = [res for res in results if res is not None]
            print(f"Successfully retrieved details for {len(items)} deliveries.")
        else:
            print(f"API request ended or failed with status code {resp.status_code}.")
            
    except requests.RequestException as e:
        print(f"Error fetching data: {e}")
            
    return items, session

def resolve_athletes(items, session):
    """
    Find all unique athlete IDs and fetch their names to replace 'Unknown (ID: ...)'
    """
    athlete_ids = set()
    for item in items:
        # Collect IDs from batsman and bowler refs
        for role in ["batsman", "bowler", "otherBatsman", "otherBowler"]:
            ref = item.get(role, {}).get("athlete", {}).get("$ref", "")
            if ref:
                athlete_ids.add(ref.replace("http://", "https://"))
                
    print(f"Resolving {len(athlete_ids)} unique athletes for clean commentary...")
    
    athlete_map = {}
    
    def fetch_athlete(url):
        try:
            r = session.get(url, timeout=10)
            if r.status_code == 200:
                data = r.json()
                name = data.get("fullName", data.get("displayName", "Unknown"))
                return url, name
        except Exception:
            pass
        return url, "Unknown"
        
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = executor.map(fetch_athlete, list(athlete_ids))
        for url, name in results:
            athlete_map[url] = name
            # Also map the http version just in case
            athlete_map[url.replace("https://", "http://")] = name
            
    return athlete_map

def extract_text(item):
    """Helper to try to find the actual commentary text string in the item"""
    if "text" in item and isinstance(item["text"], str): return item["text"]
    if "shortText" in item and isinstance(item["shortText"], str): return item["shortText"]
    if "commentText" in item:
        ctx = item["commentText"]
        if isinstance(ctx, dict):
            return ctx.get("plainText") or ctx.get("html") or ""
        elif isinstance(ctx, str):
            return ctx
    if "title" in item and isinstance(item["title"], str): return item["title"]
    return ""

def parse_commentary(items, athlete_map):
    """
    Convert commentary into structured JSON handling missing data gracefully.
    """
    parsed_deliveries = []
    
    for item in items:
        # The mobile API stores descriptions in text or shortText
        text = extract_text(item)
        
        bowler_ref = item.get("bowler", {}).get("athlete", {}).get("$ref", "")
        batter_ref = item.get("batsman", {}).get("athlete", {}).get("$ref", "")
        
        bowler_name = athlete_map.get(bowler_ref, "Unknown (ID: {})".format(bowler_ref.split("/")[-1] if bowler_ref else "?"))
        batter_name = athlete_map.get(batter_ref, "Unknown (ID: {})".format(batter_ref.split("/")[-1] if batter_ref else "?"))
        
        if not text:
            # Fallback for mobile API format where batsman/bowler are provided but no description
            if bowler_ref and batter_ref:
                text = f"{bowler_name} to {batter_name}"
            else:
                continue
            
        bowler = bowler_name
        batter = batter_name
        runs = item.get("scoreValue", 0)
        
        if not runs and runs != 0:
            runs = item.get("runs", 0)
                
        over_val = 0
        ball_val = 0
        
        # In the mobile API, events sometimes are "Player to Player, FOUR"
        # We only want to use parsing from text if the API didn't give us a valid ref/name
        if "Unknown" in bowler or "Unknown" in batter:
            actors_match = re.search(r"^([A-Za-z\s\-]+) to ([A-Za-z\s\-]+),", str(text))
            if actors_match:
                if "Unknown" in bowler: bowler = actors_match.group(1).strip()
                if "Unknown" in batter: batter = actors_match.group(2).strip()
            
        over_info = item.get("over", {})
        if isinstance(over_info, dict):
            over_val = over_info.get("number", over_info.get("overs", over_val))
            ball_val = over_info.get("ball", over_info.get("balls", ball_val))
            
            # Format correction for 19.6
            if isinstance(over_val, float):
                actual = float(over_val)
                over_val = int(actual)
                ball_val = int(round((actual - over_val) * 10))
                
            if over_val == 0 and "actual" in over_info:
                try:
                    actual = float(over_info["actual"])
                    over_val = int(actual)
                    ball_val = int(round((actual - over_val) * 10))
                except (ValueError, TypeError):
                    pass
        elif isinstance(over_info, (int, float)):
            try:
                actual = float(over_info)
                over_val = int(actual)
                ball_val = int(round((actual - over_val) * 10))
            except (ValueError, TypeError):
                pass
                
        event = determine_event(str(text), runs)
        
        parsed_deliveries.append({
            "over": over_val,
            "ball": ball_val,
            "bowler": bowler,
            "batter": batter,
            "runs": runs,
            "event": event,
            "description": str(text)
        })
        
    # Optional: Reverse if chronological mapping puts them backwards
    return parsed_deliveries

def main():
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = "https://www.espncricinfo.com/series/icc-men-s-t20-world-cup-2025-26-1502138/india-vs-west-indies-52nd-match-super-eights-group-1-1512770/ball-by-ball-commentary"
        print(f"No URL provided, defaulting to: {url}")
        
    league_id, match_id = extract_match_id(url)
    if not match_id:
        print("Error: Could not extract match ID from the URL.")
        sys.exit(1)
        
    raw_items, session = fetch_commentary(league_id, match_id)
    athlete_map = resolve_athletes(raw_items, session)
    
    deliveries = parse_commentary(raw_items, athlete_map)
    print(f"Parsed {len(deliveries)} deliveries.")
    
    # Save output to data/commentary.json
    out_dir = "data"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "commentary.json")
    
    # Bundle everything nice including the RAG-friendly grouped version
    output_data = {
        "match_id": match_id,
        "commentary": deliveries,
        "grouped_by_over": group_by_over(deliveries)
    }
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)
        
    print(f"Saved {out_file}")

if __name__ == "__main__":
    main()
