import requests
import os
from dotenv import load_dotenv
from datetime import datetime, timezone
import math

load_dotenv()

API_KEY = os.getenv("API_FOOTBALL_KEY")

BASE_URL = "https://v3.football.api-sports.io"

HEADERS = {
    "x-apisports-key": API_KEY
}


def search_team(team_name):
    url = f"{BASE_URL}/teams"

    params = {
        "search": team_name
    }

    response = requests.get(
        url,
        headers=HEADERS,
        params=params
    )

    return response.json()


def find_working_season_for_team(team_id, start_season=2019, end_season=2026, debug=False):
    """Return the first season in [start_season, end_season] with any fixtures for the team."""
    url = f"{BASE_URL}/fixtures"

    for s in range(end_season, start_season - 1, -1):
        params = {"team": team_id, "season": s}
        data = requests.get(url, headers=HEADERS, params=params).json()
        resp = data.get("response", [])
        n = len(resp) if isinstance(resp, list) else 0
        if debug:
            print(f"find_working_season_for_team: team={team_id} season={s} -> {n}")
        if n > 0:
            return s

    return None


def _parse_fixture_dt(match):
    """Parse fixture date into an aware datetime (UTC)."""
    s = match.get("fixture", {}).get("date")
    if not s:
        return None
    try:
        # API returns ISO8601 with offset
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _sort_and_take_last(fixtures, last=30):
    """Sort fixtures by fixture date and return the most recent `last` entries."""
    if not isinstance(fixtures, list):
        return []
    with_dt = []
    for m in fixtures:
        dt = _parse_fixture_dt(m)
        if dt is None:
            continue
        with_dt.append((dt, m))
    with_dt.sort(key=lambda x: x[0])
    return [m for _, m in with_dt[-last:]]


def build_h2h_from_team_fixtures(home_team_id, away_team_id, home_fixtures, away_fixtures, last=10, league_ids=None):
    """Fallback H2H builder using the union of both teams' fixtures."""
    union = []
    for m in (home_fixtures or []):
        union.append(m)
    for m in (away_fixtures or []):
        union.append(m)

    # Filter to matches where the two teams played each other
    out = []
    seen = set()
    for m in union:
        if league_ids is not None and m.get("league", {}).get("id") not in league_ids:
            continue
        hid = m.get("teams", {}).get("home", {}).get("id")
        aid = m.get("teams", {}).get("away", {}).get("id")
        if {hid, aid} != {home_team_id, away_team_id}:
            continue
        fid = m.get("fixture", {}).get("id")
        if fid and fid in seen:
            continue
        if fid:
            seen.add(fid)
        out.append(m)

    out = _sort_and_take_last(out, last=last)
    return out


def get_last_matches(team_id, last=30, season=2024, status=None, debug=False):
    """Fetch recent matches for a team.

    Note: On API-Football free plans, the `last` query parameter may be unavailable.
    This function therefore fetches fixtures for the season, then sorts by date and slices locally.
    """
    url = f"{BASE_URL}/fixtures"

    seasons_to_try = [season, season - 1, season + 1]
    tried = set()

    def _fetch(s):
        params = {"team": team_id, "season": s}
        if status:
            params["status"] = status

        data = requests.get(url, headers=HEADERS, params=params).json()
        api_errors = data.get("errors")

        resp = data.get("response", [])
        if isinstance(resp, list) and len(resp) > 0:
            data["response"] = _sort_and_take_last(resp, last=last)
            if debug:
                print(
                    f"get_last_matches: team={team_id} season={s} status={status} -> {len(data['response'])} matches"
                )
            return data

        if debug:
            if api_errors:
                print(
                    f"get_last_matches: team={team_id} season={s} status={status} -> 0 matches (api errors: {api_errors})"
                )
            else:
                print(f"get_last_matches: team={team_id} season={s} status={status} -> 0 matches")
        return None

    for s in seasons_to_try:
        if s in tried:
            continue
        tried.add(s)
        out = _fetch(s)
        if out is not None:
            return out

    detected = find_working_season_for_team(team_id, start_season=2019, end_season=2026, debug=debug)
    if detected is not None:
        out = _fetch(detected)
        if out is not None:
            return out

    return {"response": []}


def get_h2h(team1_id, team2_id, last=10):
    url = f"{BASE_URL}/fixtures/headtohead"

    params = {
        "h2h": f"{team1_id}-{team2_id}",
        "last": last
    }

    response = requests.get(
        url,
        headers=HEADERS,
        params=params
    )

    return response.json()


def get_team_statistics(team_id, league_id, season):
    url = f"{BASE_URL}/teams/statistics"

    params = {
        "league": league_id,
        "season": season,
        "team": team_id
    }

    response = requests.get(
        url,
        headers=HEADERS,
        params=params
    )

    return response.json()

def analyze_home_or_draw(features):
    # Use neutral priors when data is missing
    prior_home_unbeaten = 0.60
    prior_away_loss = 0.30
    prior_h2h_home_unbeaten = 0.50

    home_unbeaten = features.get("home_unbeaten_rate")
    away_loss = features.get("away_loss_rate")
    h2h_home_unbeaten = features.get("h2h_home_unbeaten")

    if home_unbeaten is None:
        home_unbeaten = prior_home_unbeaten
    if away_loss is None:
        away_loss = prior_away_loss
    if h2h_home_unbeaten is None:
        h2h_home_unbeaten = prior_h2h_home_unbeaten

    score = 0
    score += home_unbeaten * 0.45
    score += away_loss * 0.35

    # Only use H2H if we have enough samples; otherwise blend toward prior implicitly
    if features.get("h2h_sample_size", 0) >= 3:
        score += h2h_home_unbeaten * 0.20
    else:
        score += prior_h2h_home_unbeaten * 0.20

    # Confidence scaling: shrink toward 50 when there is little data
    conf_home = features.get("home_confidence", 0.0)
    conf_away = features.get("away_confidence", 0.0)
    conf_h2h = features.get("h2h_confidence", 0.0)
    overall_conf = 0.45 * conf_home + 0.35 * conf_away + 0.20 * conf_h2h

    raw = score * 100
    calibrated = 50.0 + (raw - 50.0) * max(0.0, min(1.0, overall_conf))
    return round(calibrated, 2)


def analyze_home_or_away(features, *, baseline_draw=0.27):
    # Neutral priors
    prior_draw = baseline_draw

    home_draw_rate = features.get("home_draw_rate")
    away_draw_rate = features.get("away_draw_rate")
    if home_draw_rate is None:
        home_draw_rate = prior_draw
    if away_draw_rate is None:
        away_draw_rate = prior_draw

    h2h_draw_rate = features.get("h2h_draw_rate")
    if h2h_draw_rate is None:
        h2h_draw_rate = prior_draw

    h2h_n = features.get("h2h_sample_size", 0)
    h2h_w = 0.20 if h2h_n >= 3 else 0.0

    base_w = 1.0 - h2h_w
    w_home = base_w * 0.50
    w_away = base_w * 0.50

    draw_risk = (home_draw_rate * w_home) + (away_draw_rate * w_away) + (h2h_draw_rate * h2h_w)
    draw_risk = max(0.0, min(1.0, draw_risk))
    no_draw = 1.0 - draw_risk

    denom = max(1e-9, max(1.0 - baseline_draw, baseline_draw))
    calibrated = 50.0 + 50.0 * ((no_draw - (1.0 - baseline_draw)) / denom)

    # Confidence scaling
    conf_home = features.get("home_confidence", 0.0)
    conf_away = features.get("away_confidence", 0.0)
    conf_h2h = features.get("h2h_confidence", 0.0)
    overall_conf = 0.50 * conf_home + 0.50 * conf_away
    if h2h_w > 0:
        overall_conf = 0.40 * conf_home + 0.40 * conf_away + 0.20 * conf_h2h

    final_score = 50.0 + (calibrated - 50.0) * max(0.0, min(1.0, overall_conf))
    return round(max(0.0, min(100.0, final_score)), 2)



def analyze_draw_or_away(features):
    # Neutral priors
    prior_away_unbeaten = 0.50
    prior_home_not_winning = 0.50
    prior_h2h_away_unbeaten = 0.50

    away_unbeaten = features.get("away_unbeaten_rate")
    home_not_winning = features.get("home_not_winning_rate")
    h2h_away_unbeaten = features.get("h2h_away_unbeaten")

    if away_unbeaten is None:
        away_unbeaten = prior_away_unbeaten
    if home_not_winning is None:
        home_not_winning = prior_home_not_winning
    if h2h_away_unbeaten is None:
        h2h_away_unbeaten = prior_h2h_away_unbeaten

    score = 0
    score += away_unbeaten * 0.50
    score += home_not_winning * 0.30

    if features.get("h2h_sample_size", 0) >= 3:
        score += h2h_away_unbeaten * 0.20
    else:
        score += prior_h2h_away_unbeaten * 0.20

    # Confidence scaling
    conf_home = features.get("home_confidence", 0.0)
    conf_away = features.get("away_confidence", 0.0)
    conf_h2h = features.get("h2h_confidence", 0.0)
    overall_conf = 0.30 * conf_home + 0.50 * conf_away + 0.20 * conf_h2h

    raw = score * 100
    calibrated = 50.0 + (raw - 50.0) * max(0.0, min(1.0, overall_conf))
    return round(calibrated, 2)


def extract_match_features(
    home_matches,
    away_matches,
    h2h_matches,
    home_team_id,
    away_team_id,
    *,
    league_id=None,
    league_ids=None,
    debug=False,
    half_life_days=45,
):
    features = {}

    if not home_matches or not away_matches:
        if debug:
            print(
                "WARNING: empty match dataset used (home_matches=%s, away_matches=%s)"
                % (len(home_matches) if home_matches else 0, len(away_matches) if away_matches else 0)
            )
        # Keep neutral defaults, but don't hide the fact it happened.
        features["home_unbeaten_rate"] = 0.0
        features["home_win_rate"] = 0.0
        features["away_loss_rate"] = 0.0
        features["away_not_winning_rate"] = 0.0
        features["h2h_home_unbeaten"] = 0.0
        return features

    allowed_leagues = None
    if league_ids is not None:
        allowed_leagues = set(league_ids)
    elif league_id is not None:
        allowed_leagues = {league_id}

    now = datetime.now(timezone.utc)
    ln2 = math.log(2)

    def weight_for_match(match):
        dt = _parse_fixture_dt(match)
        if dt is None:
            return 0.0
        age_days = max(0.0, (now - dt).total_seconds() / 86400.0)
        # w = 0.5^(age/half_life)
        return math.exp(-ln2 * (age_days / max(1e-9, half_life_days)))

    # -------------------------
    # HOME FORM (time-weighted)
    # -------------------------
    w_home_total = 0.0
    w_home_unbeaten = 0.0
    w_home_wins = 0.0
    w_home_draws = 0.0
    w_home_losses = 0.0

    # Goal-based performance
    w_home_gf = 0.0
    w_home_ga = 0.0

    valid_home_matches = 0

    for match in home_matches:
        if allowed_leagues is not None and match.get("league", {}).get("id") not in allowed_leagues:
            continue

        home_id = match["teams"]["home"]["id"]
        away_id = match["teams"]["away"]["id"]

        goals_home = match["goals"]["home"]
        goals_away = match["goals"]["away"]
        if goals_home is None or goals_away is None:
            continue

        if home_team_id != home_id and home_team_id != away_id:
            continue

        valid_home_matches += 1
        w = weight_for_match(match)
        if w <= 0:
            continue

        w_home_total += w

        # get result from home-team perspective
        if home_id == home_team_id:
            gf, ga = goals_home, goals_away
        else:
            gf, ga = goals_away, goals_home

        if gf > ga:
            w_home_wins += w
            w_home_unbeaten += w
        elif gf == ga:
            w_home_draws += w
            w_home_unbeaten += w
        else:
            w_home_losses += w

        w_home_gf += w * float(gf)
        w_home_ga += w * float(ga)

    denom_home = w_home_total if w_home_total > 0 else 1.0

    if debug and valid_home_matches == 0:
        print("WARNING: no valid home-team matches after filtering")

    features["home_unbeaten_rate"] = w_home_unbeaten / denom_home
    features["home_win_rate"] = w_home_wins / denom_home
    features["home_draw_rate"] = w_home_draws / denom_home
    features["home_not_winning_rate"] = (w_home_draws + w_home_losses) / denom_home

    features["home_goals_for_per_match"] = w_home_gf / denom_home
    features["home_goals_against_per_match"] = w_home_ga / denom_home
    features["home_goal_diff_per_match"] = (w_home_gf - w_home_ga) / denom_home

    # Track sample sizes + availability
    features["home_sample_size"] = valid_home_matches
    features["home_data_available"] = valid_home_matches > 0

    # Confidence in [0,1] using a soft saturation curve
    # 0 matches => 0, 10 matches => ~0.67, 20 matches => ~0.80, 30 matches => ~0.86
    features["home_confidence"] = valid_home_matches / (valid_home_matches + 5.0) if valid_home_matches > 0 else 0.0

    # -------------------------
    # AWAY FORM (time-weighted)
    # -------------------------
    w_away_total = 0.0
    w_away_unbeaten = 0.0
    w_away_wins = 0.0
    w_away_draws = 0.0
    w_away_losses = 0.0

    w_away_gf = 0.0
    w_away_ga = 0.0

    valid_away_matches = 0

    for match in away_matches:
        if allowed_leagues is not None and match.get("league", {}).get("id") not in allowed_leagues:
            continue

        home_id = match["teams"]["home"]["id"]
        away_id = match["teams"]["away"]["id"]

        goals_home = match["goals"]["home"]
        goals_away = match["goals"]["away"]
        if goals_home is None or goals_away is None:
            continue

        if away_team_id != home_id and away_team_id != away_id:
            continue

        valid_away_matches += 1
        w = weight_for_match(match)
        if w <= 0:
            continue

        w_away_total += w

        # result from away-team perspective
        if away_id == away_team_id:
            gf, ga = goals_away, goals_home
        else:
            gf, ga = goals_home, goals_away

        if gf > ga:
            w_away_wins += w
            w_away_unbeaten += w
        elif gf == ga:
            w_away_draws += w
            w_away_unbeaten += w
        else:
            w_away_losses += w

        w_away_gf += w * float(gf)
        w_away_ga += w * float(ga)

    denom_away = w_away_total if w_away_total > 0 else 1.0

    if debug and valid_away_matches == 0:
        print("WARNING: no valid away-team matches after filtering")

    features["away_loss_rate"] = w_away_losses / denom_away
    features["away_not_winning_rate"] = (w_away_losses + w_away_draws) / denom_away
    features["away_unbeaten_rate"] = w_away_unbeaten / denom_away
    features["away_draw_rate"] = w_away_draws / denom_away

    features["away_goals_for_per_match"] = w_away_gf / denom_away
    features["away_goals_against_per_match"] = w_away_ga / denom_away
    features["away_goal_diff_per_match"] = (w_away_gf - w_away_ga) / denom_away

    features["away_sample_size"] = valid_away_matches
    features["away_data_available"] = valid_away_matches > 0
    features["away_confidence"] = valid_away_matches / (valid_away_matches + 5.0) if valid_away_matches > 0 else 0.0

    # -------------------------
    # H2H (time-weighted)
    # -------------------------
    w_h2h_total = 0.0
    w_h2h_home_unbeaten = 0.0
    w_h2h_away_unbeaten = 0.0
    w_h2h_draws = 0.0
    valid_h2h_matches = 0

    for match in h2h_matches:
        if allowed_leagues is not None and match.get("league", {}).get("id") not in allowed_leagues:
            continue

        home_id = match["teams"]["home"]["id"]
        away_id = match["teams"]["away"]["id"]

        goals_home = match["goals"]["home"]
        goals_away = match["goals"]["away"]
        if goals_home is None or goals_away is None:
            continue

        valid_h2h_matches += 1
        w = weight_for_match(match)
        if w <= 0:
            continue

        w_h2h_total += w

        if goals_home == goals_away:
            w_h2h_draws += w

        # home team unbeaten
        if (home_id == home_team_id and goals_home >= goals_away) or (
            away_id == home_team_id and goals_away >= goals_home
        ):
            w_h2h_home_unbeaten += w

        # away team unbeaten
        if (home_id == away_team_id and goals_home >= goals_away) or (
            away_id == away_team_id and goals_away >= goals_home
        ):
            w_h2h_away_unbeaten += w

    features["h2h_sample_size"] = valid_h2h_matches
    features["h2h_data_available"] = valid_h2h_matches > 0
    features["h2h_confidence"] = valid_h2h_matches / (valid_h2h_matches + 3.0) if valid_h2h_matches > 0 else 0.0

    denom_h2h = w_h2h_total if w_h2h_total > 0 else 1.0
    features["h2h_home_unbeaten"] = w_h2h_home_unbeaten / denom_h2h
    features["h2h_away_unbeaten"] = w_h2h_away_unbeaten / denom_h2h
    features["h2h_draw_rate"] = w_h2h_draws / denom_h2h

    if debug and valid_h2h_matches == 0:
        print("WARNING: no valid H2H matches after filtering")

    return features


def run_analysis():
    
    print("API KEY:", API_KEY)

    home_team = "Cerro Porteno"
    away_team = "Sporting Cristal"
    print(search_team("Cerro") )
    

    # 1. Get teams
    home_data = search_team(home_team)
    away_data = search_team(away_team)

    home_team_id = home_data["response"][0]["team"]["id"]
    away_team_id = away_data["response"][0]["team"]["id"]

    # 2. Fetch last 30 matches (auto-detect season if needed)
    home_matches = get_last_matches(home_team_id, 30, season=2024, status=None, debug=True)["response"]
    away_matches = get_last_matches(away_team_id, 30, season=2024, status=None, debug=True)["response"]

    # 3. H2H (fallback to built H2H if endpoint unusable/empty)
    h2h_raw = get_h2h(home_team_id, away_team_id, 10)
    h2h_matches = h2h_raw.get("response", []) if isinstance(h2h_raw, dict) else []

    # Competition filtering: default to domestic leagues for the HOME team.
    # Cerro Porteño 2024 fixtures include Apertura (250) and Clausura (252).
    # Set to None to include all competitions.
    LEAGUE_IDS = {250, 252}

    h2h_filtered = build_h2h_from_team_fixtures(
        home_team_id,
        away_team_id,
        home_matches,
        away_matches,
        last=10,
        league_ids=LEAGUE_IDS,
    )
    if not h2h_matches:
        h2h_matches = h2h_filtered

    # 4. Extract features (time-weighted + competition filtering)
    features = extract_match_features(
        home_matches,
        away_matches,
        h2h_matches,
        home_team_id,
        away_team_id,
        league_ids=LEAGUE_IDS,
        debug=True,
        half_life_days=45,
    )

    # 5. Scores
    score_1x = analyze_home_or_draw(features)
    score_12 = analyze_home_or_away(features)
    score_x2 = analyze_draw_or_away(features)

    print("FEATURES:", features)
    print("HOME OR DRAW (1X) SCORE:", score_1x)
    print("HOME OR AWAY (12) SCORE:", score_12)
    print("DRAW OR AWAY (X2) SCORE:", score_x2)

if __name__ == "__main__":
    run_analysis()