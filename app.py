import json, re, urllib.parse, html
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
MODEL = BASE / "bike_resale_model.joblib"
BIKES = BASE / "bike_names.json"
st.set_page_config(page_title="Bike Resale AI", layout="centered", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
.stApp{background:#f7f7f5;font-family:'DM Sans',sans-serif}.block-container{max-width:820px;padding-top:2.5rem!important}
/* Prevent Streamlit's fixed toolbar from covering the app's brand header. */
[data-testid="stHeader"]{background:transparent!important;height:0!important;min-height:0!important;display:none!important}
[data-testid="stToolbar"]{display:none!important}
[data-testid="stDecoration"]{display:none!important}
.brand{background:#171717;color:#fff;padding:18px 20px;border-radius:15px;border-bottom:4px solid #facc15;display:flex;gap:13px;align-items:center;margin-bottom:24px}
.mark{background:#facc15;color:#171717;border-radius:10px;padding:10px}.brandname{font-family:Manrope,sans-serif;font-weight:800;font-size:20px;letter-spacing:.04em}.brandsub{font-size:10px;color:#bdbdbd;letter-spacing:.12em}
.hero{font-family:Manrope,sans-serif;font-size:33px;font-weight:800;line-height:1.15;letter-spacing:-.04em;color:#171717;margin:8px 0}.intro{color:#737373;font-size:14px;line-height:1.6;margin-bottom:20px}
.botlabel,.userlabel{font-size:10px;font-weight:800;letter-spacing:.12em;color:#737373!important;text-transform:uppercase;margin-top:10px}.userlabel{text-align:right;color:#a16207!important}
.bot{background:#fff!important;color:#171717!important;border:1px solid #e5e5e5;border-radius:4px 15px 15px 15px;padding:13px 15px;margin:5px 0 12px;line-height:1.6}.bot *{color:#171717!important}
.user{background:#fef3c7!important;color:#422006!important;border:1px solid #fde68a;border-radius:15px 4px 15px 15px;padding:12px 15px;margin:5px 0 12px;line-height:1.6}.user *{color:#422006!important}
.result{background:#171717;color:#fff;border-radius:15px;padding:21px;border-bottom:4px solid #facc15;margin:12px 0}.label{color:#d4d4d4;font-size:11px;letter-spacing:.1em}.price{font-family:Manrope,sans-serif;font-weight:800;font-size:36px;margin:8px 0}.note{font-size:12px;color:#a3a3a3;line-height:1.5}
.panel{background:white;border:1px solid #e5e5e5;border-radius:12px;padding:15px;margin:9px 0}.paneltitle{font-size:13px;font-weight:800}.small{font-size:12px;color:#737373;line-height:1.5;margin-top:4px}
.warning{background:#fff7ed;border-left:4px solid #ea580c;border-radius:8px;padding:11px 13px;color:#7c2d12;font-size:12px;line-height:1.55;margin:12px 0}
.green{background:#f0fdf4;border-left:4px solid #15803d;border-radius:8px;padding:11px 13px;color:#14532d;font-size:12px;line-height:1.55;margin:12px 0}
div.stButton>button{background:#facc15!important;color:#171717!important;border:1px solid #eab308;border-radius:10px;font-weight:800;min-height:42px}div.stButton>button p{color:#171717!important}.stMarkdown p,.stMarkdown strong{color:#171717}a{color:#b45309!important;font-weight:700;text-decoration:none}.foot{text-align:center;color:#a3a3a3;font-size:10px;margin:22px 0}
/* Keep all Streamlit form controls readable in light mode. */
.stApp, .stApp p, .stApp label, .stApp .stMarkdown, .stApp .stCaption, .stApp [data-testid="stCaptionContainer"] p {color:#171717!important}
.stApp h1,.stApp h2,.stApp h3,.stApp h4 {color:#171717!important}
.stRadio [role="radiogroup"] label, .stRadio [role="radiogroup"] label *, .stRadio label, .stRadio label * {color:#171717!important; opacity:1!important}
.stRadio [role="radio"] {border-color:#525252!important}
.stRadio [role="radio"][aria-checked="true"] {background:#facc15!important;border-color:#ca8a04!important}
.stRadio [role="radio"] > div {border-color:#525252!important}
.stSelectbox label,.stNumberInput label,.stTextInput label {color:#171717!important}
.stSelectbox div[data-baseweb="select"] *, .stNumberInput input {color:#171717!important}
.stProgress > div > div > div {background:#facc15!important}
[data-testid="stProgressBar"] {background:#e5e5e5!important}
.stApp [data-testid="stWidgetLabel"] p {color:#171717!important}
.stApp [data-testid="stMarkdownContainer"] p {color:#171717!important}


/* Global control contrast fix: light surfaces with dark readable text. */
.stApp [data-testid="stNumberInput"] input,
.stApp [data-testid="stTextInput"] input,
.stApp [data-baseweb="input"] input,
.stApp [data-baseweb="textarea"] textarea {
  background: #ffffff !important;
  background-color: #ffffff !important;
  color: #171717 !important;
  -webkit-text-fill-color: #171717 !important;
  caret-color: #171717 !important;
  border-color: #a3a3a3 !important;
  opacity: 1 !important;
}
.stApp [data-testid="stNumberInput"] input::placeholder,
.stApp [data-testid="stTextInput"] input::placeholder,
.stApp [data-baseweb="input"] input::placeholder,
.stApp [data-baseweb="textarea"] textarea::placeholder {
  color: #737373 !important;
  -webkit-text-fill-color: #737373 !important;
  opacity: 1 !important;
}
.stApp [data-testid="stNumberInput"] button {
  color: #171717 !important;
  background: #f5f5f5 !important;
}
.stApp [data-baseweb="select"] > div {
  background: #ffffff !important;
  color: #171717 !important;
  border-color: #a3a3a3 !important;
}
.stApp [data-baseweb="select"] input,
.stApp [data-baseweb="select"] div,
.stApp [data-baseweb="select"] span {
  color: #171717 !important;
  -webkit-text-fill-color: #171717 !important;
}
.stApp [role="radiogroup"] label,
.stApp [role="radiogroup"] label * {
  color: #171717 !important;
  -webkit-text-fill-color: #171717 !important;
  opacity: 1 !important;
}

</style>
""", unsafe_allow_html=True)

def is_side_conversation(text):
    low = text.lower().strip()
    return low in {"hi", "hello", "hey", "hii", "yo", "thanks", "thank you", "ok", "okay", "cool", "good morning", "good afternoon", "good evening"} or low.endswith("?")

def icon(name="brain", color="#171717", size=24):
    paths={
    "bike":'<circle cx="5.5" cy="17.5" r="3.5"/><circle cx="18.5" cy="17.5" r="3.5"/><path d="M8 17.5 11 9h4l3.5 8.5M9.5 13h6M5.5 17.5H10l3-8.5M14 9h3"/>',
    "brain":'<path d="M12 5a4 4 0 0 0-7.5 2A4.5 4.5 0 0 0 5 16a4 4 0 0 0 7 2V5ZM12 5a4 4 0 0 1 7.5 2A4.5 4.5 0 0 1 19 16a4 4 0 0 1-7 2"/>',
    "external":'<path d="M14 4h6v6M20 4l-9 9"/><path d="M18 13v6H5V6h6"/>'}
    return f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">{paths[name]}</svg>'

@st.cache_resource
def load_model(): return joblib.load(MODEL)
@st.cache_data
def load_bikes():
    with open(BIKES, encoding="utf-8") as f: data=json.load(f)
    if isinstance(data,dict):
        for k in ("bike_names","bikes","names","models"):
            if isinstance(data.get(k),list): data=data[k]; break
        else: data=list(data.keys())
    return sorted(set(map(str,data)))
def rupees(x): return f"₹{x:,.0f}"
def market_links(bike,city):
    q=urllib.parse.quote_plus(f"{bike} used bike {city}")
    slug=urllib.parse.quote_plus(bike).replace("+","-")
    return [
    ("OLX India",f"https://www.olx.in/items/q-{slug}"),
    ("Facebook Marketplace",f"https://www.facebook.com/marketplace/search/?query={q}"),
    ("BikeWale","https://www.bikewale.com/used/"),
    ("CarDekho",f"https://www.cardekho.com/used-bikes+{urllib.parse.quote_plus(city)}"),
    ("Google cross-market search",f"https://www.google.com/search?q={urllib.parse.quote_plus(bike+' used bike price '+city+' OLX BikeWale CarDekho')}")]

defaults={"messages":[],"step":"company","company":None,"bike":None,"year":None,"km":None,"owner":None,"seller":None,"city":None,"prediction":None,"bike_options":[],"pending_year":None}
for k,v in defaults.items():
    if k not in st.session_state: st.session_state[k]=v

st.markdown(f'<div class="brand"><div class="mark">{icon("bike",size=26)}</div><div><div class="brandname">BIKE RESALE AI</div><div class="brandsub">MOTORCYCLE PRICE ESTIMATOR</div></div></div><div style="color:#a16207;font-size:10px;font-weight:800;letter-spacing:.15em">YOUR BIKE. YOUR MARKET. YOUR ESTIMATE.</div><div class="hero">Let’s figure out what your bike is worth.</div><div class="intro">Answer a short series of questions to estimate your motorcycle’s resale value.</div>',unsafe_allow_html=True)

if not MODEL.exists() or not BIKES.exists():
    st.error("Keep app.py, bike_resale_model.joblib and bike_names.json in the same folder.")
    st.stop()
model=load_model(); bikes=load_bikes()
if not bikes: st.error("No bike models found in bike_names.json."); st.stop()

COMPANY_ALIASES = {
    "Royal Enfield": ["royal enfield", "re"], "Honda": ["honda"], "Hero": ["hero", "hero motocorp"],
    "TVS": ["tvs"], "Bajaj": ["bajaj"], "Yamaha": ["yamaha"], "Suzuki": ["suzuki"],
    "KTM": ["ktm"], "Kawasaki": ["kawasaki"], "Jawa": ["jawa"], "Yezdi": ["yezdi"],
    "Aprilia": ["aprilia"], "BMW": ["bmw"], "Triumph": ["triumph"], "Harley-Davidson": ["harley davidson", "harley"],
    "Mahindra": ["mahindra"], "LML": ["lml"], "Benelli": ["benelli"], "Husqvarna": ["husqvarna"],
    "Ola Electric": ["ola electric", "ola"], "Ather": ["ather"], "Ampere": ["ampere"], "Bgauss": ["bgauss"]
}
# Dataset names often omit the manufacturer (e.g. "Activa 3G" rather than "Honda Activa 3G").
MODEL_HINTS = {
    "Honda": ["activa", "shine", "unicorn", "dio", "dream", "hornet", "livo", "sp 125", "cb", "aviator", "grazia", "xcd"],
    "Hero": ["splendor", "passion", "hf deluxe", "glamour", "karizma", "xpulse", "xtreme", "maestro", "destini", "pleasure"],
    "TVS": ["apache", "jupiter", "ntorq", "xl 100", "sport", "radeon", "star city", "scooty", "victor", "ronin", "zest"],
    "Bajaj": ["pulsar", "platina", "ct 100", "ct 110", "discover", "avenger", "dominar", "chetak", "v15", "v12"],
    "Yamaha": ["fazer", "fz", "fzs", "r15", "mt-15", "mt 15", "fascino", "ray", "saluto", "sz-r", "szr", "crux", "libero"],
    "Suzuki": ["access", "burgman", "gixxer", "intruder", "hayate", "sling shot", "swish"],
    "Royal Enfield": ["classic", "bullet", "thunderbird", "continental gt", "himalayan", "interceptor", "meteor", "electra", "thunder 350"],
    "KTM": ["duke", "rc 125", "rc 200", "rc 390", "adventure 390"],
    "Kawasaki": ["ninja", "z900", "versys", "vulcan"], "Jawa": ["jawa", "forty two", "42"],
    "Yezdi": ["yezdi", "roadster", "scrambler", "adventure"], "Aprilia": ["sr 150", "sr 125", "storm", "rs 457"],
    "BMW": ["g 310", "f 750", "r 1250", "s 1000"], "Triumph": ["street triple", "speed 400", "scrambler 400", "bonneville"],
    "Harley-Davidson": ["street 750", "iron 883", "x440", "fat boy"], "Mahindra": ["mojo", "centuro", "gusto"],
    "LML": ["freedom", "adreno", "vespa"], "Benelli": ["tnt", "imperiale", "leoncino"], "Husqvarna": ["vitpilen", "svartpilen"],
    "Ola Electric": ["s1", "s1 pro", "s1 air"], "Ather": ["450x", "450 plus", "rizta"], "Ampere": ["magnus", "zeal", "reo"], "Bgauss": ["b8", "d15", "c12"]
}

def models_for_company(company):
    aliases = COMPANY_ALIASES.get(company, [company.lower()])
    hints = MODEL_HINTS.get(company, [])
    found = []
    for name in bikes:
        low_name = name.lower()
        if any(re.search(r"(?<![a-z])" + re.escape(a) + r"(?![a-z])", low_name) for a in aliases) or any(h in low_name for h in hints):
            found.append(name)
    # If names are manufacturer-prefixed, prefer those; otherwise use brand-specific model hints.
    return sorted(set(found), key=str.casefold)

def reply(text): st.session_state.messages.append({"role":"assistant","content":text})
def user(text): st.session_state.messages.append({"role":"user","content":text})
def find_year(t):
    m=re.search(r'\b(?:19|20)\d{2}\b',t); return int(m.group()) if m else None
def bike_candidates(t, limit=8):
    """Find genuinely similar model names; avoid unrelated token-overlap guesses."""
    from difflib import SequenceMatcher
    def norm(value):
        value = value.lower().replace("active", "activa")
        value = re.sub(r"[^a-z0-9]+", " ", value)
        return " ".join(value.split())

    query = norm(t)
    ignored = {"hi", "hello", "hey", "hii", "yo", "yes", "no", "ok", "okay", "bike", "model", "motorcycle", "my", "is", "a", "the", "and", "year"}
    tokens = [x for x in query.split() if x not in ignored and len(x) > 1]
    query = " ".join(tokens)
    if len(query.replace(" ", "")) < 3:
        return []

    scored = []
    q_tokens = set(query.split())
    for name in bikes:
        candidate = norm(name)
        c_tokens = set(candidate.split())
        if not candidate:
            continue
        if candidate == query:
            score = 1.0
        elif candidate in query or query in candidate:
            score = 0.92
        else:
            seq = SequenceMatcher(None, query, candidate).ratio()
            overlap = len(q_tokens & c_tokens) / max(1, len(q_tokens | c_tokens))
            # Compare each query token to candidate tokens to tolerate small typos.
            token_sim = 0.0
            if q_tokens and c_tokens:
                token_sim = sum(max([SequenceMatcher(None, q, c).ratio() for c in c_tokens] or [0]) for q in q_tokens) / len(q_tokens)
            score = max(seq, 0.55 * overlap + 0.45 * token_sim)
        if score >= 0.56:
            scored.append((score, name))
    scored.sort(key=lambda item: item[0], reverse=True)
    # Keep only near-best matches; never flood the user with unrelated models.
    if not scored:
        return []
    best = scored[0][0]
    return [name for score, name in scored if score >= max(0.56, best - 0.12)][:limit]

def find_bike(t):
    """Only accept a unique strong exact match; ambiguous/partial matches require confirmation."""
    query = re.sub(r"[^a-z0-9]+", " ", t.lower().replace("active", "activa")).strip()
    def norm(value):
        return re.sub(r"[^a-z0-9]+", " ", value.lower().replace("active", "activa")).strip()
    exact = [b for b in bikes if norm(b) == query]
    if exact:
        return exact[0]
    contained = [b for b in bikes if norm(b) and norm(b) in query]
    # A full name within a phrase like 'Activa 3g 2018 model' is okay only if unique.
    if len(contained) == 1:
        return contained[0]
    return None
def handle(t):
    low=t.lower().strip()
    step=st.session_state.step

    # Extract details from free-form text, regardless of which question is currently active.
    # Only update fields when the text contains a reasonably explicit signal.
    if low in {"cancel", "stop", "never mind", "nevermind"}:
        reply("No problem. Your current details are still here. Whenever you're ready, continue in your own words or type ‘new valuation’ to restart.")
        return

    # Year can be mentioned anywhere in the message.
    parsed_year = find_year(t)
    if parsed_year and 1988 <= parsed_year <= 2026 and (re.search(r"\b(?:19|20)\d{2}\b", t)):
        st.session_state.year = parsed_year

    # Mileage supports 25k, 25,000 km, 25000 kilometres, etc.
    parsed_km = None
    km_short = re.search(r"\b([0-9]+(?:\.[0-9]+)?)\s*k\b", low)
    km_unit = re.search(r"\b([0-9][0-9,]*)\s*(?:km|kms|kilometres?|kilometers?)\b", low)
    if km_unit:
        parsed_km = int(km_unit.group(1).replace(",", ""))
    elif km_short:
        parsed_km = int(float(km_short.group(1)) * 1000)
    if parsed_km is not None and 0 <= parsed_km <= 1000000:
        st.session_state.km = parsed_km

    owner_patterns = [
        ("4th owner", r"\b(?:4th|fourth)\s*(?:previous\s*)?owner\b|\b4\s*owners?\b|\bfour\s*owners?\b"),
        ("3rd owner", r"\b(?:3rd|third)\s*(?:previous\s*)?owner\b|\b3\s*owners?\b|\bthree\s*owners?\b|\bowner\s*(?:number\s*)?3\b"),
        ("2nd owner", r"\b(?:2nd|second)\s*(?:previous\s*)?owner\b|\b2\s*owners?\b|\btwo\s*owners?\b|\bowner\s*(?:number\s*)?2\b"),
        ("1st owner", r"\b(?:1st|first)\s*(?:previous\s*)?owner\b|\b(?:single|one)\s*owner\b|\bowner\s*(?:number\s*)?1\b"),
    ]
    for label, pattern in owner_patterns:
        if re.search(pattern, low):
            st.session_state.owner = label
            break

    if re.search(r"\bdealer(ship)?\b", low):
        st.session_state.seller = "Dealer"
    elif re.search(r"\b(individual|private seller|selling privately|sell it myself|personal seller)\b", low):
        st.session_state.seller = "Individual"

    city_aliases = {
        "Hyderabad": ["hyderabad", "hyd", "hyderbad", "hydrabad"], "Bengaluru": ["bengaluru", "bangalore", "blr"],
        "Chennai": ["chennai", "madras"], "Mumbai": ["mumbai", "bombay"], "Pune": ["pune"],
        "Delhi": ["delhi", "new delhi", "ncr"], "Kolkata": ["kolkata", "calcutta"], "Ahmedabad": ["ahmedabad"],
        "Jaipur": ["jaipur"], "Lucknow": ["lucknow"], "India-wide": ["india-wide", "india wide", "all india", "pan india"]
    }
    for canonical, aliases in city_aliases.items():
        if any(re.search(r"(?<![a-z])" + re.escape(alias) + r"(?![a-z])", low) for alias in aliases):
            st.session_state.city = canonical
            break

    # If a bike is not selected yet, try to identify a model from the message after stripping
    # years, mileage, ownership and seller/location phrases. Avoid guessing if there are several matches.
    if not st.session_state.bike and step in {"bike", "confirm_bike"}:
        model_text = re.sub(r"\b(?:19|20)\d{2}\b", " ", t, flags=re.I)
        model_text = re.sub(r"\b[0-9][0-9,]*(?:\.[0-9]+)?\s*(?:k|km|kms|kilometres?|kilometers?)\b", " ", model_text, flags=re.I)
        model_text = re.sub(r"\b(?:1st|2nd|3rd|4th|first|second|third|fourth|one|two|three|four)\s*(?:previous\s*)?owners?\b|\b[1-4]\s*owners?\b", " ", model_text, flags=re.I)
        model_text = re.sub(r"\b(?:individual|private seller|dealer|dealership|in|from|based in|located in|city|year|model|bike|motorcycle|my|i have|i own|i am|im|i|am|it's|its|is|a|an|the|and|around|about|approximately|approx|driven|covered|done|kms|km|kilometres|kilometers|owner|owners|hyderabad|hyd|hyderbad|hydrabad|bengaluru|bangalore|blr|chennai|madras|mumbai|bombay|pune|delhi|new delhi|ncr|kolkata|calcutta|ahmedabad|jaipur|lucknow|india-wide|india wide|all india|pan india)\b", " ", model_text, flags=re.I)
        model_text = re.sub(r"[,;.!]+", " ", model_text).strip()
        if model_text and len(re.sub(r"\W", "", model_text)) >= 3:
            found = find_bike(model_text)
            if found:
                st.session_state.bike = found
            else:
                candidates = bike_candidates(model_text)
                # Prefer a match only when the entire model phrase is contained uniquely.
                if len(candidates) == 1 and len(model_text.split()) >= 2:
                    st.session_state.bike = candidates[0]

    # If the user provided enough details in one natural-language message, run valuation directly.
    if all(st.session_state.get(k) is not None for k in ["bike", "year", "km", "owner", "seller", "city"]):
        try:
            row = pd.DataFrame([{"name": st.session_state.bike, "year": st.session_state.year,
                "seller_type": st.session_state.seller, "owner": st.session_state.owner,
                "km_driven": st.session_state.km}])
            prediction = max(0.0, float(model.predict(row)[0]))
            st.session_state.prediction = prediction
            st.session_state.step = "results"
            reply(f"I picked out the details from your message: {st.session_state.bike}, {st.session_state.year}, {st.session_state.km:,} km, {st.session_state.owner}, {st.session_state.seller.lower()} seller, {st.session_state.city}. The model estimates {rupees(prediction)}. If I misunderstood any detail, you can ask me to change it.")
        except Exception as e:
            reply(f"I identified the details, but the model couldn't calculate the estimate: {e}")
        return

    # If several fields were supplied, keep them and ask only for the next missing one.
    if any(st.session_state.get(k) is not None for k in ["year", "km", "owner", "seller", "city"]) and step == "bike" and st.session_state.bike:
        if st.session_state.year is None:
            st.session_state.step = "year"
            reply(f"I picked up the bike model as {st.session_state.bike}. What year is it? I've kept the other details you gave me.")
            return
        if st.session_state.km is None or st.session_state.owner is None:
            st.session_state.step = "usage"
            missing = "kilometres driven" if st.session_state.km is None else "owner history"
            if st.session_state.km is not None and st.session_state.owner is not None: missing = ""
            if missing:
                reply(f"I picked up {st.session_state.bike} and the year {st.session_state.year}. I still need the {missing}. You can give both mileage and ownership together if convenient.")
                return

    # Interruptions: answer side questions or jump to edit a field without losing
    # the current valuation progress. These checks run before the normal step flow.
    if low in {"help", "what can you do", "commands", "options"}:
        reply("You can interrupt me anytime. Ask why I need a detail, how the estimate works, or type `change bike`, `change year`, `change kilometres`, `change owner`, `change seller`, or `change city`. I'll return to the valuation flow afterwards.")
        return
    if any(q in low for q in ["how does this work", "how do you calculate", "how is the price", "what is ai", "how are you ai", "explain the model"]):
        reply("I use a trained regression model built from historical bike listings. It estimates resale price from the bike model, year, kilometres driven, seller type and owner history. This is a machine-learning prediction, not a live quote; marketplace links are searches and do not automatically retrieve current listing prices.")
        return
    if any(q in low for q in ["why do you need", "why ask", "why are you asking"]):
        reply("Those details are inputs the model uses to estimate a price. You can ask another question or keep answering whenever you're ready; your current step is saved.")
        return

    # Allow editing an already collected detail from anywhere in the flow.
    edit_map = {
        "bike": ["change bike", "change model", "different bike", "wrong bike", "edit bike", "edit model"],
        "year": ["change year", "edit year", "wrong year"],
        "usage": ["change kilometres", "change kilometers", "change mileage", "edit mileage", "change km", "edit km"],
        "owner": ["change owner", "edit owner", "change ownership"],
        "seller": ["change seller", "edit seller", "change seller type"],
        "city": ["change city", "edit city", "change location", "edit location"],
    }
    for target, phrases in edit_map.items():
        if any(phrase in low for phrase in phrases):
            if target == "bike":
                st.session_state.step = "bike"
                st.session_state.bike = None
                st.session_state.year = None
                st.session_state.bike_options = []
                reply("Sure — let's change the bike. Type the model name and year if you know it.")
            elif target == "year":
                st.session_state.step = "year"
                st.session_state.year = None
                reply(f"What is the manufacturing year for {st.session_state.bike or 'your bike'}?")
            elif target == "usage":
                st.session_state.step = "usage"
                st.session_state.km = None
                reply("Enter the kilometres driven and owner history again, for example `25000 km, 2nd owner`.")
            elif target == "owner":
                st.session_state.step = "usage"
                st.session_state.owner = None
                reply("What is the ownership history? You can give the owner and kilometres together, e.g. `25000 km, 2nd owner`.")
            elif target == "seller":
                st.session_state.step = "market"
                st.session_state.seller = None
                reply("Are you selling as an individual or a dealer? You can answer that now and provide the city separately.")
            elif target == "city":
                st.session_state.step = "market"
                st.session_state.city = None
                reply("Which city should I use for the market search? You can provide just the city.")
            return

    # Offline conversational handling. No API key, network call, or external chatbot service.
    if low in ("reset","restart","start over","new valuation"):
        for k,v in defaults.items(): st.session_state[k]=v
        reply("Let’s start fresh, one question at a time. First, which company makes your motorcycle? For example, Honda, TVS, Hero, Bajaj, Yamaha or Royal Enfield.")
        return
    if low in {"hi", "hello", "hey", "hii", "yo", "good morning", "good afternoon", "good evening"}:
        reply("Hi! Let’s do this one step at a time. First, which company makes your motorcycle? For example, Honda, TVS, Hero, Bajaj, Yamaha or Royal Enfield.")
        return
    if any(q in low for q in ["how does this work", "how do you calculate", "how is the price", "what is ai", "explain the model", "how accurate", "is this accurate"]):
        reply("I use a trained regression model to estimate resale price from the bike model, year, kilometres, seller type and ownership history. It was evaluated on held-out data and its performance was modest, so treat the estimate as a starting point rather than a guaranteed sale price. Condition and service history are not fully represented. Marketplace links open searches; they don't fetch verified live listings automatically.")
        return
    if any(q in low for q in ["help", "what can you do", "commands", "options", "i don't understand", "i dont understand"]):
        reply("You can chat normally, ask questions, correct details, or give several details in one message. Try ‘change year’, ‘change kilometres’, ‘change owner’, ‘change city’, or simply tell me the corrected value. I'll keep the details already collected unless you change them.")
        return
    if any(q in low for q in ["why do you need", "why ask", "why are you asking"]):
        reply("Those details help the model estimate resale price. Share what you know in any wording; if I can't confidently identify a value, I'll ask a short follow-up instead of guessing.")
        return
    if step == "company":
        if low in {"hi", "hello", "hey", "start", "help"}:
            reply("First, which company makes your motorcycle? For example, Honda, TVS, Hero, Bajaj, Yamaha or Royal Enfield.")
            return
        company_aliases = COMPANY_ALIASES
        matches = []
        for brand, aliases in company_aliases.items():
            if any(re.search(r"(?<![a-z])" + re.escape(alias) + r"(?![a-z])", low) for alias in aliases):
                matches.append((max(map(len, aliases)), brand))
        if not matches:
            reply("I couldn't identify the company yet. Please tell me the manufacturer, such as Honda, TVS, Hero, Bajaj, Yamaha or Royal Enfield.")
            return
        st.session_state.company = sorted(matches, reverse=True)[0][1]
        st.session_state.step = "bike"
        reply(f"Got it — **{st.session_state.company}**. What is the model name? For example, Activa 3G, Apache RTR 160, Splendor Plus or Classic 350.")
        return

    if step=="bike":
        # Reject obvious accidental repeated backreference text or nonsensical pasted input.
        cleaned = re.sub(r"(?i)(?:\\\\1|\\\\\\\\1)+", "", t).strip()
        if len(cleaned) < 2 or len(re.findall(r"[a-zA-Z]", cleaned)) == 0:
            reply("I couldn't read that as a bike model. Please type the model name, for example `Activa 3g, 2018 model`.")
            return
        t = cleaned
        low = t.lower()
        year=find_year(t)
        if low.strip() in {"hi", "hello", "hey", "hii", "yo", "good morning", "good afternoon", "good evening", "start", "help"}:
            reply("First, which company makes your motorcycle? For example, Honda, TVS, Hero, Bajaj, Yamaha or Royal Enfield.")
            return
        # Remove the year and conversational filler before model matching.
        model_query = re.sub(r'\b(?:19|20)\d{2}\b', ' ', t)
        model_query = re.sub(r'\b(?:model|bike|motorcycle|my|is|a|an|the|year|manufactured|made|from)\b', ' ', model_query, flags=re.I)
        model_query = re.sub(r'[,;]+', ' ', model_query).strip()
        company_hint = (st.session_state.company + " " + model_query).strip() if st.session_state.company else model_query
        bike=find_bike(company_hint) or find_bike(model_query)

        if not bike:
            candidates=bike_candidates(company_hint or model_query or t)
            if not candidates and st.session_state.company:
                candidates=bike_candidates(model_query or t)
            # Require explicit confirmation when input is partial or multiple candidates are plausible.
            if candidates:
                st.session_state.bike_options=candidates
                st.session_state.pending_year=year
                st.session_state.step="choose_bike"
                reply("I found a few possible matches. Please choose the exact model below, or type a more specific model name. I won't pick one automatically.")
            else:
                reply("I need a little more of the model name to narrow it down. Type at least 3 meaningful characters or words, such as `Activa 3g` or `Himalayan`. I won't show unrelated guesses.")
            return

        st.session_state.bike=bike
        if year:
            if not 1988<=year<=2026: reply("Please use a manufacturing year between 1988 and 2026."); return
            st.session_state.year=year; st.session_state.step="usage"
            reply(f"Just to confirm: is **{bike}, {year}** the exact bike you mean? Type `yes` to confirm, or type another model name to search again.")
            st.session_state.step="confirm_bike"
        else:
            st.session_state.step="confirm_bike"
            reply(f"I matched **{bike}**. Is this the exact model you mean? Type `yes` to confirm, or type a different/partial model name to see similar matches.")
        return

    if step=="choose_bike":
        chosen = next((b for b in st.session_state.bike_options if b.lower() == t.lower()), None)
        if chosen is None:
            candidates = bike_candidates(t)
            # If user typed a number/index, allow selecting an option by number.
            m = re.fullmatch(r'\s*(\d+)\s*', t)
            if m and 1 <= int(m.group(1)) <= len(st.session_state.bike_options):
                chosen = st.session_state.bike_options[int(m.group(1))-1]
            elif candidates:
                st.session_state.bike_options = candidates
                reply("These are the closest matches to that wording. Type the exact model name or its number from the list.")
                return
            else:
                reply("I couldn't match that choice. Type one of the exact model names shown, or enter a more specific name.")
                return
        st.session_state.bike=chosen
        year=st.session_state.pending_year
        st.session_state.bike_options=[]
        if year:
            st.session_state.year=year
            st.session_state.step="confirm_bike"
            reply(f"You selected **{chosen}, {year}**. Is that correct? Type `yes` to continue or type another model name.")
        else:
            st.session_state.step="confirm_bike"
            reply(f"You selected **{chosen}**. Is that the correct model? Type `yes` to confirm or type another model name.")
        return

    if step=="confirm_bike":
        if low.strip() in {"yes","y","correct","that's right","thats right","confirm","yes that's it","yes thats it"}:
            if st.session_state.year is None:
                st.session_state.step="year"
                reply(f"Confirmed: **{st.session_state.bike}**. What year was it manufactured?")
            else:
                st.session_state.step="usage"
                reply(f"Confirmed: **{st.session_state.bike}, {st.session_state.year}**. How many kilometres has it covered, and how many previous owners? Example: `24000 km, 1st owner`.")
            return
        if low.strip() in {"no","n","not this","wrong","different model"}:
            st.session_state.bike=None
            st.session_state.year=None
            st.session_state.step="bike"
            reply(f"No problem. Type the {st.session_state.company or 'motorcycle'} model name again, and I'll help identify it.")
            return
        # Any other answer is treated as a new model search.
        st.session_state.step="bike"
        handle(t)
        return
    if step=="year":
        year=find_year(t)
        if not year or not 1988<=year<=2026: reply("Please enter a year between 1988 and 2026."); return
        st.session_state.year=year; st.session_state.step="usage"
        reply(f"Thanks. How many kilometres has your **{st.session_state.bike}** covered, and how many previous owners? Example: `24000 km, 1st owner`.")
        return
    if step=="usage":
        # Parse mileage robustly without deleting commas that separate mileage and owner.
        # Examples accepted: "10000, 2nd owner", "100,000 km and third owner",
        # "10k, second owner", "I rode 24000 kms, I'm the first owner".
        km = None
        km_k = re.search(r'(?<![a-z])([0-9]+(?:\.[0-9]+)?)\s*(k|thousand)\b', low)
        if km_k:
            km = int(float(km_k.group(1)) * 1000)
        else:
            # Prefer a number explicitly followed by a mileage unit.
            km_unit = re.search(r'([0-9][0-9,]*)\s*(?:km|kms|kilometres?|kilometers?)\b', low)
            if km_unit:
                km = int(km_unit.group(1).replace(",", ""))
            else:
                # Strip owner phrases first, then use the first remaining number as mileage.
                without_owner = re.sub(r'\b(?:1st|2nd|3rd|4th|first|second|third|fourth|one|two|three|four)\s*(?:owner|owners)?\b', ' ', low)
                without_owner = re.sub(r'\b(?:owner|owners)\b', ' ', without_owner)
                nums = re.findall(r'\d[\d,]*', without_owner)
                if nums:
                    try:
                        km = int(nums[0].replace(",", ""))
                    except ValueError:
                        km = None

        owner = None
        # Accept common variants such as "3 owners", "owner 3", and "I have three owners".
        if re.search(r"\b(?:3\s*owners?|owners?\s*3|three\s*owners?)\b", low):
            owner = "3rd owner"
        elif re.search(r"\b(?:2\s*owners?|owners?\s*2|two\s*owners?)\b", low):
            owner = "2nd owner"
        elif re.search(r"\b(?:4\s*owners?|owners?\s*4|four\s*owners?)\b", low):
            owner = "4th owner"
        elif re.search(r"\b(?:1\s*owner|owner\s*1|one\s*owner|single owner)\b", low):
            owner = "1st owner"

        owner_patterns = [
            ("4th owner", [r'\b4\s*(?:th|rd)?\s*owner\b', r'\bfourth owner\b', r'\bfour owners\b']),
            ("3rd owner", [r'\b3\s*rd\s*owner\b', r'\bthird owner\b', r'\b3rd\b', r'\bthird\b']),
            ("2nd owner", [r'\b2\s*nd\s*owner\b', r'\bsecond owner\b', r'\b2nd\b', r'\bsecond\b']),
            ("1st owner", [r'\b1\s*st\s*owner\b', r'\bfirst owner\b', r'\b1st\b', r'\bfirst\b', r'\bsingle owner\b', r'\bone owner\b']),
        ]
        if owner is None:
            for label, patterns in owner_patterns:
                if any(re.search(pattern, low) for pattern in patterns):
                    owner = label
                    break

        if km is None or km < 0 or km > 1000000 or owner is None:
            missing=[]
            if km is None or km > 1000000: missing.append("kilometres driven")
            if owner is None: missing.append("ownership (1st, 2nd, 3rd or 4th owner)")
            reply("I couldn't identify " + " and ".join(missing) + ". You can type it casually, for example: `10000, 2nd owner` or `100,000 km, third owner`."); return
        st.session_state.km=km; st.session_state.owner=owner; st.session_state.step="market"
        reply("Thanks. Are you selling as an individual or dealer, and which city? Example: `Individual, Hyderabad`."); return
    if step=="market":
        seller = None
        if re.search(r"\bdealer\b", low):
            seller = "Dealer"
        elif re.search(r"\b(individual|private|privately|myself|personal seller)\b", low):
            seller = "Individual"

        city_aliases = {
            "Hyderabad": ["hyderabad", "hyd", "hyderbad", "hydrabad"],
            "Bengaluru": ["bengaluru", "bangalore", "blr"],
            "Chennai": ["chennai", "madras"],
            "Mumbai": ["mumbai", "bombay"],
            "Pune": ["pune"],
            "Delhi": ["delhi", "new delhi", "ncr"],
            "Kolkata": ["kolkata", "calcutta"],
            "Ahmedabad": ["ahmedabad"],
            "Jaipur": ["jaipur"],
            "Lucknow": ["lucknow"],
            "India-wide": ["india-wide", "india wide", "all india", "pan india", "india"],
        }
        city = None
        for canonical, aliases in city_aliases.items():
            if any(re.search(r"(?<![a-z])" + re.escape(alias) + r"(?![a-z])", low) for alias in aliases):
                city = canonical
                break

        # Save any new detail before asking for the missing one.
        if seller:
            st.session_state.seller = seller
        if city:
            st.session_state.city = city
        seller = st.session_state.seller
        city = st.session_state.city

        if not seller and not city:
            reply("Are you selling as an individual or a dealer, and which city are you in? You can send these in separate messages.")
            return
        if not seller:
            reply(f"Got it — {city}. Are you selling as an individual or a dealer?")
            return
        if not city:
            reply(f"Got it — {seller} seller. Which city should I use for the market search?")
            return

        try:
            row = pd.DataFrame([{
                "name": st.session_state.bike,
                "year": st.session_state.year,
                "seller_type": seller,
                "owner": st.session_state.owner,
                "km_driven": st.session_state.km,
            }])
            prediction = max(0.0, float(model.predict(row)[0]))
            st.session_state.prediction = prediction
            st.session_state.step = "results"
            reply(f"Valuation complete. The trained machine-learning model estimates {rupees(prediction)}. Your results and marketplace searches are below.")
        except Exception as e:
            reply(f"Prediction failed: {e}. Check that the model file matches the training features.")
        return
    p=st.session_state.prediction
    if any(x in low for x in ["why","price","worth","estimate","lower","value"]):
        reply(f"The model estimates **{rupees(p)}** from model, year, kilometres, seller type and owner history. Its test performance was modest, so treat it as a starting point; condition, service history and local demand may not be fully captured.")
    elif any(x in low for x in ["olx","facebook","highest","compare","listing","marketplace","link"]):
        reply("The links below open searches; this app does not automatically retrieve live prices yet. Compare exact model, year, mileage, condition and location. Paste listing details here if you'd like help comparing them.")
    else: reply("I can explain the estimate, help compare listings you paste here, or update any detail. I work best when you tell me what you mean naturally; if something is unclear, I'll ask a follow-up rather than pretend I understood.")

# Simple guided questionnaire: no chatbot or free-text conversation.
st.markdown('<div class="intro">Answer each question using the options. Your choices are saved as you go.</div>', unsafe_allow_html=True)

# Progress indicator
steps = ["company", "bike", "year", "km", "owner", "seller", "city", "results"]
step_titles = {"company":"Manufacturer", "bike":"Model", "year":"Year", "km":"Kilometres", "owner":"Ownership", "seller":"Seller type", "city":"City", "results":"Your estimate"}
current_index = steps.index(st.session_state.step) if st.session_state.step in steps else 0
st.progress(current_index / (len(steps)-1))
st.caption(f"Step {min(current_index+1, len(steps))} of {len(steps)} · {step_titles.get(st.session_state.step, 'Your estimate')}")

if st.session_state.step == "company":
    st.markdown("### Which company makes your motorcycle?")
    st.caption("Choose the manufacturer first. We'll show its available models next.")
    company_choices = list(COMPANY_ALIASES.keys())
    company_cols = st.columns(3)
    for idx, company_name in enumerate(company_choices):
        with company_cols[idx % 3]:
            if st.button(company_name, key=f"company_choice_{company_name}", use_container_width=True):
                st.session_state.company = company_name
                st.session_state.step = "bike"
                st.rerun()

elif st.session_state.step == "bike":
    st.markdown(f"### Which {st.session_state.company} model do you have?")
    available_models = models_for_company(st.session_state.company)
    if available_models:
        selected_model = st.selectbox("Motorcycle model", available_models, key=f"model_select_{st.session_state.company}", index=None, placeholder="Select a model…")
        if st.button("Continue", key="continue_model", use_container_width=True, disabled=not selected_model):
            st.session_state.bike = selected_model
            st.session_state.step = "year"
            st.rerun()
    else:
        st.warning("No matching models were found for this manufacturer in the dataset.")
    if st.button("Back to manufacturer", key="back_company"):
        st.session_state.step = "company"
        st.rerun()

elif st.session_state.step == "year":
    st.markdown(f"### What year was your {st.session_state.bike} manufactured?")
    selected_year = st.selectbox("Manufacturing year", list(range(2026, 1987, -1)), index=None, placeholder="Choose year…")
    if st.button("Continue", key="continue_year", use_container_width=True, disabled=selected_year is None):
        st.session_state.year = int(selected_year)
        st.session_state.step = "km"
        st.rerun()

elif st.session_state.step == "km":
    st.markdown("### How many kilometres has it covered?")
    st.caption("Enter the total odometer reading, e.g. 25,000 km.")
    selected_km = st.number_input("Kilometres driven", min_value=0, max_value=1000000, value=10000, step=1000)
    if st.button("Continue", key="continue_km", use_container_width=True):
        st.session_state.km = int(selected_km)
        st.session_state.step = "owner"
        st.rerun()

elif st.session_state.step == "owner":
    st.markdown("### How many owners has the bike had?")
    owner_choice = st.radio("Ownership history", ["1st owner", "2nd owner", "3rd owner", "4th owner"], horizontal=True, label_visibility="collapsed")
    if st.button("Continue", key="continue_owner", use_container_width=True):
        st.session_state.owner = owner_choice
        st.session_state.step = "seller"
        st.rerun()

elif st.session_state.step == "seller":
    st.markdown("### Who is selling the motorcycle?")
    seller_choice = st.radio("Seller type", ["Individual", "Dealer"], horizontal=True, label_visibility="collapsed")
    if st.button("Continue", key="continue_seller", use_container_width=True):
        st.session_state.seller = seller_choice
        st.session_state.step = "city"
        st.rerun()

elif st.session_state.step == "city":
    st.markdown("### Which city is the bike being sold in?")
    city_choice = st.selectbox("City", ["Hyderabad", "Bengaluru", "Chennai", "Mumbai", "Pune", "Delhi", "Kolkata", "Ahmedabad", "Jaipur", "Lucknow", "India-wide"], index=None, placeholder="Choose a city…")
    if st.button("Estimate resale price", key="estimate_price", use_container_width=True, disabled=not city_choice):
        st.session_state.city = city_choice
        try:
            row = pd.DataFrame([{"name": st.session_state.bike, "year": st.session_state.year,
                "seller_type": st.session_state.seller, "owner": st.session_state.owner,
                "km_driven": st.session_state.km}])
            st.session_state.prediction = max(0.0, float(model.predict(row)[0]))
            st.session_state.step = "results"
        except Exception as e:
            st.error(f"Prediction failed: {e}. Check that the model file matches the training features.")
        st.rerun()

if st.session_state.step not in ("company", "results"):
    if st.button("Back", key="back_one_step"):
        back_map = {"bike":"company", "year":"bike", "km":"year", "owner":"km", "seller":"owner", "city":"seller"}
        st.session_state.step = back_map.get(st.session_state.step, "company")
        st.rerun()

if st.session_state.step == "results":
    p = st.session_state.prediction
    st.markdown(f'<div class="result"><div class="label">MACHINE-LEARNING RESALE ESTIMATE</div><div class="price">{rupees(p)}</div><div class="note">{html.escape(str(st.session_state.bike))} · {st.session_state.year} · {st.session_state.km:,} km · {html.escape(str(st.session_state.owner))}</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="panel"><div class="paneltitle">Indicative asking-price range</div><div style="font-family:Manrope,sans-serif;font-size:23px;font-weight:800;margin-top:6px">{rupees(max(0,p*.9))} – {rupees(p*1.1)}</div><div class="small">Planning range around the model estimate, not a validated confidence interval.</div></div>', unsafe_allow_html=True)
    st.markdown("#### Compare marketplace searches")
    st.markdown('<div class="warning">These links open searches; they do not provide automatically verified live listing prices.</div>', unsafe_allow_html=True)
    for name, url in market_links(st.session_state.bike, st.session_state.city):
        st.markdown(f'<div class="panel"><div class="paneltitle">{html.escape(name)}</div><div class="small">Search {html.escape(str(st.session_state.bike))} around {html.escape(str(st.session_state.city))}.</div><div style="margin-top:7px"><a href="{html.escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">Open search {icon("external", "#b45309", 13)}</a></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="green">The highest asking price is not necessarily the most realistic selling price. Check model, year, kilometres and condition.</div>', unsafe_allow_html=True)
    if st.button("Start a new valuation", use_container_width=True):
        for k, v in defaults.items():
            st.session_state[k] = v.copy() if isinstance(v, list) else v
        st.rerun()

st.markdown('<div class="foot">BIKE RESALE AI · ACADEMIC AIML PROJECT · MULTIPLE LINEAR REGRESSION</div>',unsafe_allow_html=True)
