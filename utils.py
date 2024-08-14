import re, math, numpy as np, json
from sklearn.metrics import log_loss
from collections import Counter 

get_json = lambda x: re.match('(.|\n)*(\{(?:[^{}]|\{.*?\})*\})(.|\n)*', x)
strip_json = lambda x:   json.loads(x.strip('```json\n').strip('```'))

country_to_nation = {
    "Albania":  "Albanian", 
    "Andorra":  "Andorran", 
    "Angola":  "Angolan", 
    "Angola (Non-national sample)":  "Angolan", 
    "Argentina":  "Argentinian", 
    "Armenia":  "Armenian", 
    "Australia":  "Australian", 
    "Austria":  "Austrian", 
    "Azerbaijan":  "Azerbaijani", 
    "Bangladesh":  "Bangladeshi", 
    "Bangladesh (Non-national sample)":  "Bangladeshi", 
    "Belarus":  "Belarusian", 
    "Belgium":  "Belgian", 
    "Bolivia":  "Bolivian", 
    "Bolivia (Non-national sample)":  "Bolivian", 
    "Bosnia Herzegovina":  "Bosnian Herzegovinian", 
    "Brazil":  "Brazilian", 
    "Brazil (Non-national sample)":  "Brazilian", 
    "Britain":  "British", 
    "Bulgaria":  "Bulgarian", 
    "Burkina Faso":  "Burkinabe", 
    "Canada":  "Canadian", 
    "Chile":  "Chilean", 
    "China":  "Chinese", 
    "China (Non-national sample)":  "Chinese", 
    "Colombia":  "Colombian", 
    "Colombia (Non-national sample)":  "Colombian", 
    "Croatia":  "Croatian", 
    "Cyprus":  "Cypriot", 
    "Czech Rep.":  "Czech", 
    "Czechia":  "Czech", 
    "Denmark":  "Danish", 
    "Ecuador":  "Ecuadorean", 
    "Egypt":  "Egyptian", 
    "Egypt (Non-national sample)":  "Egyptian", 
    "El Salvador":  "Salvadoran", 
    "Estonia":  "Estonian", 
    "Ethiopia":  "Ethiopian", 
    "Ethiopia (Non-national sample)":  "Ethiopian", 
    "Finland":  "Finnish", 
    "France":  "French", 
    "Georgia":  "Georgian", 
    "Germany":  "German", 
    "Ghana":  "Ghanaifan", 
    "Great Britain":  "British", 
    "Greece":  "Greek", 
    "Guatemala":  "Guatemalan", 
    "Guatemala (Non-national sample)":  "Guatemalan", 
    "Honduras":  "Honduran", 
    "Honduras (Non-national sample)":  "Honduran", 
    "Hong Kong SAR":  "Hong Konger", 
    "Hungary":  "Hungarian", 
    "Iceland":  "Icelander", 
    "India (Current national sample)":  "Indian", 
    "India (Non-national sample)":  "Indian", 
    "India (Old national sample)":  "Indian", 
    "Indonesia":  "Indonesian", 
    "Indonesia (Non-national sample)":  "Indonesian", 
    "Iran":  "Iranian", 
    "Iraq":  "Iraqi", 
    "Israel":  "Israeli", 
    "Italy":  "Italian", 
    "Ivory Coast":  "Ivorian", 
    "Ivory Coast (Non-national sample)":  "Ivorian", 
    "Japan":  "Japanese", 
    "Jordan":  "Jordanian", 
    "Jordan (Non-national sample)":  "Jordanian", 
    "Kazakhstan":  "Kazakhstani", 
    "Kenya":  "Kenyan", 
    "Kuwait":  "Kuwaiti", 
    "Kyrgyzstan":  "Kyrgyzstani", 
    "Latvia":  "Latvian", 
    "Lebanon":  "Lebanese", 
    "Libya":  "Libyan", 
    "Lithuania":  "Lithuanian", 
    "Macau SAR":  "Macanese", 
    "Malaysia":  "Malaysian", 
    "Maldives":  "Maldivian", 
    "Mali":  "Malian", 
    "Mali (Non-national sample)":  "Malian", 
    "Mexico":  "Mexican", 
    "Mongolia":  "Mongolian", 
    "Montenegro":  "Montenegrin", 
    "Morocco":  "Moroccan", 
    "Morocco (Non-national sample)":  "Moroccan", 
    "Myanmar":  "Myanmar", 
    "Netherlands":  "Dutch", 
    "New Zealand":  "New Zealander", 
    "Nicaragua":  "Nicaraguan", 
    "Nigeria":  "Nigerian", 
    "Nigeria (Non-national sample)":  "Nigerian", 
    "North Macedonia":  "Macedonian", 
    "Northern Ireland":  "Northern Irish", 
    "Norway":  "Norwegian", 
    "Pakistan":  "Pakistani", 
    "Pakistan (Non-national sample)":  "Pakistani", 
    "Palest. ter.":  "Palestinian", 
    "Peru":  "Peruvian", 
    "Philippines":  "Filipino", 
    "Philippines (Non-national sample)":  "Filipino", 
    "Poland":  "Polish", 
    "Poland (Non-national sample)":  "Polish", 
    "Portugal":  "Portuguese", 
    "Puerto Rico":  "Puerto Rican", 
    "Romania":  "Romanian", 
    "Russia":  "Russian", 
    "Russia (Non-national sample)":  "Russian", 
    "S. Africa":  "South African", 
    "S. Africa (Non-national sample)":  "South African", 
    "S. Korea":  "South Korean", 
    "Senegal":  "Senegalese", 
    "Senegal (Non-national sample)":  "Senegalese", 
    "Serbia":  "Serbian", 
    "Singapore":  "Singaporean", 
    "Slovakia":  "Slovak", 
    "Slovenia":  "Slovenian", 
    "South Korea":  "South Korean", 
    "Spain":  "Spanish", 
    "Sweden":  "Swedish", 
    "Switzerland":  "Swiss", 
    "Taiwan":  "Taiwanese", 
    "Taiwan ROC":  "Taiwanese", 
    "Tajikistan":  "Tajikistani", 
    "Tanzania":  "Tanzanian", 
    "Tanzania (Non-national sample)":  "Tanzanian", 
    "Thailand":  "Thai", 
    "Tunisia":  "Tunisian", 
    "Turkey":  "Turkish", 
    "Uganda":  "Ugandan", 
    "Ukraine":  "Ukrainian", 
    "United States":  "American", 
    "Uruguay":  "Uruguayan", 
    "Uzbekistan":  "Uzbekistani", 
    "Venezuela":  "Venezuelan", 
    "Venezuela (Non-national sample)":  "Venezuelan", 
    "Vietnam":  "Vietnamese", 
    "Vietnam (Non-national sample)":  "Vietnamese", 
    "Zimbabwe":  "Zimbabwean"
}

def pred_extract(reply, chars):
    match_response = get_json(reply)
    if match_response:
        json_string = match_response.group(2)
        try: 
            response = strip_json(json_string)
            if 'opinion' in response:
                if response['opinion'] in chars:
                    return response['opinion']
        except Exception as e: 
            print(e)
    return 

def cross_entropy(y_true, opinion):
    y_true = np.array(y_true)
    y_pred = np.zeros_like(y_true)
    y_pred[ord(opinion)-ord('A')] = 1
    return log_loss(y_true=y_pred, y_pred=y_true) 


def validate_agents_count(example, count=2):
    return example['question'] is not None  and len(eval(example['selections'][28:-1])) > count


def get_entropy(values: list):
    counter = Counter(values)
    h = 0 
    for v in counter.values():
        p = v/sum(counter.values())
        h += -p*math.log(p, 2)
    return h
