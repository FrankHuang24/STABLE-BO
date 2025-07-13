import pickle

# Global minimum values for each name
GLOBAL_MINIMUM = {
    'branin2': 0.397887,
    'hartmann3': -3.86278,
    'hartmann6': -3.32237,
    'griewank5': 0,
    'levy8': 0,
    'levy10': 0,
    'levy30':0,
    'exp5': 0,
    'exp10': 0,
    'rosen14': 0,
    'rosen20': 0,
    'robot3': 0,
    'robot4': 0,
    'portfolio5': 30,
    'SVM3': 1,
    'XGBoost9': 1,
    'LightGBM16': 1,


}

REAL_WORLD = ['robot3', 'robot4', 'portfolio5', 'SVM3', 'XGBoost9', 'LightGBM16']
NEGATE = ['portfolio5', 'SVM3', 'XGBoost9', 'XGBoost14', 'LightGBM16']

BAR_DICT = {
    'branin2': {
        "errorevery": 1,
        "beta": 0.1
    },
    'hartmann3': {
        "errorevery": 1,
        "beta": 0.1
    },
    'hartmann6': {
        "errorevery": 5,
        "beta": 0.1
    },
    'griewank5': {
        "errorevery": 5,
        "beta": 0.5
    },
    'exp5': {
        "errorevery": 5,
        "beta": 0.1
    },
    'exp10': {
        "errorevery": 10,
        "beta": 0.1
    },
    'levy8': {
        "errorevery": 5,
        "beta": 1
    },
    'levy10': {
        "errorevery": 5,
        "beta": 1
    },
    'levy30': {
        "errorevery": 20,
        "beta": 0.1
    },
    'rosen14': {
        "errorevery": 10,
        "beta": 0.1
    },
    'rosen20': {
        "errorevery": 20,
        "beta": 0.1
    },
    'robot3': {
        "errorevery": 5,
        "beta": 0.1
    },
    'robot4': {
        "errorevery": 10,
        "beta": 0.1
    },
    'portfolio5': {
        "errorevery": 10,
        "beta": 0.1
    },
    'SVM3': {
        "errorevery": 5,
        "beta": 0.1
    },
    'XGBoost9': {
        "errorevery": 10,
        "beta": 0.1
    },


}

PARAMS_DICT = {
    "rbf": {
        "label": "RBF",
        "marker": "+",
        "linestyle": "dotted",
        "color": u"#1f77b4",
    },
    "rq": {
        "label": "RQ",
        "marker": "+",
        "linestyle": "dotted",
        "color": u"#8D38C9",
    },
    "matern": {
        "label": "MA52",
        "marker": "+",
        "linestyle": "dotted",
        "color": u"#4EE2EC",
    },
    "ABO": {
        "label": "ABO",
        "marker": "x",
        "linestyle": "dashed",
        "color": u"#348017",
    },
    "ada": {
        "label": "ADA",
        "marker": "x",
        "linestyle": "dashed",
        "color": u"#99C68E",
    },
    "sdk": {
        "label": "SDK",
        "marker": "x",
        "linestyle": "dashed",
        "color": u"#FFA62F",
    },
    "sinc": {
        "label": "SINC",
        "marker": "x",
        "linestyle": "dashed",
        "color": u"#FFFF00",
    },
    "c6g1": {
        "label": "CSM+GSM",
        "marker": "p",
        "linestyle": (0, (5, 10)),
        "color": u"#d62728",
    },
    "csm7": {
        "label": "CSM7",
        "marker": "p",
        "linestyle": (0, (5, 10)),
        "color": u"#E77471",
    },
    "gsm7": {
        "label": "GSM7",
        "marker": "p",
        "linestyle": (0, (5, 10)),
        "color": u"#e377c2",
    },
    "csm3": {
        "label": "CSM3",
        "marker": "+",
        "linestyle": "dotted",
        "color": u"#1f77b4",
    },
    "csm5": {
        "label": "CSM5",
        "marker": "x",
        "linestyle": "dashed",
        "color": u"#FF8040",
    },
    "csm9": {
        "label": "CSM9",
        "marker": "s",
        "color": u"#2ca02c",
    },
    "gsm3": {
        "label": "GSM3",
        "marker": "+",
        "linestyle": "dotted",
        "color": u"#d62728",
    },
    "gsm5": {
        "label": "GSM5",
        "marker": "x",
        "linestyle": "dashed",
        "color": u"#9467bd",
    },
    "gsm9": {
        "label": "GSM9",
        "marker": "s",
        "color": u"#7f7f7f",
    },
}


# Function to load pickle file and extract results
def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


# Function to parse filename into components
def parse_filename(filename):
    parts = filename.split('_')
    name = parts[0]
    kernel = parts[1]
    acq = parts[2].split('.')[0]  # Remove ".pkl"
    return name, kernel, acq
