from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from cleaner.cleaner import Cleaner
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, ShuffleSplit
import time
from sklearn.metrics import mean_absolute_error

df = Cleaner.clean_data()

numerical_features = ["bedroomCount", "habitableSurface", "facedeCount", "streetFacadeWidth", "kitchenSurface", "landSurface", 
                    "terraceSurface", "gardenSurface", "toiletCount", "bathroomCount"]
categorical_features = [
    "type",
    "subtype",
    "postCode",
    "hasBasement",
    "buildingCondition",
    "buildingConstructionYear",
    "hasTerrace",
    "floodZoneType",
    "heatingType",
    "kitchenType",
    "gardenOrientation",
    "hasSwimmingPool",
    "terraceOrientation",
    "epcScore"
]

target_name = "price"

imputer = SimpleImputer(strategy = 'most_frequent', add_indicator=True)

n_unique_categories = df[categorical_features].nunique().sort_values(ascending=False)

high_cardinality_features = n_unique_categories[n_unique_categories > 255].index
low_cardinality_features = n_unique_categories[n_unique_categories <= 255].index

low_card_preprocessor = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-2)
high_card_preprocessor = TargetEncoder(target_type="continuous")

mixed_encoded_preprocessor = make_column_transformer(
    (imputer, numerical_features),
    (high_card_preprocessor, high_cardinality_features),
    (low_card_preprocessor, low_cardinality_features),
    verbose_feature_names_out = False
    )
mixed_encoded_preprocessor.set_output(transform = "pandas")

mixed_pipe = make_pipeline(
mixed_encoded_preprocessor,
    HistGradientBoostingRegressor(
        random_state = 0, 
        max_iter = 1000, 
        early_stopping=True, 
        categorical_features = low_cardinality_features,
        learning_rate=0.01, 
        max_depth=254, 
        max_features=0.25, 
        max_leaf_nodes=100, 
        min_samples_leaf=50, 
        warm_start=True
    ),
    verbose = True
)

X = df[numerical_features + categorical_features]
Y = df[target_name]

start_time = time.time()
mixed_pipe.fit(X, Y)
print('Training took', time.time() - start_time, 'seconds')

app = FastAPI()

class request_body(BaseModel):
    bedroomCount: int
    habitableSurface: int
    facedeCount: int
    streetFacadeWidth: float
    kitchenSurface: int
    landSurface: int
    terraceSurface: int
    gardenSurface: int
    toiletCount: int
    bathroomCount: int
    type: str
    subtype: str
    postCode: int
    hasBasement: int
    buildingCondition: str
    buildingConstructionYear: int
    hasTerrace: int
    floodZoneType: str
    heatingType: str
    kitchenType: str
    gardenOrientation: str
    hasSwimmingPool: int
    terraceOrientation: str
    epcScore: str

# Creating an Endpoint to receive the data
# to make prediction on.
@app.post('/predict')
def predict(data : request_body):
    # Making the data in a form suitable for prediction
    test_data = [[
        data.bedroomCount,
        data.habitableSurface,
        data.facedeCount,
        data.streetFacadeWidth,
        data.kitchenSurface,
        data.landSurface,
        data.terraceSurface,
        data.gardenSurface,
        data.toiletCount,
        data.bathroomCount,
        data.type,
        data.subtype,
        data.postCode,
        data.hasBasement,
        data.buildingCondition,
        data.buildingConstructionYear,
        data.hasTerrace,
        data.floodZoneType,
        data.heatingType,
        data.kitchenType,
        data.gardenOrientation,
        data.hasSwimmingPool,
        data.terraceOrientation,
        data.epcScore
    ]]

    prediction = mixed_pipe.predict(test_data)
    return { 'price' : prediction[0] }
