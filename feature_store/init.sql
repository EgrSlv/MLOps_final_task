CREATE SCHEMA IF NOT EXISTS feature_store;

CREATE TABLE IF NOT EXISTS feature_store.passenger_features (
    id SERIAL PRIMARY KEY,
    passenger_id VARCHAR(50) UNIQUE NOT NULL,
    pclass INT,
    sex VARCHAR(10),
    age FLOAT,
    sibsp INT,
    parch INT,
    fare FLOAT,
    embarked VARCHAR(1),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS feature_store.predictions (
    id SERIAL PRIMARY KEY,
    passenger_id VARCHAR(50) REFERENCES feature_store.passenger_features(passenger_id),
    prediction INT,
    survival_probability FLOAT,
    model_version VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_passenger_id ON feature_store.passenger_features(passenger_id);
CREATE INDEX idx_created_at ON feature_store.predictions(created_at);
