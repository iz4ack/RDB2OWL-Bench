CREATE TABLE Territory (
    id SERIAL PRIMARY KEY,
    territory_name VARCHAR(100) ,
    territory_type VARCHAR(50) CHECK(type IN ('country', 'region', 'province', 'locality'))
);

CREATE TABLE PartOf (
    parent_id INTEGER NOT NULL REFERENCES Territory(id) ON DELETE CASCADE,
    child_id INTEGER NOT NULL REFERENCES Territory(id) ON DELETE CASCADE,
    PRIMARY KEY (parent_id, child_id),
    CONSTRAINT no_self_reference CHECK (parent_id <> child_id)
);

CREATE TABLE BordersWith (
    territory1_id INTEGER NOT NULL REFERENCES Territory(id) ON DELETE CASCADE,
    territory2_id INTEGER NOT NULL REFERENCES Territory(id) ON DELETE CASCADE,
    PRIMARY KEY (territory1_id, territory2_id),
    CONSTRAINT no_self_border CHECK (territory1_id <> territory2_id)
);
