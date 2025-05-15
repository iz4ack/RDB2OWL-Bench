CREATE TABLE Territorio (
    id SERIAL PRIMARY KEY,
    nombre VARCHAR(100) NOT NULL,
    tipo VARCHAR(50) NOT NULL CHECK(tipo IN ('pais', 'region', 'provincia', 'localidad'))
);

CREATE TABLE ParteDe (
    id_padre INTEGER NOT NULL REFERENCES Territorio(id) ON DELETE CASCADE,
    id_hijo INTEGER NOT NULL REFERENCES Territorio(id) ON DELETE CASCADE,
    PRIMARY KEY (id_padre, id_hijo),
    CONSTRAINT no_autoreferencia CHECK (id_padre <> id_hijo)
);


CREATE TABLE LimitaCon (
    territorio1_id INTEGER PRIMARY KEY REFERENCES Territorio(id) ON DELETE CASCADE,
    territorio2_id INTEGER UNIQUE NOT NULL REFERENCES Territorio(id) ON DELETE CASCADE,
    CONSTRAINT no_autolimite CHECK (territorio1_id <> territorio2_id)
);



