CREATE TABLE Persona (
  NIF INT PRIMARY KEY,
  email VARCHAR(255) UNIQUE,
  nombre VARCHAR(100)
);

CREATE TABLE Empleado (
  id INT PRIMARY KEY,
  salario DECIMAL
);

CREATE TABLE Consultor (
  id INT PRIMARY KEY,
  tarifaHora DECIMAL
);

CREATE TABLE Mentorea (
  mentor_id INT,
  mentee_id INT,
  PRIMARY KEY (mentor_id, mentee_id),
  FOREIGN KEY (mentor_id) REFERENCES Persona(id),
  FOREIGN KEY (mentee_id) REFERENCES Persona(id)
);

CREATE TABLE Empresa (
  NIF VARCHAR(9) PRIMARY KEY,
  nombre VARCHAR(100)
);

CREATE TABLE Ubicacion (
  id INT PRIMARY KEY,
  calle VARCHAR(255),
  ciudad VARCHAR(100),
  provincia VARCHAR(100),
  coordenadas VARCHAR(100)
);

CREATE TABLE TrabajaCon (
  persona_id INT,
  empresa_id INT,
  PRIMARY KEY (persona_id, empresa_id)
);

CREATE TABLE TieneUbicacion (
  empresa_id INT,
  ubicacion_id INT,
  PRIMARY KEY (empresa_id, ubicacion_id)
);

