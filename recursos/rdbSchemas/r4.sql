CREATE TABLE Person (
  NIF INT PRIMARY KEY,
  email VARCHAR(255) UNIQUE,
  first_name VARCHAR(100)
);

CREATE TABLE Employee (
  id INT PRIMARY KEY,
  salary DECIMAL
);

CREATE TABLE Consultant (
  id INT PRIMARY KEY,
  hourly_rate DECIMAL
);

CREATE TABLE Mentors (
  mentor_id INT,
  mentee_id INT,
  PRIMARY KEY (mentor_id, mentee_id),
  FOREIGN KEY (mentor_id) REFERENCES Person(NIF),
  FOREIGN KEY (mentee_id) REFERENCES Person(NIF)
);

CREATE TABLE Company (
  NIF VARCHAR(9) PRIMARY KEY,
  company_name VARCHAR(100)
);

CREATE TABLE Locationn (
  id INT PRIMARY KEY,
  street VARCHAR(255),
  city VARCHAR(100),
  province VARCHAR(100),
  coordinates VARCHAR(100)
);

CREATE TABLE WorksWith (
  person_id INT,
  company_id VARCHAR(9),
  PRIMARY KEY (person_id, company_id),
  FOREIGN KEY (person_id) REFERENCES Person(NIF),
  FOREIGN KEY (company_id) REFERENCES Company(NIF)
);

CREATE TABLE HasLocation (
  company_id VARCHAR(9),
  location_id INT,
  PRIMARY KEY (company_id, location_id),
  FOREIGN KEY (company_id) REFERENCES Company(NIF),
  FOREIGN KEY (location_id) REFERENCES Location(id)
);
