CREATE TABLE Person (
  NIF VARCHAR(9) PRIMARY KEY,
  email VARCHAR(255) UNIQUE,
  first_name VARCHAR(100),
  roleType VARCHAR(20) CHECK(type IN 'Employee', 'Consultant'),
  employee_salary DECIMAL,
  consultant_hourly_rate DECIMAL,
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
  company_name VARCHAR(100), 
  location_id INT NOT NULL UNIQUE,
  FOREIGN KEY (location_id) REFERENCES Locationn(id)
);

CREATE TABLE Locationn (
  id INT PRIMARY KEY,
  street VARCHAR(255),
  city VARCHAR(100),
  province VARCHAR(100),
  coordinates VARCHAR(100)
);

CREATE TABLE WorksWith (
  person_id INT NOT NULL,
  company_id VARCHAR(9) NOT NULL,
  PRIMARY KEY (person_id, company_id),
  FOREIGN KEY (person_id) REFERENCES Person(NIF),
  FOREIGN KEY (company_id) REFERENCES Company(NIF)
);

