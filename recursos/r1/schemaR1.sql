CREATE TABLE Departamentos (
    ID_Departamento    INT          NOT NULL AUTO_INCREMENT,
    Nombre             VARCHAR(100) NOT NULL,
    PRIMARY KEY (ID_Departamento)
);

CREATE TABLE Estudiantes (
    ID_Estudiante      INT           NOT NULL AUTO_INCREMENT,
    Nombre             VARCHAR(100)  NOT NULL,
    Apellido           VARCHAR(100)  NOT NULL,
    Email              VARCHAR(150)  NOT NULL,
    FechaNacimiento    DATE          NOT NULL,
    PRIMARY KEY (ID_Estudiante),
);

CREATE TABLE Profesores (
    ID_Profesor        INT           NOT NULL AUTO_INCREMENT,
    Nombre             VARCHAR(100)  NOT NULL,
    Apellido           VARCHAR(100)  NOT NULL,
    Email              VARCHAR(150)  NOT NULL,
    ID_Departamento    INT           NOT NULL,
    PRIMARY KEY (ID_Profesor),
    FOREIGN KEY (ID_Departamento) REFERENCES Departamentos(ID_Departamento)
);

CREATE TABLE Cursos (
    ID_Curso           INT           NOT NULL AUTO_INCREMENT,
    Nombre             VARCHAR(100)  NOT NULL,
    Nivel              VARCHAR(50)   NOT NULL,
    Creditos           INT           NOT NULL,
    ID_Departamento    INT           NOT NULL,
    PRIMARY KEY (ID_Curso),
    FOREIGN KEY (ID_Departamento) REFERENCES Departamentos(ID_Departamento)
);

CREATE TABLE Inscripciones (
    ID_Estudiante      INT           NOT NULL,
    ID_Curso           INT           NOT NULL,
    FechaInscripcion   DATE          NOT NULL,
    PRIMARY KEY (ID_Estudiante, ID_Curso),
    FOREIGN KEY (ID_Estudiante) REFERENCES Estudiantes(ID_Estudiante),
    FOREIGN KEY (ID_Curso)      REFERENCES Cursos(ID_Curso)
);

CREATE TABLE Evaluaciones (
    ID_Evaluacion      INT           NOT NULL AUTO_INCREMENT,
    ID_Estudiante      INT           NOT NULL,
    ID_Curso           INT           NOT NULL,
    Nota               DECIMAL(4,2)  NOT NULL,
    FechaEvaluacion    DATE          NOT NULL,
    PRIMARY KEY (ID_Evaluacion),
    FOREIGN KEY (ID_Estudiante) REFERENCES Estudiantes(ID_Estudiante),
    FOREIGN KEY (ID_Curso)      REFERENCES Cursos(ID_Curso)
);