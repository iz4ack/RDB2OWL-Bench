CREATE TABLE Departamentos (
    ID_Departamento    INT          NOT NULL AUTO_INCREMENT,
    Nombre             VARCHAR(100) ,
    PRIMARY KEY (ID_Departamento)
);

CREATE TABLE Estudiantes (
    ID_Estudiante      INT           NOT NULL AUTO_INCREMENT,
    DNI                VARCHAR(20)   ,
    Nombre             VARCHAR(100)  ,
    Apellido           VARCHAR(100)  ,
    Email              VARCHAR(150)  ,
    FechaNacimiento    DATE          ,
    FechaMatricula     DATE          ,
    PRIMARY KEY (ID_Estudiante),
    CONSTRAINT UC_Estudiante UNIQUE (DNI)
);

CREATE TABLE Profesores (
    ID_Profesor        INT           NOT NULL AUTO_INCREMENT,
    Nombre             VARCHAR(100)  ,
    Apellido           VARCHAR(100)  ,
    Email              VARCHAR(150)  ,
    FechaNacimiento    DATE          ,
    ID_Departamento    INT           NOT NULL,
    PRIMARY KEY (ID_Profesor),
    FOREIGN KEY (ID_Departamento) REFERENCES Departamentos(ID_Departamento),
    CONSTRAINT UC_Profesor UNIQUE (DNI)
);

CREATE TABLE Cursos (
    ID_Curso           INT           NOT NULL AUTO_INCREMENT,
    Nombre             VARCHAR(100)  ,
    Nivel              VARCHAR(50)   ,
    Creditos           INT           ,
    ID_Departamento    INT           NOT NULL,
    PRIMARY KEY (ID_Curso),
    FOREIGN KEY (ID_Departamento) REFERENCES Departamentos(ID_Departamento)
);

CREATE TABLE Inscripciones (
    ID_Estudiante      INT           NOT NULL,
    ID_Curso           INT           NOT NULL,
    FechaInscripcion   DATE          ,
    PRIMARY KEY (ID_Estudiante, ID_Curso),
    FOREIGN KEY (ID_Estudiante) REFERENCES Estudiantes(ID_Estudiante),
    FOREIGN KEY (ID_Curso)      REFERENCES Cursos(ID_Curso)
);

CREATE TABLE Evaluaciones (
    ID_Estudiante      INT           NOT NULL,
    ID_Curso           INT           NOT NULL,
    Nota               DECIMAL(4,2)  ,
    FechaEvaluacion    DATE          ,
    PRIMARY KEY (ID_Estudiante, ID_Curso, FechaEvaluacion),
    FOREIGN KEY (ID_Estudiante) REFERENCES Estudiantes(ID_Estudiante),
    FOREIGN KEY (ID_Curso)      REFERENCES Cursos(ID_Curso)
);