CREATE TABLE Departamento (
    ID_Departamento INT NOT NULL AUTO_INCREMENT,
    Nombre          VARCHAR(100) NOT NULL,
    Presupuesto     DECIMAL(12,2) NOT NULL,
    ID_Jefe         INT NOT NULL,
    PRIMARY KEY (ID_Departamento),
    FOREIGN KEY (ID_Jefe) REFERENCES EmpleadoJefe(ID_Empleado)
);

CREATE TABLE Proyecto (
    ID_Proyecto     INT NOT NULL AUTO_INCREMENT,
    Nombre          VARCHAR(100) NOT NULL,
    Presupuesto     DECIMAL(12,2) NOT NULL,
    ID_Departamento INT NOT NULL,
    PRIMARY KEY (ID_Proyecto),
    FOREIGN KEY (ID_Departamento) REFERENCES Departamento(ID_Departamento)
);

CREATE TABLE Empleado (
    ID_Empleado     INT NOT NULL AUTO_INCREMENT,
    Nombre          VARCHAR(100) NOT NULL,
    Apellido        VARCHAR(100) NOT NULL,
    ID_Departamento INT NOT NULL,
    PRIMARY KEY (ID_Empleado),
    FOREIGN KEY (ID_Departamento) REFERENCES Departamento(ID_Departamento),
    FOREIGN KEY (ID_Conyugue) REFERENCES Empleado(ID_Empleado)
        ON DELETE SET NULL
        ON UPDATE CASCADE
);

CREATE TABLE EmpleadoTecnico (
    ID_Empleado INT NOT NULL,
    Especialidad VARCHAR(100) NOT NULL,
    PRIMARY KEY (ID_Empleado),
    FOREIGN KEY (ID_Empleado) REFERENCES Empleado(ID_Empleado)
);

CREATE TABLE EmpleadoAdministrativo(
    ID_Empleado INT NOT NULL,
    HorasTrabajadas INT NOT NULL,
    PRIMARY KEY (ID_Empleado),
    FOREIGN KEY (ID_Empleado) REFERENCES Empleado(ID_Empleado)
);

CREATE TABLE EmpleadoJefe (
    ID_Empleado INT NOT NULL,
    antiguedadAños  INT NOT NULL,
    PRIMARY KEY (ID_Empleado),
    FOREIGN KEY (ID_Empleado) REFERENCES Empleado(ID_Empleado),
    FOREIGN KEY (ID_Departamento) REFERENCES Departamento(ID_Departamento)
);

CREATE TABLE Tarea (
    ID_Tarea        INT NOT NULL AUTO_INCREMENT,
    Descripcion     VARCHAR(100) NOT NULL,
    PRIMARY KEY (ID_Tarea),
    FOREIGN KEY (ID_Proyecto) REFERENCES Proyecto(ID_Proyecto)
);

CREATE TABLE Asignacion (
    ID_Empleado   INT NOT NULL,
    ID_Proyecto   INT NOT NULL,
    ID_Tarea      INT NOT NULL,
    FechaInicio   DATE NOT NULL,
    PRIMARY KEY (ID_Empleado, ID_Proyecto, ID_Tarea),
    FOREIGN KEY (ID_Empleado) REFERENCES Empleado(ID_Empleado),
    FOREIGN KEY (ID_Proyecto) REFERENCES Proyecto(ID_Proyecto),
    FOREIGN KEY (ID_Tarea) REFERENCES Tarea(ID_Tarea)
);
