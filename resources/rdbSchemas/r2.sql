CREATE TABLE Department (
    DepartmentID   INT           NOT NULL AUTO_INCREMENT,
    DepartmentName           VARCHAR(100),
    Budget         DECIMAL(12,2),
    ManagerID      INT          NOT NULL,
    PRIMARY KEY (DepartmentID),
    FOREIGN KEY (ManagerID) REFERENCES Manager(EmployeeID)
);

CREATE TABLE Project (
    ProjectID      INT           NOT NULL AUTO_INCREMENT,
    ProjectName           VARCHAR(100)  ,
    Budget         DECIMAL(12,2) ,
    DepartmentID   INT           NOT NULL,
    PRIMARY KEY (ProjectID),
    FOREIGN KEY (DepartmentID) REFERENCES Department(DepartmentID)
);

CREATE TABLE Employee (
    EmployeeID     INT           NOT NULL AUTO_INCREMENT,
    FirstName      VARCHAR(100)  ,
    LastName       VARCHAR(100)  ,
    DepartmentID   INT           NOT NULL,
    PRIMARY KEY (EmployeeID),
    FOREIGN KEY (DepartmentID) REFERENCES Department(DepartmentID)
);

CREATE TABLE TechnicalEmployee (
    EmployeeID     INT           NOT NULL,
    Specialty      VARCHAR(100)  ,
    PRIMARY KEY (EmployeeID),
    FOREIGN KEY (EmployeeID) REFERENCES Employee(EmployeeID)
);

CREATE TABLE AdministrativeEmployee (
    EmployeeID     INT           NOT NULL,
    HoursWorked    INT           ,
    PRIMARY KEY (EmployeeID),
    FOREIGN KEY (EmployeeID) REFERENCES Employee(EmployeeID)
);

CREATE TABLE Manager (
    EmployeeID       INT           NOT NULL,
    YearsOfService   INT           ,
    PRIMARY KEY (EmployeeID),
    FOREIGN KEY (EmployeeID)   REFERENCES Employee(EmployeeID)
);

CREATE TABLE Task (
    TaskID          INT           NOT NULL AUTO_INCREMENT,
    Description     VARCHAR(100)  ,
    PRIMARY KEY (TaskID)
);

CREATE TABLE Assignment (
    EmployeeID      INT           NOT NULL,
    ProjectID       INT           NOT NULL,
    TaskID          INT           NOT NULL,
    StartDate       DATE          ,
    PRIMARY KEY (EmployeeID, ProjectID, TaskID),
    FOREIGN KEY (EmployeeID)  REFERENCES Employee(EmployeeID),
    FOREIGN KEY (ProjectID)   REFERENCES Project(ProjectID),
    FOREIGN KEY (TaskID)      REFERENCES Task(TaskID)
);
