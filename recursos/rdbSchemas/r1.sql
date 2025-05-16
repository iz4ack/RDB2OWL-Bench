CREATE TABLE Departments (
    DepartmentID      INT           NOT NULL AUTO_INCREMENT,
    FullName              VARCHAR(100),
    PRIMARY KEY (DepartmentID)
);

CREATE TABLE Students (
    StudentID         INT            NOT NULL AUTO_INCREMENT,
    NationalID        VARCHAR(20),
    FirstName         VARCHAR(100),
    LastName          VARCHAR(100),
    Email             VARCHAR(150),
    BirthDate         DATE,
    EnrollmentDate    DATE,
    PRIMARY KEY (StudentID),
    CONSTRAINT UQ_Student UNIQUE (NationalID)
);

CREATE TABLE Professors (
    ProfessorID       INT            NOT NULL AUTO_INCREMENT,
    FirstName         VARCHAR(100),
    LastName          VARCHAR(100),
    Email             VARCHAR(150),
    BirthDate         DATE,
    DepartmentID      INT            NOT NULL,
    PRIMARY KEY (ProfessorID),
    FOREIGN KEY (DepartmentID) REFERENCES Departments(DepartmentID),
    CONSTRAINT UQ_Professor UNIQUE (NationalID)
);

CREATE TABLE Courses (
    CourseID          INT            NOT NULL AUTO_INCREMENT,
    FullName              VARCHAR(100),
    CourseLevel             VARCHAR(50),
    Credits           INT,
    DepartmentID      INT            NOT NULL,
    PRIMARY KEY (CourseID),
    FOREIGN KEY (DepartmentID) REFERENCES Departments(DepartmentID)
);

CREATE TABLE Enrollments (
    StudentID         INT            NOT NULL,
    CourseID          INT            NOT NULL,
    EnrollmentDate    DATE,
    PRIMARY KEY (StudentID, CourseID),
    FOREIGN KEY (StudentID) REFERENCES Students(StudentID),
    FOREIGN KEY (CourseID)  REFERENCES Courses(CourseID)
);

CREATE TABLE Evaluations (
    StudentID         INT            NOT NULL,
    CourseID          INT            NOT NULL,
    Grade             DECIMAL(4,2),
    EvaluationDate    DATE,
    PRIMARY KEY (StudentID, CourseID, EvaluationDate),
    FOREIGN KEY (StudentID) REFERENCES Students(StudentID),
    FOREIGN KEY (CourseID)  REFERENCES Courses(CourseID)
);
