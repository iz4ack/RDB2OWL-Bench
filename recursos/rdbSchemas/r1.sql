CREATE TABLE Departments (
    DepartmentID      INT           NOT NULL AUTO_INCREMENT,
    DepartmentName              VARCHAR(100),
    PRIMARY KEY (DepartmentID)
);

CREATE TABLE Students (
    DNI         VARCHAR(9)            NOT NULL ,
    FirstName         VARCHAR(100),
    LastName          VARCHAR(100),
    Email             VARCHAR(150),
    BirthDate         DATE,
    EnrollmentDate    DATE,
    PRIMARY KEY (StudentID)
);

CREATE TABLE Professors (
    DNI         VARCHAR(9)            NOT NULL ,
    FirstName         VARCHAR(100),
    LastName          VARCHAR(100),
    Email             VARCHAR(150),
    BirthDate         DATE,
    DepartmentID      INT            NOT NULL,
    PRIMARY KEY (ProfessorID),
    FOREIGN KEY (DepartmentID) REFERENCES Departments(DepartmentID)
);

CREATE TABLE Courses (
    CourseID          INT            NOT NULL AUTO_INCREMENT,
    CourseName              VARCHAR(100),
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
