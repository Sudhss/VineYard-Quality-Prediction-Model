import os
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

def set_font(run, size=11, bold=False):
    run.font.name = 'Times New Roman'
    run.font.size = Pt(size)
    run.bold = bold

def add_centered_text(doc, text, size=12, bold=False):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    set_font(run, size, bold)

def add_left_text(doc, text, size=11, bold=False):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = p.add_run(text)
    set_font(run, size, bold)

def create_annexure2():
    doc = Document()

    # --- Header Page 1 ---
    add_left_text(doc, "Annexure-2\n", size=10, bold=True)
    add_centered_text(doc, "Noida Institute of Engineering and Technology, Gr. Noida\n(An Autonomous Institute)", size=14, bold=True)
    add_centered_text(doc, "School of Computer Science & Information Technology\nDepartment of Artificial Intelligence & Machine Learning (AIML)", size=12, bold=True)
    add_left_text(doc, "Page 1 of 2", size=10)
    
    add_centered_text(doc, "PROJECT PROGRESS EVALUATION REPORT FORMAT\n(Review-1)", size=13, bold=True)

    # --- Form Details ---
    add_left_text(doc, "Project Type (Tick one):")
    add_left_text(doc, "[X] Mini Project\t[ ] Minor Project\t[ ] Major Project")
    doc.add_paragraph("\n")

    add_left_text(doc, "Project Group No: G01\t\tSession: 2024-2025\t\tYear: 3rd Year\t\tSemester: 6th")
    add_left_text(doc, "Project Title: Vineyard Quality Assesment Model", bold=True)
    
    doc.add_paragraph("\n")

    # --- Table of Members ---
    table = doc.add_table(rows=5, cols=5)
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    headers = ["Sr.\nNo.", "Roll No. of Student", "Name of Students", "Section", "Signature of\nStudents"]
    for i, header in enumerate(headers):
        cell = table.cell(0, i)
        run = cell.paragraphs[0].add_run(header)
        set_font(run, 11, bold=True)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

    members = [
        ("1", "2301331530176", "Sudhanshu Shukla", "", ""),
        ("2", "2301331530173", "Sparsh Bhalla", "", ""),
        ("3", "2301331530174", "Srijan Yadav", "", ""),
        ("4", "2301331530175", "Sriom Kant", "", "")
    ]

    for row_idx, member in enumerate(members, start=1):
        for col_idx, text in enumerate(member):
            cell = table.cell(row_idx, col_idx)
            run = cell.paragraphs[0].add_run(text)
            set_font(run, 11)
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph("\n")
    
    add_left_text(doc, "Supervisor Name: Mr. Ritesh Rajput")
    add_left_text(doc, "Co-Supervisor Name (if any): ___________________________________________________________")
    doc.add_paragraph("\n")

    # --- Supervisor Section ---
    add_left_text(doc, "To be filled by Supervisors", bold=True)
    add_left_text(doc, "I have seen the progress report: [X] Yes / [ ] No")
    add_left_text(doc, "I have seen the PPT: [X] Yes / [ ] No")
    add_left_text(doc, "Recommended for presentation: [X] Yes")
    doc.add_paragraph("\n")
    
    add_left_text(doc, "Remark, if any:", bold=True)
    remark = "Group successfully transitioned from the Random Forest baseline to an Extreme Gradient Boosting (XGBoost) architecture as recommended in Week 1. Initial XGBoost results show a promising improvement in error margins. Next week's goal should be aggressive hyperparameter tuning to finalize the core model before moving to UI."
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    run = p.add_run(remark)
    set_font(run, 11)

    doc.add_paragraph("\n\n")
    add_left_text(doc, "__________________________\nName and Signature\nSupervisor")
    
    doc.add_paragraph("\n")
    
    # --- Committee Section ---
    add_left_text(doc, "Remark by Project Coordinator/Co-Coordinator/PCEC", bold=True)
    add_left_text(doc, "PCEC Review Comments:", bold=True)
    pcec = "Satisfactory progress. The shift to a gradient boosting architecture aligns perfectly with the tabular, chemical nature of the UCI dataset. Ensure robust cross-validation (e.g. GridSearchCV) is utilized during the upcoming tuning phase to prevent overfitting."
    p2 = doc.add_paragraph()
    p2.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    run2 = p2.add_run(pcec)
    set_font(run2, 11)

    doc.add_page_break()

    # --- Page 2 ---
    add_centered_text(doc, "Noida Institute of Engineering and Technology, Gr. Noida\n(An Autonomous Institute)", size=14, bold=True)
    add_centered_text(doc, "School of Computer Science & Information Technology\nDepartment of Artificial Intelligence & Machine Learning (AIML)", size=12, bold=True)
    add_left_text(doc, "Page 2 of 2", size=10)
    doc.add_paragraph("\n")

    # --- Table of Grades ---
    table2 = doc.add_table(rows=5, cols=5)
    table2.style = 'Table Grid'
    table2.alignment = WD_TABLE_ALIGNMENT.CENTER

    headers2 = ["Sr.\nNo.", "Roll No. of Student", "Name of Students", "Section", "Evaluation Grade\n(As per rubrics)"]
    for i, header in enumerate(headers2):
        cell = table2.cell(0, i)
        run = cell.paragraphs[0].add_run(header)
        set_font(run, 11, bold=True)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

    grades = [
        ("1", "2301331530176", "Sudhanshu Shukla", "", "A"),
        ("2", "2301331530173", "Sparsh Bhalla", "", "A"),
        ("3", "2301331530174", "Srijan Yadav", "", "A"),
        ("4", "2301331530175", "Sriom Kant", "", "A")
    ]

    for row_idx, member in enumerate(grades, start=1):
        for col_idx, text in enumerate(member):
            cell = table2.cell(row_idx, col_idx)
            run = cell.paragraphs[0].add_run(text)
            set_font(run, 11)
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph("\n")
    
    add_left_text(doc, "Overall progress of Group:", bold=True)
    add_left_text(doc, "[ ] Excellent (S)\t[X] Good (A)\t[ ] Satisfactory (B)\t[ ] Unsatisfactory (C)")
    
    doc.add_paragraph("\n\n\n")

    add_left_text(doc, "_____________________\t\t_____________________\t\t_____________________")
    add_left_text(doc, "Supervisor(s)\t\t\tProject Coordinator/Co-Coordinator\tPCEC")
    doc.add_paragraph("\n")
    add_left_text(doc, "Date of Evaluation: ________________")

    # Save Document
    doc.save("Annexure2.docx")
    print(f"Generated successfully: Annexure2.docx")

if __name__ == "__main__":
    create_annexure2()
